import re
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import logging
from datetime import datetime

# Set up logging
def setup_logging(rank):
    log_filename = f'inference_rank_{rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for rank {rank}")
    return logger

# Load dataset (global, as it's lightweight and serializable)
ds = load_dataset("vikhyatk/CountBenchQA")['test']

# SYSTEM_PROMPT remains the same
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Function to prepare inputs for a batch
def prep_inputs_batch(examples, rank, logger, processor):
    logger.info(f"Preparing inputs for batch on rank {rank}")
    queries = [
        f"{example['question']}\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "When thinking, please count each object you see out loud and describe which part of the image it is in."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>NUMBER</answer>\n"
        "NUMBER must be an integer consisting of digits, e.g. 11, not eleven.\n"
        "Please strictly follow the format."
        for example in examples
    ]

    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example['image']},
                    {"type": "text", "text": query},
                ]
            }
        ]
        for example, query in zip(examples, queries)
    ]

    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    images = [process_vision_info(messages)[0] for messages in messages_batch]
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    logger.info(f"Inputs prepared and moved to cuda:{rank}")
    return inputs.to(f"cuda:{rank}")

# Inference function for batches
def inference_batch(model, inputs, logger, rank, processor):
    logger.info(f"Starting inference on rank {rank}")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    responses = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    logger.info(f"Inference completed on rank {rank}")
    return responses

# Evaluation function for batches
def eval_batch(responses, examples, logger, rank):
    logger.info(f"Evaluating responses on rank {rank}")
    preds, corrects = [], []
    for response, example in zip(responses, examples):
        content_match = re.search(r'<answer>(.*?)</answer>', response)
        pred = int(content_match.group(1).strip() if content_match else response.strip())
        correct = pred == example['number']
        preds.append(pred)
        corrects.append(correct)
    batch_accuracy = sum(corrects) / len(corrects)
    logger.info(f"Batch accuracy on rank {rank}: {batch_accuracy:.4f}")
    return preds, corrects

# Multi-GPU setup with DistributedDataParallel
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(run_process, args=(world_size, ds), nprocs=world_size)
    else:
        run_process(0, world_size, ds)

def run_process(rank, world_size, dataset):
    # Setup logging for this rank
    logger = setup_logging(rank)

    # Initialize model and processor inside the process
    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto").eval()
    logger.info(f"Model and processor initialized on rank {rank}")

    if world_size > 1:
        setup_distributed(rank, world_size)
        logger.info(f"Distributed setup completed for rank {rank} in world size {world_size}")

    # Move model to the correct GPU
    model.to(f"cuda:{rank}")
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    logger.info(f"Model moved to cuda:{rank}")

    # Create DataLoader for batching
    batch_size = 4  # Adjust based on GPU memory
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"DataLoader created with batch size {batch_size} on rank {rank}")

    n_correct = 0
    total = 0

    # Enhanced tqdm with dynamic updates
    with tqdm(total=len(dataloader), desc=f"Rank {rank} Progress", disable=(rank != 0)) as pbar:
        for i, batch in enumerate(dataloader):
            inputs = prep_inputs_batch(batch, rank, logger, processor)
            responses = inference_batch(model, inputs, logger, rank, processor)
            preds, corrects = eval_batch(responses, batch, logger, rank)
            n_correct += sum(corrects)
            total += len(corrects)

            # Update tqdm with running stats
            running_accuracy = n_correct / total if total > 0 else 0
            pbar.set_postfix({
                'batch': i + 1,
                'correct': n_correct,
                'total': total,
                'acc': f"{running_accuracy:.4f}"
            })
            pbar.update(1)

    logger.info(f"Rank {rank} processed {total} examples with {n_correct} correct")

    # Gather results from all GPUs
    if world_size > 1:
        n_correct_tensor = torch.tensor(n_correct, device=f"cuda:{rank}")
        total_tensor = torch.tensor(total, device=f"cuda:{rank}")
        dist.reduce(n_correct_tensor, dst=0)
        dist.reduce(total_tensor, dst=0)

        if rank == 0:
            n_correct = n_correct_tensor.item()
            total = total_tensor.item()
            accuracy = n_correct / total
            logger.info(f"Final aggregated accuracy across all ranks: {accuracy:.4f}")
            print(f'{accuracy=}')

        dist.destroy_process_group()
        logger.info(f"Distributed process group destroyed on rank {rank}")
    else:
        accuracy = n_correct / total
        logger.info(f"Final accuracy on single GPU (rank {rank}): {accuracy:.4f}")
        print(f'{accuracy=}')

    cleanup()

if __name__ == "__main__":
    main()
