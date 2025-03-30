import functools, os, logging
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# **** Setup ****

def setup_logging(rank):
    global logger
    log_filename = f'logs/inference_rank_{rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12347"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()

# **** Data *****

# https://github.com/eitanturok/Visual-RFT/blob/4632f4565361bf40ebeb7cf55aaa0118b6376b26/lisa_evaluation/Qwen2_VL_lisa_infere.py#L17C1-L22C2
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Step 2: Updated CustomHFDataset with prep_inputs logic
class CustomHFDataset(Dataset):
    def __init__(self, hf_dataset, processor, system_prompt=SYSTEM_PROMPT):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        question = example["question"]
        image = example["image"]  # Assuming PIL image or tensor

        # Build query as in prep_inputs
        query = (
            f"{question}\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "When thinking, please count each object you see out loud and describe which part of the image it is in.\n"
            "The output answer format should be as follows:\n"
            "<think> ... </think> <answer>NUMBER</answer>\n"
            "NUMBER must be an integer consisting of digits, e.g. 11, not eleven.\n"
            "Please strictly follow the format."
        )

        # Construct messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},  # Add '<image>\n' if needed per your comment
                ]
            }
        ]

        # Apply chat template (untokenized)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process vision info (extract images)
        images, _ = process_vision_info(messages)

        # Tokenize and process text + images
        inputs = self.processor(
            text=[text],
            images=images,
            padding=False,  # Defer padding to collate_fn
            return_tensors="pt",
        )

        return inputs | {'label': example['number']}

def pad_collate_fn(batch, processor):

    logger.info(f'{batch[0].keys()=}')

    # Pad text sequences
    input_ids = [item["input_ids"].squeeze() for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)

    # Pad attention mask
    attention_mask = [item["attention_mask"].squeeze() for item in batch]
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Pad images to max size in batch
    pixel_values = [item["pixel_values"] for item in batch]
    max_height = max(img.shape[-2] for img in pixel_values)  # Assuming [C, H, W]
    max_width = max(img.shape[-1] for img in pixel_values)
    pixel_values_padded = [
        torch.nn.functional.pad(img, (0, max_width - img.shape[-1], 0, max_height - img.shape[-2]))
        for img in pixel_values
    ]
    pixel_values_padded = torch.stack(pixel_values_padded)

    return {
        "input_ids": input_ids_padded,
        "pixel_values": pixel_values_padded,
        "labels": torch.tensor([item["label"] for item in batch]),
        "attention_mask": attention_mask_padded,
        "image_grid_thw": torch.stack([item["image_grid_thw"].squeeze() for item in batch]) # is this for padding?
    }

def get_datloader(processor, batch_size, rank, world_size):
    hf_dataset = load_dataset("vikhyatk/CountBenchQA")['test']
    dataset = CustomHFDataset(hf_dataset, processor)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=functools.partial(pad_collate_fn, processor=processor),
        # num_workers=4
    )
    return dataloader

def inference(dataloader, model, processor, rank):
    all_responses = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(f"cuda:{rank}") for k, v in batch.items()}
        generated_ids = model.generate(**batch, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch['input_ids'], generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        all_responses.extend(response)
    return all_responses


def main(rank: int, world_size: int, batch_size: int):
    logger = setup_logging(rank)
    logger.info(f'Setup {rank=} with {world_size=}')
    try:
        if world_size > 1: ddp_setup(rank, world_size)
        model_path = "Qwen/Qwen2-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
        dataloader = get_datloader(processor, batch_size, rank, world_size)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto").to(f"cuda:{rank}")
        # model = DDP(model, device_ids=[rank])
        model.eval()
        all_responses = inference(dataloader, model, processor, rank)
        logger.info(all_responses)
    finally:
        if world_size > 1:
            ddp_cleanup()
            logger.info('Cleaned up.')

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 8
    mp.spawn(main, args=(world_size, batch_size), nprocs=world_size)
    main()
