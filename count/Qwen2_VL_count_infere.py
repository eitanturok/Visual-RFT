import logging, functools, re
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

mp.set_start_method('spawn', force=True)  # Set this immediately after imports

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

dataset_path = "vikhyatk/CountBenchQA"
model_path = "Qwen/Qwen2-VL-2B-Instruct"
processor_path = "Qwen/Qwen2-VL-2B-Instruct"

# https://github.com/eitanturok/Visual-RFT/blob/4632f4565361bf40ebeb7cf55aaa0118b6376b26/lisa_evaluation/Qwen2_VL_lisa_infere.py#L17C1-L22C2
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def prep_inputs(processor, example, device):
  query = (
      f"{example['question']}\n"
      "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
      "When thinking, please count each object you see out loud and describe which part of the image it is in."
      "The output answer format should be as follows:\n"
      "<think> ... </think> <answer>NUMBER</answer>\n"
      "NUMBER must be an integer consisting of digits, e.g. 11, not eleven.\n"
      "Please strictly follow the format."
  )

  messages = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {
          "role": "user",
          "content": [
              {"type": "image", "image": example['image']},
              # add <image>/n in evals: lvis, coco
              # https://github.com/eitanturok/Visual-RFT/blob/4632f4565361bf40ebeb7cf55aaa0118b6376b26/lvis_evaluation/Qwen2_VL_lvis_infere.py#L256
              # {"type": "text", "text": '<image>\n' + query}, #
              {"type": "text", "text": query},
          ]
      }
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  images, _ = process_vision_info(messages)
  inputs = processor(
      text=[text],
      images=images,
      padding=True,
      return_tensors="pt",
  )
  return inputs.to(device)

def inference(model, processor, inputs):
  generated_ids = model.generate(**inputs, max_new_tokens=1024)
  generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
  response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  return response


def eval(response, example):
  content_match = re.search(r'<answer>(.*?)</answer>', response)
  pred = int(content_match.group(1).strip() if content_match else response.strip())
  correct = pred == example['number']
  return pred, correct


def run(rank, world_size, dataset_split):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(processor_path)

    model = model.to(torch.device(rank))
    model.eval()

    n_correct, predictions = 0, []
    for example in dataset_split(rank):
        logger.info(f"Rank:\n{rank}\n\nType:\n{type(example)}\n\nExample:\n{example}")
        inputs = prep_inputs(processor, example, model.device)
        response = inference(model, processor, inputs)
        pred, correct = eval(response, example)
        n_correct += correct
        predictions.append(pred)

    return n_correct, predictions

def main():
    ds = load_dataset(dataset_path)['test']
    multiprocess = torch.cuda.device_count() >= 2
    logger.info(f'Started generation with {torch.cuda.device_count()} GPUs')

    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus

        chunk_size = len(ds) // world_size
        dataset_splits = [ds[i * chunk_size:(i + 1) * chunk_size] if i < world_size - 1 else ds[i * chunk_size:] for i in range(world_size)]

        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, dataset_split=functools.partial(dataset_splits.__getitem__))
            result_lists = pool.map(func, range(world_size))

        global_n_correct = sum([result_lists[i][0] for i in range(world_size)])
        global_n_error = len(ds) - global_n_correct
        global_results = [result_lists[i][1] for i in range(world_size)]

        logger.info('Error number: ' + str(global_n_error))
        logger.info('Total Right Number: ' + str(global_n_correct))
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()
