import logging, functools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

import torch

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run(rank, world_size):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(ori_processor_path)

    model = model.to(torch.device(rank))
    model = model.eval()

def main():
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        global_count_error = 0
        global_count_right = 0
        global_results = []
        for i in range(world_size):
            global_count_error += int(result_lists[i][0])
            global_count_right = global_count_right + result_lists[i][1]

        logger.info('Error number: ' + str(global_count_error))
        logger.info('Total Right Number: ' + str(global_count_right))
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()
