import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from icecream import ic, install
install()

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
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def inference(model, processor, inputs):
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def evaluate(response, example):
    try:
        content_match = re.search(r'<answer>(.*?)</answer>', response)
        pred = int(content_match.group(1).strip() if content_match else response.strip())
        correct = pred == example['number']
    except Exception as e:
        print(f'{response=} failed with exception {e=}')
        correct, pred = 0, None
    return pred, correct

def main():
    # load dataset, processor, and model
    device = "cuda"
    model_path, dataset_path = 'Qwen/Qwen2-VL-2B-Instruct', 'eturok/CountBenchQA'
    ds = load_dataset(dataset_path)['test']
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map=device)
    model.eval()

    responses, predictions, oom_examples, n_correct = [], [], [], 0
    for i, example in enumerate(tqdm(ds)):
        inputs = prep_inputs(processor, example, device)
        if inputs['input_ids'].shape[1] > 1500:
            oom_examples.append(i)
            print(f'skipping {i}th example because it is too big')
            continue
        response = inference(model, processor, inputs)
        pred, correct = evaluate(response, example)

        responses.append(response)
        predictions.append(pred)
        n_correct += correct

    accuracy = n_correct / len(ds)
    print(f'{accuracy=}')




if __name__ == '__main__':
    main()

