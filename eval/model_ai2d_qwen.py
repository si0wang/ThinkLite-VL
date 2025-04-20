import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from PIL import Image
import math
from datasets import load_dataset
import numpy as np
import re

instruct_prompt = r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i] for i in range(k*chunk_size, np.min([(k+1)*chunk_size, len(lst)]), 1)]


def eval_model(args):
    # Model
    device = "cuda:{}".format(args.gpu_id)
    # Model
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device)

    greedy_generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=1024
    )
    questions = load_dataset("lmms-lab/ai2d", split="test")
    questions_chunk = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    final_response = []
    for data in tqdm(questions_chunk):
        try:
            question = data['question']
    
            images = data['image']
    
            question += " Options:"
            for i in range(len(data["options"])):
                option = data["options"][i]
                question += f"\n{chr(ord('A')+i)}. {option}"
            qs = question + f"\nAnswer with the option's letter from the given choices."
            qs = qs + instruct_prompt
    
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": qs},
                    ],
                },
            ]
            prompts = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device)
    
            output = model.generate(**inputs, generation_config=greedy_generation_config,
                                    tokenizer=processor.tokenizer, max_new_tokens=4096)
            decodes = processor.decode(output[0], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
    
            data['response'] = decodes
            data['image'] = []
            final_response.append(data)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    dump_to_jsonl(final_response, args.answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
