import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import GenerationConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import math
from datasets import load_dataset
import numpy as np
import re
from vllm import LLM, SamplingParams

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
    model = LLM(
        model=args.model_id,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=4,
        disable_mm_preprocessor_cache=True
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2048
    )


    questions = list(load_dataset("lmms-lab/MMVet", split="test"))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    final_response = []
    batch_size = args.batch_size
    for chunk in tqdm(split_list(questions, math.ceil(len(questions) / batch_size))):
        batch_inputs = []
        for data in chunk:
            img = data['image']
            if img.height < 28 or img.width < 28:
                print('Skip sample')
                continue

            question = data['question']
            qs = question
            images = data['image']
            qs = qs + instruct_prompt

            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            # "image": 'image.png',
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {"type": "text", "text": qs},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_inputs.append({"prompt": text, "multi_modal_data": {"image": images}})

        outputs = model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        for out, data in zip(outputs, chunk):
            data['response'] = out.outputs[0].text
            data['image'] = []
            final_response.append(data)

    dump_to_jsonl(final_response, args.answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
