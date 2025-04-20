import argparse
import pandas as pd
import re
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, GenerationConfig
import torch
from PIL import Image
import requests
import json
import os
import math
from tqdm import tqdm
from io import BytesIO
import base64
import ast

def extract_answer(text):
    response = text.split('assistant\n')[1]
    if 'boxed{' in response:
        final_answer = response.split('boxed{')[-1].split('}')[0]
        return final_answer
    else:
        return response.split('\n')[-1]

def main(args):
    datas = []
    with open(args.data_path, 'r') as files:
        for line in files:
            datas.append(json.loads(line))

    for data in datas:
        if data['question_type'] == 'multiple-choice':
            options = ast.literal_eval(data['options'])
            if data['answer'] == 'A':
                data['num_answer'] = options[0]
            elif data['answer'] == 'B':
                data['num_answer'] = options[1]
            elif data['answer'] == 'C':
                data['num_answer'] = options[2]
            elif data['answer'] == 'D':
                data['num_answer'] = options[3]
            elif data['answer'] == 'E':
                data['num_answer'] = options[4]
            elif data['answer'] == 'F':
                data['num_answer'] = options[5]
            elif data['answer'] == 'G':
                data['num_answer'] = options[6]
            elif data['answer'] == 'H':
                data['num_answer'] = options[7]
            elif data['answer'] == 'I':
                data['num_answer'] = options[8]

    total = 0
    correct = 0
    for data in datas:
        total += 1
        if data['question_type'] == 'multiple-choice':
            if data['answer'] in extract_answer(data['response']) or data['num_answer'] in extract_answer(
                    data['response']):
                correct += 1
        else:
            if data['answer'] in extract_answer(data['response']) \
                    or data['answer'].lower() in extract_answer(data['response']).lower() \
                    or extract_answer(data['response']) in data['answer'] \
                    or extract_answer(data['response']).lower() in data['answer'].lower():
                correct += 1

    print('MMMU ACC:', correct / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/blob/v-xiyaowang/v-xiyaowang/llava_prm/llava_prm/eval_files/mathvision/answers/Qwen2.5-VL-7B-MCTSUnsolved_Node1_Rollout32-2025-03-22-22-45/global_step_80.jsonl")
    args = parser.parse_args()

    main(args)