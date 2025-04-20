import argparse

from datasets import load_dataset
import math
import numpy as np
import json

def extract_boxed_content(s):
    keyword = r'\boxed{'
    start = s.find(keyword)
    if start == -1:
        return None
    start_brace = s.find('{', start)
    if start_brace == -1:
        return None
    count = 1
    i = start_brace + 1
    while i < len(s) and count > 0:
        if s[i] == '{':
            count += 1
        elif s[i] == '}':
            count -= 1
        i += 1
    return s[start_brace+1:i-1]

def extract_answer(text):
    response = text.split('assistant\n')[1]
    if 'boxed{' in response:
        final_answer = extract_boxed_content(response)
        return final_answer
    else:
        return 'Unknown'

def main(args):
    datas = []
    with open(args.data_path, 'r') as files:
        for line in files:
            datas.append(json.loads(line))

    for data in datas:
        if 'A:' in data['question']:
            if data['answer'] == 'A':
                data['num_answer'] = data['question'].split('A: ')[1].split(',')[0]
            elif data['answer'] == 'B':
                data['num_answer'] = data['question'].split('B: ')[1].split(',')[0]
            elif data['answer'] == 'C':
                data['num_answer'] = data['question'].split('C: ')[1].split(',')[0]
            elif data['answer'] == 'D':
                data['num_answer'] = data['question'].split('D: ')[1].split(',')[0]

    total = 0
    correct = 0
    for data in datas:
        if 'A:' in data['question']:
            if data['answer'] in extract_answer(data['response']) or data['num_answer'] in extract_answer(
                    data['response']):
                correct += 1
        else:
            if data['answer'] in extract_answer(data['response']):
                correct += 1
        total += 1

    print('MMStar ACC:', correct / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    main(args)