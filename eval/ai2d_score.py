import argparse
from datasets import load_dataset
import json
import re

def extract_answer(text):
    response = text.split('assistant\n')[1]
    if 'boxed{' in response:
        final_answer = response.split('boxed{')[-1].split('}')[0]
        return final_answer
    else:
        return 'Unknown'

def main(args):
    datas = []
    with open(args.data_path, 'r') as files:
        for line in files:
            datas.append(json.loads(line))

    for data in datas:
        if data['answer'] == '0':
            data['answer'] = 'A'
            data['num_answer'] = data['options'][0]
        if data['answer'] == '1':
            data['answer'] = 'B'
            data['num_answer'] = data['options'][1]
        if data['answer'] == '2':
            data['answer'] = 'C'
            data['num_answer'] = data['options'][2]
        if data['answer'] == '3':
            data['answer'] = 'D'
            data['num_answer'] = data['options'][3]

    total = 0
    correct = 0
    for data in datas:
        if data['answer'] in extract_answer(data['response']) or data['num_answer'] in extract_answer(data['response']):
            correct += 1
        total += 1

    print('AI2D ACC:', correct / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    main(args)