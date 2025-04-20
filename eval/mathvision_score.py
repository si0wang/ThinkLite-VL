import argparse
import json
import torch
import pandas as pd
from tqdm import tqdm
from utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number

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

    id_raw = {example['id']: example for example in datas}
    for line in tqdm(datas, desc='gen_correct'):
        raw_exampe = id_raw[line['id']]

        gt_answer = str(raw_exampe['answer'])
        if len(raw_exampe['options']) > 0:
            gt_answer_value = raw_exampe['options'][ord(gt_answer) - ord('A')]
        else:
            gt_answer_value = ''

        if 'model_answer' not in line or regen_answer:
            model_answer = line['response'].strip()
            for c in 'ABCDE':
                if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(
                        f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                    model_answer = c
            if is_number(model_answer.split('is ')[-1].rstrip('.')):
                model_answer = model_answer.split('is ')[-1].rstrip('.')
            if 'oxed{' not in model_answer:
                for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
                    flag = flag.replace('the', 'The')
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
            elif model_answer.count('oxed{') > 1:
                model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]

            model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)',
                                                                                                          'c').replace(
                '(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace(
                '{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
            line['model_answer'] = model_answer
        else:
            model_answer = line['model_answer']
        line['gt_answer_value'] = gt_answer_value
        line['correct'] = (is_equal(gt_answer, model_answer)
                           or is_equal(gt_answer_value,model_answer)
                           or model_answer.lower() in gt_answer_value.lower()
                           or model_answer.lower() in gt_answer_value.replace(" ", "").lower())

    correct = 0

    for data in datas:
        if data['correct'] == True:
            correct += 1

    print('MathVision ACC:', correct / len(datas))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    main(args)