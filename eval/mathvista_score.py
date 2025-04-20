import numpy as np
import json

import argparse

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
        if data['question_type']== 'multi_choice':
            index = data['choices'].index(data['answer'])
            if index ==0:
                data['c_answer'] = 'A'
                data['num_answer'] = data['choices'][0]
            elif index==1:
                data['c_answer'] = 'B'
                data['num_answer'] = data['choices'][1]
            elif index==2:
                data['c_answer'] = 'C'
                data['num_answer'] = data['choices'][2]
            elif index==3:
                data['c_answer'] = 'D'
                data['num_answer'] = data['choices'][3]
            elif index==4:
                data['c_answer'] = 'E'
                data['num_answer'] = data['choices'][4]
            elif index==5:
                data['c_answer'] = 'F'
                data['num_answer'] = data['choices'][5]
            elif index==6:
                data['c_answer'] = 'G'
                data['num_answer'] = data['choices'][6]

    total = 0
    correct = 0
    for data in datas:
        if data['question_type'] == 'multi_choice':
            if data['c_answer'] in extract_answer(data['response']) or data['num_answer'] in extract_answer(
                    data['response']):
                correct += 1
        else:
            if data['answer'] in extract_answer(data['response']):
                correct += 1
        total += 1

    print('MathVista ACC:', correct / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    main(args)