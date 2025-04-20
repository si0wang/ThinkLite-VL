import argparse
from datasets import load_dataset
import json
import re

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

def normalize_expr(expr_str):
    expr_str = expr_str.replace(r"\sqrt", "sqrt")
    expr_str = expr_str.replace("{", "(").replace("}", ")")
    expr_str = expr_str.replace(" ", "")
    expr_str = re.sub(r'(\d)(sqrt)', r'\1*\2', expr_str)
    return expr_str

def normalize_sqrt(expr_str):
    return expr_str.replace("√", "sqrt")

def normalize_space(expr_str):
    return expr_str.replace(" ", "")

def extract_answer(text):
    response = text.split('assistant\n')[1]
    if 'boxed{' in response:
        final_answer = extract_boxed_content(response)
        return final_answer
    else:
        return response.split('\n')[-1]

def main(args):
    datas = []
    with open(args.data_path, 'r') as files:
        for line in files:
            datas.append(json.loads(line))

    for data in datas:
        if data['question_type'] == 'multi-choice':
            if 'Choices:\n' in data['question_for_eval']:
                if data['answer'] == 'A' or data['answer'] == '(A)':
                    if 'A.' in data['question_for_eval'].split('Choices:\n')[-1].split('\n')[0]:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[0].split('A.')[-1]
                    else:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[0].split('A:')[-1]
                elif data['answer'] == 'B' or data['answer'] == '(B)':
                    if 'B.' in data['question_for_eval'].split('Choices:\n')[-1].split('\n')[1]:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[1].split('B.')[-1]
                    else:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[1].split('B:')[-1]
                elif data['answer'] == 'C' or data['answer'] == '(C)':
                    if 'C.' in data['question_for_eval'].split('Choices:\n')[-1].split('\n')[2]:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[2].split('C.')[-1]
                    else:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[2].split('C:')[-1]
                elif data['answer'] == 'D' or data['answer'] == '(D)':
                    if 'D.' in data['question_for_eval'].split('Choices:\n')[-1].split('\n')[3]:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[3].split('D.')[-1]
                    else:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[3].split('D:')[-1]
                elif data['answer'] == 'E' or data['answer'] == '(E)':
                    if 'E.' in data['question_for_eval'].split('Choices:\n')[-1].split('\n')[4]:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[4].split('E.')[-1]
                    else:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[4].split('E:')[-1]
                elif data['answer'] == 'F' or data['answer'] == '(F)':
                    if 'F.' in data['question_for_eval'].split('Choices:\n')[-1].split('\n')[5]:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[5].split('F.')[-1]
                    else:
                        data['num_answer'] = \
                        data['question_for_eval'].split('Choices:\n')[-1].split('\n')[5].split('F:')[-1]
                elif '\n' in data['answer']:
                    data['multi_answers'] = [data['answer'].split('\n')[0], data['answer'].split('\n')[1]]
                    if data['multi_answers'][0] == 'A':
                        data['multi_num_answers'] = [
                            data['question_for_eval'].split('Choices:\n')[-1].split('\n')[0].split('A:')[-1],
                            data['question_for_eval'].split('Choices:\n')[-1].split('\n')[2].split('C:')[-1]]
                    elif data['multi_answers'][0] == 'C':
                        data['multi_num_answers'] = [
                            data['question_for_eval'].split('Choices:\n')[-1].split('\n')[2].split('C:')[-1],
                            data['question_for_eval'].split('Choices:\n')[-1].split('\n')[3].split('D:')[-1]]

    total = 0
    correct = 0
    for data in datas:
        if 'Choices:\n' in data['question_for_eval']:
            if '\n' in data['answer']:
                if (data['multi_answers'][0] in extract_answer(data['response']) and data['multi_answers'][
                    1] in extract_answer(data['response'])) or (
                        data['multi_num_answers'][0] in extract_answer(data['response']) and data['multi_num_answers'][
                    1] in extract_answer(data['response'])):
                    correct += 1
                    data['correct'] = True
                else:
                    data['correct'] = False
            elif data['answer'] in extract_answer(data['response']) \
                    or extract_answer(data['response']) in data['answer'] \
                    or data['num_answer'] in extract_answer(data['response']) \
                    or normalize_space(data['answer']) in normalize_space(extract_answer(data['response'])) \
                    or normalize_space(data['num_answer']) in normalize_space(extract_answer(data['response'])) \
                    or normalize_space(extract_answer(data['response'])) in normalize_space(data['num_answer']) \
                    or normalize_space(extract_answer(data['response'])) in normalize_space(data['answer']) \
                    or normalize_sqrt(data['num_answer']) in normalize_sqrt(extract_answer(data['response'])) \
                    or normalize_sqrt(data['num_answer']).replace("{", "").replace("}", "") in extract_answer(
                data['response']):
                correct += 1
                data['correct'] = True
            else:
                data['correct'] = False
        elif data['answer'] in ['True', 'False']:
            if extract_answer(data['response']) == 'A':
                if data['answer'] == 'True':
                    correct += 1
                    data['correct'] = True
                else:
                    data['correct'] = False
            elif extract_answer(data['response']) == 'B':
                if data['answer'] == 'False':
                    correct += 1
                    data['correct'] = True
                else:
                    data['correct'] = False
            else:
                if data['answer'] in extract_answer(data['response']):
                    correct += 1
                    data['correct'] = True
                else:
                    data['correct'] = False
        else:
            if data['answer'] in extract_answer(data['response']) \
                    or extract_answer(data['response']) in data['answer'] \
                    or normalize_space(extract_answer(data['response'])) in normalize_space(data['answer']) \
                    or normalize_space(data['answer']) in normalize_space(extract_answer(data['response'])) \
                    or normalize_sqrt(data['answer']) in normalize_sqrt(extract_answer(data['response'])) \
                    or normalize_sqrt(data['answer']).replace("{", "").replace("}", "") in extract_answer(
                data['response']):
                correct += 1
                data['correct'] = True
            else:
                data['correct'] = False
        total += 1

    print('MathVerse ACC:', correct / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    main(args)