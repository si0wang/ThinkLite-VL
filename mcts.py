from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration, GenerationConfig
import json
import os
import numpy as np
import math
import torch
import torch.nn as nn
from PIL import Image
import requests
import torch.nn.init as init
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from safetensors.torch import load_file
import os
from datasets import load_dataset
import pandas as pd
import io
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm

eval_prompt_template = '''Please help me judge the correctness of the generated answer and the corresponding rationale. 
Question: {}
Ground truth answer: {}
Generated rationale and answer: {}
Your output should only be one sentence: the generated answer is true or false.'''

few_shot_cot_prompt = '''Answer the question **step by step** and provide the final answer at the end, each step should end with **<end>** and put your final answer within $\boxed{}$. Below are two examples:
Question: BoatsRUs built 7 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatsRUs by the end of May of this year?
### Step1: To find the result of the total number of canoes built by BoatsRUs by the end of May, I need to find the number of canoes built in each month from January to May and then add them up. <end>
### Step2: To find the number of canoes built in each month, I need to use the formula for the number of canoes built in a given month, which is the number of canoes built in the previous month times 2. <end>
### Step3: So, the number of canoes built in January is 7, the number of canoes built in February is 7 times 2, which is 14, the number of canoes built in March is 14 times 2, which is 28, the number of canoes built in April is 28 times 2, which is 56, and the number of canoes built in May is 56 times 2, which is 112. <end>
### Step4: Now, I can add up these numbers to get the total number of canoes built by BoatsRUs by the end of May: 7 plus 14 plus 28 plus 56 plus 112, which is 217. <end>
### Final Answer: The answer is: $boxed{217}$.
Question: Find the number of blue circles in the figure.
### Step 1: To find the result of the number of blue circles, I need to interpret the figure. The figure is a Venn diagram with two labeled sets: - One set labeled "blue" contains all the shapes that are blue in color. - The other set labeled "circle" contains all the shapes that are circular in shape. The overlapping region of the Venn diagram contains shapes that are both blue and circular. <end>
### Step 2: The overlapping region contains shapes that meet both criteria: Blue color and Circle shape. From the diagram: - There is **one blue circle** in the overlapping region. <end>
### Final Answer: The answer is: $boxed{1}$.
Remember to answer the question **step by step**! Here is your question:
'''


def read_all_parquet_to_list(directory: str):
    parquet_files = [
        f for f in os.listdir(directory) if f.endswith(".parquet")
    ]

    df_list = []

    for parquet_file in parquet_files:
        file_path = os.path.join(directory, parquet_file)
        df = pd.read_parquet(file_path)
        df_list.append(df)

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
    else:
        return []

    data_list = combined_df.to_dict(orient='records')

    return data_list

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

class State:

    def __init__(self, image_feat, text_context, solution_steps=None):
        self.image_feat = image_feat
        self.text_context = text_context
        self.solution_steps = solution_steps if solution_steps else []
        self.is_terminal = False

    def copy(self):
        new_state = State(
            image_feat=self.image_feat,
            text_context=self.text_context,
            solution_steps=self.solution_steps.copy()
        )
        new_state.is_terminal = self.is_terminal
        return new_state

    def __repr__(self):
        return f"<State steps={len(self.solution_steps)}, terminal={self.is_terminal}>"


class Action:

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f"<Action: {self.text}>"


class VisionLanguageModel:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _run_vlm(self, image_feat, text_context, generation_config, history=None):

        prompt = text_context
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }]
        if history:
            message.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "".join(history)}, ],
            })

        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )[:-32]
        image_inputs = Image.open(io.BytesIO(image_feat))
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        question_input_length = inputs['input_ids'].shape[1]

        generated_ids = self.model.generate(**inputs, generation_config=generation_config, stop_strings=['<end>'],
                                       max_new_tokens=2048, tokenizer=self.processor.tokenizer)
        output = self.processor.decode(
            generated_ids[0][question_input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output

    def propose_actions(self, state, generation_config, top_k=3):

        actions = []
        for i in range(top_k):
            llama_output = self._run_vlm(
                image_feat=state.image_feat,
                text_context=state.text_context,
                generation_config=generation_config,
                history=state.solution_steps
            )
            action_text = llama_output
            prob = 1.0 / top_k
            actions.append((Action(action_text), prob))
        return actions

    def transition(self, state, action):
        next_state = state.copy()
        next_state.solution_steps.append(action.text)

        if len(next_state.solution_steps) >= 10 or "The final answer is:" in next_state.solution_steps[-1]:
            next_state.is_terminal = True
        return next_state

    def evaluate_terminal_state(self, state, eval_llm, eval_llm_tokenizer, question, answer):
        if state.is_terminal:
            simulation_response = "".join(state.solution_steps)
            prompt = eval_prompt_template.format(question, answer, simulation_response)

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = eval_llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = eval_llm_tokenizer([text], return_tensors="pt").to(eval_llm.device)

            generated_ids = eval_llm.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = eval_llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if 'true' in response.split('.')[0]:
                return 1.0
            else:
                return 0.0
        return 0.0


class MCTSNode:
    def __init__(self, state):
        self.state = state
        self.children = {}  # dict(action -> MCTSNode)
        self.visit_count = 0
        self.value_sum = 0.0
        self.parent = None
        self.action_from_parent = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def ucb_score(parent, child, c_puct=1.0):
    if child.visit_count == 0:
        return float('inf')
    return (child.value
            + c_puct * math.sqrt(math.log(parent.visit_count) / (child.visit_count)))


def select_child(node, c_puct=1.0):
    best_score = -float('inf')
    best_action = None
    best_child = None
    for action, child in node.children.items():
        score = ucb_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def expand(node, vlm, generation_config, top_k=3):
    if node.state.is_terminal:
        return
    actions_probs = vlm.propose_actions(node.state, generation_config, top_k)
    for action, prob in actions_probs:
        if action not in node.children:
            next_state = vlm.transition(node.state, action)
            child_node = MCTSNode(next_state)
            child_node.parent = node
            child_node.action_from_parent = action
            node.children[action] = child_node


def simulate(state, vlm, eval_llm, eval_llm_tokenizer, question, answer, generation_config, rollout_limit=10):
    temp_state = state.copy()
    steps = 0
    while not temp_state.is_terminal and steps < rollout_limit:
        actions_probs = vlm.propose_actions(temp_state, generation_config, top_k=1)
        action, prob = random.choice(actions_probs)
        temp_state = vlm.transition(temp_state, action)
        steps += 1

    return vlm.evaluate_terminal_state(temp_state, eval_llm, eval_llm_tokenizer, question, answer), temp_state


def backpropagate(node, reward):
    cur = node
    while cur is not None:
        cur.visit_count += 1
        cur.value_sum += reward
        cur = cur.parent

def mcts_search(root_state, vlm, eval_llm, eval_llm_tokenizer, question, answer, generation_config, n_iterations,
                c_puct=1.0, top_k=3):
    root_node = MCTSNode(root_state)
    solution = None

    for iter in range(n_iterations):
        node = root_node
        while not node.state.is_terminal and len(node.children) > 0:
            _, child = select_child(node, c_puct)
            node = child

        if not node.state.is_terminal:
            expand(node, vlm, generation_config, top_k=top_k)
            if len(node.children) > 0:
                action = random.choice(list(node.children.keys()))
                node = node.children[action]


        reward, simulate_state = simulate(node.state, vlm, eval_llm, eval_llm_tokenizer, question, answer,
                                          generation_config, rollout_limit=10)
        if reward == 1:
            solution = simulate_state
            break

        backpropagate(node, reward)

    best_path = []
    current = root_node
    while not current.state.is_terminal and len(current.children) > 0:
        best_child = max(current.children.values(), key=lambda c: c.visit_count)
        best_path.append(best_child.action_from_parent.text)
        current = best_child
    return root_node, best_path, solution, iter

def solve_math_reasoning_vlm(image_data, text_prompt, model, generation_config, processor, eval_llm,
                                eval_llm_tokenizer, question, answer, n_iterations):
    image_feat = image_data

    init_state = State(
        image_feat=image_feat,
        text_context=text_prompt,
        solution_steps=[]
    )

    vlm = VisionLanguageModel(model, processor)

    root, steps, solution, n_iter = mcts_search(
        root_state=init_state,
        vlm=vlm,
        eval_llm=eval_llm,
        eval_llm_tokenizer=eval_llm_tokenizer,
        question=question,
        answer=answer,
        generation_config=generation_config,
        n_iterations=n_iterations,
        c_puct=1.0,
        top_k=3
    )
    return root, steps, solution, n_iter


def main(args):
    device = "cuda:{}".format(args.gpu_id)
    generation_config = GenerationConfig(
        temperature=0.5,
        do_sample=True,
        top_p=0.9,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    eval_llm = AutoModelForCausalLM.from_pretrained(
        args.eval_model_name,
        torch_dtype="auto",
        device_map=device
    )
    eval_llm_tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name)
    final_response = []

    df = pd.read_parquet(args.data_pths, engine='pyarrow')  # Your path of dataset
    datas = df.to_dict(orient='records')

    data_chunk = get_chunk(datas, args.num_chunks, args.chunk_idx)
    for data in tqdm(data_chunk, desc="MCTS Progress"):
        image_data = data['image']
        question = data['problem'].split('<image>')[1]
        answer = data['answer']
        text_prompt = few_shot_cot_prompt + '{}'.format(question)

        root, solution_steps, solution, n_iter = solve_math_reasoning_vlm(
            image_data=image_data,
            text_prompt=text_prompt,
            model=model,
            generation_config=generation_config,
            processor=processor,
            eval_llm=eval_llm,
            eval_llm_tokenizer=eval_llm_tokenizer,
            question=question,
            answer=answer,
            n_iterations=args.max_num_iterations,
        )

        if solution is not None:
            try:
                data['solution'] = ''.join(solution.solution_steps)
                data['iters'] = n_iter
                final_response.append(data)
            except Exception as e:
                continue

    df = pd.DataFrame(final_response)
    df.to_parquet(args.output_file, index=False, engine='pyarrow')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--eval_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_pths", type=str, nargs='+', default="None")
    parser.add_argument("--output_file", type=str, default="answer.jsonl")
    parser.add_argument("--max_num_iterations", type=int, default=50)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    main(args)