#!/bin/bash

num_chunks=8
model_id="Ep30-Dynamic-False-Qwen2.5-VL-72B-MCTSALL_Node6_Rollout8_GlobalBS96_ROLLBS-384-2025-04-21-22-22/global_step_6"
model_name="/blob/v-xiyaowang/zyang/Easy-R1/output/Ep30-Dynamic-False-Qwen2.5-VL-72B-MCTSALL_Node6_Rollout8_GlobalBS96_ROLLBS-384-2025-04-21-22-22/global_step_60/actor/huggingface/"

datasets=("mathvista" "mathvision" "mmmu" "mmstar" "mmbench" "mathverse" "ai2d" "mmvet")


for dataset in "${datasets[@]}"; do
   output_prefix="/blob/v-xiyaowang/v-xiyaowang/llava_prm/llava_prm/eval_files/${dataset}/answers/${model_id}"

   model_script="./model_${dataset}_qwen.py"

   python $model_script \
           --model_id $model_name \
           --answers-file "${output_prefix}.jsonl" \
           --batch-size 16
done

