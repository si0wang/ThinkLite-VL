#!/bin/bash

model_id="russwang/ThinkLite-VL-7B"
datasets=("mathvista" "mathvision" "mmmu" "mmstar" "mathverse" "ai2d")


for dataset in "${datasets[@]}"; do
    datapath="./eval_files/${dataset}/${model_id}"
    model_script="eval/${dataset}_score.py"

    python $model_script  --data_path "${datapath}.jsonl"

    wait
done

