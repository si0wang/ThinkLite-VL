#!/bin/bash

output_prefix="./output_files/mcts_qwen_"
num_chunks=8

for i in {0..7}; do
     python mcts.py \
        --data_pths './ThinkLite-VL-70k.parquet' \
        --output_file ${output_prefix}$((i+1)).parquet \
        --num-chunks $num_chunks \
        --chunk-idx $((i)) \
        --gpu-id $((i)) &
done