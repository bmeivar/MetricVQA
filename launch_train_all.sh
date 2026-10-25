#!/bin/bash
# Launch 4 LoRA training jobs in parallel (1 GPU each)

COMMON="--lr 1e-5 --epochs 2 --lora_target all --dataset bmeivar/MetricVQA --image_dir $OBJECTRON_IMAGES"

CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_id "Qwen/Qwen2.5-VL-7B-Instruct" $COMMON \
  --output_dir checkpoints/lora_Qwen2.5-VL-7B-Instruct &

CUDA_VISIBLE_DEVICES=1 python train.py \
  --model_id "Qwen/Qwen2.5-VL-3B-Instruct" $COMMON \
  --output_dir checkpoints/lora_Qwen2.5-VL-3B-Instruct &

CUDA_VISIBLE_DEVICES=2 python train.py \
  --model_id "google/paligemma-3b-mix-448" $COMMON \
  --output_dir checkpoints/lora_paligemma-3b-mix-448 &

CUDA_VISIBLE_DEVICES=3 python train.py \
  --model_id "llava-hf/llava-onevision-qwen2-7b-ov-hf" $COMMON \
  --output_dir checkpoints/lora_llava-onevision-qwen2-7b &

wait
echo "All training complete"
