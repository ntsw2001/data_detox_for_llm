#! /bin/bash

MODEL_NAME="Qwen2.5-0.5B-Instruct"
MODEL_PATH="/path/to/Qwen2.5-0.5B-Instruct"
DATASET_NAME="dghs"
DATASET_PATH="data/dghs_for_toxic_train.jsonl"

train_batch_size=16
eval_batch_size=16
max_length=512
max_steps=1000

LR=2e-5
warmup_ratio=0.05

swift sft \
    --seed 1337 \
    --data_seed 1337 \
    --dataset ${DATASET_PATH} \
    --split_dataset_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --task_type causal_lm \
    --model ${MODEL_PATH} \
    --train_type full \
    --use_chat_template false \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size ${train_batch_size} \
    --per_device_eval_batch_size ${eval_batch_size} \
    --learning_rate ${LR} \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 5 \
    --save_only_model true \
    --save_safetensors false \
    --max_length ${max_length} \
    --max_steps ${max_steps} \
    --warmup_ratio ${warmup_ratio} \
    --output_dir output/step1_sft/${MODEL_NAME} \
    --gradient_checkpointing false \
    --check_model false \
    >> step1_sft_${MODEL_NAME}_${DATASET_NAME}.log 2>&1 &