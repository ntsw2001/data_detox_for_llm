#! /bin/bash

# set -x

MODEL_NAME="gpt2-xl"
MODEL_PATH="/path/to/gpt2-xl"
DATASET_NAME="dghs"
DATASET_PATH="/path/to/swift_detoxified_texts_jsonl"

train_batch_size=2
eval_batch_size=2
max_length=1024
max_steps=2000

LR=2e-5
warmup_ratio=0.1

swift sft \
    --seed 1337 \
    --data_seed 1337 \
    --dataset ${DATASET_PATH} \
    --split_dataset_ratio 0.05 \
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
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --save_only_model true \
    --save_safetensors false \
    --max_length ${max_length} \
    --max_steps ${max_steps} \
    --warmup_ratio ${warmup_ratio} \
    --loss_scale all \
    --output_dir output/step4_sft/${MODEL_NAME} \
    --check_model false \
    >> step4_sft_${MODEL_NAME}_${DATASET_NAME}.log 2>&1 &