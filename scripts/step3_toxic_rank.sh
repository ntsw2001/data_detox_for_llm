#! /bin/bash

set -x

MODEL_TYPE="qwen2"
DATASET_NAME="dghs"

python step3_toxic_rank.py \
    --config configs/step3_toxic_rank.yaml \
    >> step3_rank_texts_${MODEL_TYPE}_${DATASET_NAME}.log 2>&1 &
