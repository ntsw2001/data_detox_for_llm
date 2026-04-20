#! /bin/bash

set -x

MODEL_TYPE="qwen2"
DATASET_NAME="dghs"

python step4_toxic_eval.py \
    --config configs/step4_toxic_eval.yaml \
    >> step4_toxic_eval_${MODEL_TYPE}_${DATASET_NAME}.log 2>&1 &
