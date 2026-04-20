#! /bin/bash

set -x

MODEL_TYPE="qwen2"
DATASET_NAME="dghs"

export TOKENIZERS_PARALLELISM=false

python step2_text_detox.py \
    --config configs/step2_text_detox.yaml \
    >> step2_detoxify_texts_${MODEL_TYPE}_${DATASET_NAME}.log 2>&1 &
