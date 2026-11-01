#!/bin/bash
# GPU1 after the scaling queue: second seed of the matched see+hear model, then 4B seed 1.
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-python}
GPU=1 CPUS=64-127 SEEDS=1 scripts/gpu0_av2.sh
export GPU=1 CPUS=64-127 BS=25 LORA_R=64 TRAIN_ARGS="--lr 2e-4"
BASE_MODEL=Qwen/Qwen3-4B SUF=_4b scripts/run_q.sh "router tags" "1"
echo "gpu1 more done $(date)"
