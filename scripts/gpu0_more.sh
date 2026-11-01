#!/bin/bash
# GPU0 after the matched see+hear run: seed 2 of it, then the 4B orchestrator seed 1.
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-python}
GPU=0 CPUS=0-63 SEEDS=2 scripts/gpu0_av2.sh
export GPU=0 CPUS=0-63 BS=25 LORA_R=64 TRAIN_ARGS="--lr 2e-4"
BASE_MODEL=Qwen/Qwen3-4B SUF=_4b scripts/run_q.sh "orch" "1"
echo "gpu0 more done $(date)"
