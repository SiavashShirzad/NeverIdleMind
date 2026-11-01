#!/bin/bash
# Scaling study with LoRA (r=64): 0.6B (LoRA vs full FT check), 1.7B and 4B; router / tags / orchestrator.
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/after_seeds.sh 1" >/dev/null; do sleep 30; done
export PYTHON=${PYTHON:-python} GPU=1 CPUS=64-127 BS=25 LORA_R=64 TRAIN_ARGS="--lr 2e-4"
BASE_MODEL=Qwen/Qwen3-0.6B SUF=_lora scripts/run_q.sh "router tags" "0"
BASE_MODEL=Qwen/Qwen3-1.7B SUF=_1.7b scripts/run_q.sh "router tags orch" "0"
BASE_MODEL=Qwen/Qwen3-4B SUF=_4b scripts/run_q.sh "router tags orch" "0"
BASE_MODEL=Qwen/Qwen3-1.7B SUF=_1.7b scripts/run_q.sh "router tags orch" "1"
echo "scale done $(date)"
