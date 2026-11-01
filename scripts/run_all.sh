#!/bin/bash
# Usage: scripts/run_all.sh "arm1 arm2 ..." "seeds"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=${PYTHON:-python}
for s in $2; do for arm in $1; do
  [ -f ckpt/${arm}_s${s}.pt ] || taskset -c 0-39 $P src/train.py --arm $arm --seed $s 2>&1 | grep -v Warn | tail -2
  for split in id stress; do
    [ -f runs/${arm}_s${s}_${split}.jsonl ] || taskset -c 0-39 $P src/rollout.py --arm $arm --ckpt ckpt/${arm}_s${s}.pt --split $split --n 300 --bs 75 --out runs/${arm}_s${s}_${split}.jsonl 2>&1 | grep -v Warn | tail -1
  done
  echo "done $arm s$s $(date)"
done; done
