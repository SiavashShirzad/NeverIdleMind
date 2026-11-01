#!/bin/bash
# Usage: GPU=1 scripts/run_q.sh "arms" "seeds" "splits"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=${PYTHON:-python}
for s in $2; do for arm in $1; do
  [ -f ckpt/${arm}${SUF}_s${s}.pt ] || taskset -c ${CPUS:-0-39} $P src/train.py --arm $arm --seed $s --suffix "${SUF}" ${TRAIN_ARGS} 2>&1 | grep -v Warn | tail -1
  for split in ${3:-id stress}; do
    f=runs/${arm}${SUF}_s${s}_${split}.jsonl
    [ -f $f ] && [ $(wc -l < $f) -ge 300 ] || taskset -c ${CPUS:-0-39} $P src/rollout.py --arm $arm --ckpt ckpt/${arm}${SUF}_s${s}.pt --split $split --n 300 --bs ${BS:-50} --out $f 2>&1 | grep -v Warn | tail -1
  done
  echo "done $arm${SUF} s$s $(date)"
done; done
