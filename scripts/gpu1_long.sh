#!/bin/bash
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=${PYTHON:-python}
for s in 0 1 2; do for arm in router tags; do for ev in mem plain full; do
  f=runs/${arm}_s${s}_long_${ev}.jsonl
  [ -f $f ] && [ $(wc -l < $f) -ge 32 ] || taskset -c 0-39 $P src/rollout.py --arm $arm --ckpt ckpt/${arm}_s${s}.pt --split long --n 32 --bs $([ $ev = full ] && echo 8 || echo 32) --evict $ev --out $f 2>&1 | grep -v Warn | tail -1
  echo "long $arm s$s $ev $(date)"
done; done; done
