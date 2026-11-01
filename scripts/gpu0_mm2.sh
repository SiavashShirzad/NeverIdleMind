#!/bin/bash
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src_mm OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f data/train_mm.pt ] || taskset -c 0-39 $P src_mm/data.py --n_id 12000 2>&1 | grep -v Warn | tail -2
export OMP_NUM_THREADS=8
[ -f ckpt/router_mm_s0.pt ] || taskset -c 0-39 $P src_mm/train.py --arm router_mm --seed 0 --tok_per_batch 12288 2>&1 | grep -v Warn | tail -2
for split in id vis math_solo math_async; do
  f=runs/router_mm_s0_${split}.jsonl
  [ -f $f ] && [ $(wc -l < $f) -ge 300 ] || taskset -c 0-39 $P src_mm/rollout.py --arm router_mm --ckpt ckpt/router_mm_s0.pt --split $split --n 300 --bs 25 --out $f 2>&1 | grep -v Warn | tail -1
  echo "mm $split $(date)"
done
export PYTHONPATH=src
BS=25 GPU=0 scripts/run_q.sh "orch" "1" "stress"
BS=25 GPU=0 scripts/run_q.sh "router_noemb" "0 1 2"
echo "gpu0 all done $(date)"
