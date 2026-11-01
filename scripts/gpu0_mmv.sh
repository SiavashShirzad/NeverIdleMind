#!/bin/bash
# Vision alignment: stage 1 projector-only on look episodes (LLaVA-style), stage 2 full FT on mm + vis episodes.
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src_mm OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f data/train_look.pt ] || taskset -c 0-39 $P src_mm/data.py --n_id 0 --n_look 30000 --out data/train_look.pt 2>&1 | grep -v Warn | tail -1
[ -f data/train_mmv.pt ] || taskset -c 0-39 $P src_mm/data.py --n_id 12000 --n_vis 15000 --out data/train_mmv.pt 2>&1 | grep -v Warn | tail -1
export OMP_NUM_THREADS=8
[ -f ckpt/router_mmv1_s0.pt ] || taskset -c 0-39 $P src_mm/train.py --arm router_mmv1 --seed 0 --data data/train_look.pt --lr 0 --lr_new 1e-3 --tok_per_batch 12288 2>&1 | grep -v Warn | tail -2
[ -f ckpt/router_mmv_s0.pt ] || taskset -c 0-39 $P src_mm/train.py --arm router_mmv --seed 0 --data data/train_mmv.pt --init ckpt/router_mmv1_s0.pt --tok_per_batch 12288 2>&1 | grep -v Warn | tail -2
for split in vis id math_solo math_async; do
  f=runs/router_mmv_s0_${split}.jsonl
  [ -f $f ] && [ $(wc -l < $f) -ge 300 ] || taskset -c 0-39 $P src_mm/rollout.py --arm router_mmv --ckpt ckpt/router_mmv_s0.pt --split $split --n 300 --bs 50 --out $f 2>&1 | grep -v Warn | tail -1
  echo "mmv $split $(date)"
done
