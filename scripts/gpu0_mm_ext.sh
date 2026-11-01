#!/bin/bash
# Multimodal extension: re-orientation after paused thoughts (router_mmr) and extra seeds for router_mmv / router_mmr.
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
C=0-63
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src_mm OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f data/train_mmr.pt ] || REORIENT=1 taskset -c $C $P src_mm/data.py --n_id 12000 --n_vis 15000 --out data/train_mmr.pt 2>&1 | grep -v Warn | tail -1
export OMP_NUM_THREADS=8
evals() {  # $1 arm, $2 seed
  for split in math_async math_solo vis id; do
    f=runs/$1_s$2_${split}.jsonl
    [ -f $f ] && [ $(wc -l < $f) -ge 300 ] || taskset -c $C $P src_mm/rollout.py --arm $1 --ckpt ckpt/$1_s$2.pt --split $split --n 300 --bs 50 --out $f 2>&1 | grep -v Warn | tail -1
    echo "$1 s$2 $split $(date)"
  done
}
stage2() {  # $1 arm, $2 seed, $3 data
  [ -f ckpt/router_mmv1_s$2.pt ] || taskset -c $C $P src_mm/train.py --arm router_mmv1 --seed $2 --data data/train_look.pt --lr 0 --lr_new 1e-3 --tok_per_batch 12288 2>&1 | grep -v Warn | tail -1
  [ -f ckpt/$1_s$2.pt ] || taskset -c $C $P src_mm/train.py --arm $1 --seed $2 --data $3 --init ckpt/router_mmv1_s$2.pt --tok_per_batch 12288 2>&1 | grep -v Warn | tail -1
  evals $1 $2
}
stage2 router_mmr 0 data/train_mmr.pt
for s in 1 2; do stage2 router_mmv $s data/train_mmv.pt; stage2 router_mmr $s data/train_mmr.pt; done
echo "gpu0 mm ext done $(date)"
