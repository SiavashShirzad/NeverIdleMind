#!/bin/bash
# See + hear + read: stage 1 aligns both sensory projectors (LLM frozen), stage 2 full fine-tuning on the av mixture.
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/gpu0_mm_ext.sh" >/dev/null; do sleep 60; done
P=${PYTHON:-python}
C=0-63
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src_av OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f data/reorient_av.flag ] && export REORIENT=1
[ -f data/train_av.pt ] || taskset -c $C $P src_av/data.py --n_id 12000 --n_vis 10000 --n_aud 10000 --out data/train_av.pt 2>&1 | grep -v Warn | tail -1
export OMP_NUM_THREADS=8
for s in ${SEEDS:-0}; do
  [ -f ckpt/router_av1_s$s.pt ] || taskset -c $C $P src_av/train.py --arm router_av1 --seed $s --data data/train_senses.pt --lr 0 --lr_new 1e-3 --tok_per_batch 12288 2>&1 | grep -v Warn | tail -1
  [ -f ckpt/router_av_s$s.pt ] || taskset -c $C $P src_av/train.py --arm router_av --seed $s --data data/train_av.pt --init ckpt/router_av1_s$s.pt --tok_per_batch 12288 2>&1 | grep -v Warn | tail -1
  for split in aud vis math_async math_solo id; do
    f=runs/router_av_s${s}_${split}.jsonl
    [ -f $f ] && [ $(wc -l < $f) -ge 300 ] || taskset -c $C $P src_av/rollout.py --arm router_av --ckpt ckpt/router_av_s$s.pt --split $split --n 300 --bs 50 --out $f 2>&1 | grep -v Warn | tail -1
    echo "av s$s $split $(date)"
  done
done
echo "gpu0 av done $(date)"
