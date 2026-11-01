#!/bin/bash
# See + hear with visual exposure matched to router_mmv: stage 1 has 120k visual and 120k heard questions, stage 2 adds 10k listening episodes to the router_mmv mixture.
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
C=${CPUS:-0-63}
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=src_av OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f data/train_senses2.pt ] || taskset -c 64-127 $P src_av/data.py --n_id 0 --n_senses 60000 --out data/train_senses2.pt 2>&1 | grep -v Warn | tail -1
[ -f data/train_av2.pt ] || taskset -c 64-127 $P src_av/data.py --n_id 12000 --n_vis 15000 --n_aud 10000 --out data/train_av2.pt 2>&1 | grep -v Warn | tail -1
echo "av2 data $(date)"
while pgrep -f "^/bin/bash scripts/gpu0_lf.sh" >/dev/null; do sleep 60; done
export OMP_NUM_THREADS=8
for s in ${SEEDS:-0}; do
  [ -f ckpt/router_avm1_s$s.pt ] || taskset -c $C $P src_av/train.py --arm router_avm1 --seed $s --data data/train_senses2.pt --lr 0 --lr_new 1e-3 --tok_per_batch 12288 2>&1 | grep -v Warn | tail -1
  [ -f ckpt/router_avm_s$s.pt ] || taskset -c $C $P src_av/train.py --arm router_avm --seed $s --data data/train_av2.pt --init ckpt/router_avm1_s$s.pt --tok_per_batch 12288 2>&1 | grep -v Warn | tail -1
  for split in aud vis math_async math_solo id; do
    f=runs/router_avm_s${s}_${split}.jsonl
    [ -f $f ] && [ $(wc -l < $f) -ge 300 ] || taskset -c $C $P src_av/rollout.py --arm router_avm --ckpt ckpt/router_avm_s$s.pt --split $split --n 300 --bs 50 --out $f 2>&1 | grep -v Warn | tail -1
    echo "avm s$s $split $(date)"
  done
done
echo "avm done $(date)"
