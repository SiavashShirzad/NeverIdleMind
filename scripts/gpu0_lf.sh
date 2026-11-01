#!/bin/bash
# Long-gap memory training: extra 600 long episodes whose revisions arrive after eviction (mem views only).
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/gpu0_av.sh" >/dev/null; do sleep 60; done
export PYTHON=${PYTHON:-python} GPU=0 CPUS=0-63 BS=25 SUF=_lf TRAIN_ARGS="--data data/train_lf.pt"
P=$PYTHON
scripts/run_q.sh "router tags" "0 1" "id stress"
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src
for s in 0 1; do for arm in router tags; do for ev in mem plain; do
  f=runs/${arm}_lf_s${s}_longfar_${ev}.jsonl
  [ -f $f ] && [ $(wc -l < $f) -ge 32 ] || taskset -c $CPUS $P src/rollout.py --arm $arm --ckpt ckpt/${arm}_lf_s${s}.pt --split longfar --n 32 --bs 32 --evict $ev --out $f 2>&1 | grep -v Warn | tail -1
  echo "lf longfar $arm s$s $ev $(date)"
done; done; done
echo "gpu0 lf done $(date)"
