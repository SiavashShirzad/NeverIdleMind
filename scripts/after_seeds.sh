#!/bin/bash
# Follow-up queue: GPU $1 waits for its seed queue, then runs task $2 (longfar | sft).
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "^/bin/bash scripts/run_q.sh ($3)" >/dev/null; do sleep 30; done
if [ "$2" = longfar ]; then
  PYTHONPATH=src $P src/oracle_eval.py longfar:32 2>&1 | grep -v Warn | tail -1
  for s in 0 1 2; do for arm in router tags; do for ev in mem plain; do
    f=runs/${arm}_s${s}_longfar_${ev}.jsonl
    [ -f $f ] && [ $(wc -l < $f) -ge 32 ] || PYTHONPATH=src taskset -c 0-39 $P src/rollout.py --arm $arm --ckpt ckpt/${arm}_s${s}.pt --split longfar --n 32 --bs 32 --evict $ev --out $f 2>&1 | grep -v Warn | tail -1
    echo "longfar $arm s$s $ev $(date)"
  done; done; done
else
  PYTHONPATH=src_mm taskset -c 0-39 $P src_mm/sft_gsm.py 5975 data/gsm8k/sft_test.jsonl 2>&1 | grep -v Warn | tail -2
fi
