#!/bin/bash
# Visual features for synthetic scenes and GSM8K self-distilled rationales (multimodal pilot).
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
export PYTHONPATH=src_mm
mkdir -p data/vis
$P src_mm/scenes.py 20000 1 data/vis/train.pt
$P src_mm/scenes.py 1000 999 data/vis/test.pt
$P src_mm/gsm_distill.py test 1 data/gsm8k/base_test_nothink.jsonl
$P src_mm/gsm_distill.py test 1 data/gsm8k/base_test_think.jsonl think
$P src_mm/gsm_distill.py train 3 data/gsm8k/train_rationales.jsonl
