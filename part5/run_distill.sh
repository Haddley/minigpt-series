#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p runs

$PY -u train_distill.py --tag baseline --teacher none --alpha 1.0                 2>&1 | tee runs/log_baseline.txt
$PY -u train_distill.py --tag gpt2     --teacher gpt2 --alpha 0.5                  2>&1 | tee runs/log_gpt2.txt
$PY -u train_distill.py --tag big      --teacher runs/teacher.safetensors --alpha 0.5 2>&1 | tee runs/log_big.txt
echo ALL DONE
