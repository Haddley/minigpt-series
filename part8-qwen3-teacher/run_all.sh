#!/bin/bash
# Four runs: tiny/scaled student, with and without the Qwen3-8B-Base teacher.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p runs

$PY -u train_qwen.py --tag tiny_base   --student tiny   --teacher none   2>&1 | tee runs/log_tiny_base.txt
$PY -u train_qwen.py --tag tiny_qwen8b --student tiny   --teacher qwen8b 2>&1 | tee runs/log_tiny_qwen8b.txt
$PY -u train_qwen.py --tag scaled_base   --student scaled --teacher none   2>&1 | tee runs/log_scaled_base.txt
$PY -u train_qwen.py --tag scaled_qwen8b --student scaled --teacher qwen8b 2>&1 | tee runs/log_scaled_qwen8b.txt
echo PART7-ALL-DONE
