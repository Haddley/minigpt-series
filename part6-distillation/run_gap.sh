#!/bin/bash
# Testing the capacity-gap explanation: retrain the teacher and three students, keep
# their checkpoints, and measure how close each student got to each teacher.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p runs
$PY -u train_teacher.py --dim 512 --layers 8 --iters 5000                                   2>&1 | tee runs/log_gap_teacher.txt
$PY -u train_distill.py --tag gap_baseline --teacher none --alpha 1.0                       2>&1 | tee runs/log_gap_baseline.txt
$PY -u train_distill.py --tag gap_big --teacher runs/teacher.safetensors --alpha 0.5        2>&1 | tee runs/log_gap_big.txt
$PY -u train_distill.py --tag gap_ts33m --teacher torch:roneneldan/TinyStories-33M --alpha 0.5 2>&1 | tee runs/log_gap_ts33m.txt
$PY -u measure_gap.py                                                                       2>&1 | tee runs/log_gap_measure.txt
echo ALL DONE
