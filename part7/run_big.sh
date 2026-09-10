#!/bin/bash
# Larger students, baseline and Qwen3-8B-Base teacher. Lower batch sizes for
# memory headroom; --no-ckpt (the size sweep only needs the curves). History is
# written incrementally so a crash still leaves usable data.
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p runs

$PY -u train_qwen.py --tag large_base    --student large --teacher none   --batch-size 12 --no-ckpt 2>&1 | tee runs/log_large_base.txt
$PY -u train_qwen.py --tag large_qwen8b  --student large --teacher qwen8b --batch-size 12 --no-ckpt 2>&1 | tee runs/log_large_qwen8b.txt
$PY -u train_qwen.py --tag xl_base       --student xl    --teacher none   --batch-size 8 --no-ckpt 2>&1 | tee runs/log_xl_base.txt
$PY -u train_qwen.py --tag xl_qwen8b     --student xl    --teacher qwen8b --batch-size 8 --no-ckpt 2>&1 | tee runs/log_xl_qwen8b.txt
echo BIG-ALL-DONE
