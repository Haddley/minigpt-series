#!/bin/bash
# What is a long row for? Three machines, the same mix of secret rows and stories.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p runs
$PY -u secret_word.py --tag secret256    --block-size 256  --batch-size 128          2>&1 | tee runs/log_secret256.txt
$PY -u secret_word.py --tag secret1024   --block-size 1024 --batch-size 32           2>&1 | tee runs/log_secret1024.txt
$PY -u secret_word.py --tag secretwin256 --block-size 1024 --batch-size 32 --window 256 2>&1 | tee runs/log_secretwin256.txt
echo ALL DONE
