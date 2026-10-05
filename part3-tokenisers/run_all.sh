#!/bin/bash
# Train all three tokenizer variants back to back, logging each to data/.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
ITERS=${1:-3000}
EVAL=${2:-300}

for TOK in char bpe8k gpt2; do
  echo "=== $TOK ===" | tee "data/log_${TOK}.txt"
  $PY train.py --tokenizer "$TOK" --iters "$ITERS" --eval-interval "$EVAL" 2>&1 | tee -a "data/log_${TOK}.txt"
done
echo "ALL DONE"
