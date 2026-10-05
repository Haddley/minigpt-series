#!/bin/bash
# The modern block, then each modern feature turned back off one at a time.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python

$PY -u train_llama.py --tag modern                  2>&1 | tee runs/log_modern.txt
$PY -u train_llama.py --tag rms_rope_gelu --mlp gelu 2>&1 | tee runs/log_gelu.txt
$PY -u train_llama.py --tag rms_rope_mha  --gqa-off  2>&1 | tee runs/log_mha.txt
$PY -u train_llama.py --tag layer_rope    --norm layer 2>&1 | tee runs/log_layer.txt
$PY -u train_llama.py --tag rms_learned   --pos learned 2>&1 | tee runs/log_learned.txt
echo ALL DONE
