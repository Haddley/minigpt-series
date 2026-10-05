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

# how much is luck? the new block and the old-MLP version, from two more random starts
for s in 1 2; do
  $PY -u train_llama.py --tag modern_seed$s --seed $s            2>&1 | tee runs/log_modern_seed$s.txt
  $PY -u train_llama.py --tag gelu_seed$s   --seed $s --mlp gelu 2>&1 | tee runs/log_gelu_seed$s.txt
done
echo ALL DONE
