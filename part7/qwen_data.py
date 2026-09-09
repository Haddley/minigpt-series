"""Qwen3 tokenizer + the TinyStories slice re-tokenised with it.

Part 7 swaps the 50,257-token GPT-2 vocabulary the series has used since Part 2
for Qwen3's 151,936-slot vocabulary, so that a Qwen3 model can be a distillation
teacher (logit distillation needs an identical vocabulary). Reuses the raw text
from part2/data/.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part3"))

PART2_DATA = os.path.join(os.path.dirname(__file__), "..", "part2", "data")
CACHE = os.path.join(os.path.dirname(__file__), "data")
QWEN_ID = "Qwen/Qwen3-4B"          # tokenizer only; 4B and 8B share it
MODEL_VOCAB = 151936               # padded output width of the Qwen3 models


class QwenTokenizer:
    name = "qwen3"
    vocab_size = MODEL_VOCAB

    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(QWEN_ID)

    def encode(self, s):
        return self.tok.encode(s)

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids])


def load_split():
    """(train_ids, val_ids, tokenizer, tokens_per_byte-on-val), with an on-disk cache."""
    os.makedirs(CACHE, exist_ok=True)
    tok = QwenTokenizer()
    out = {}
    for split in ("train", "val"):
        npy = os.path.join(CACHE, f"{split}.npy")
        if os.path.exists(npy):
            out[split] = np.load(npy)
            continue
        text = open(os.path.join(PART2_DATA, f"{split}.txt"), encoding="utf-8").read()
        ids = np.array(tok.encode(text), dtype=np.int32)
        np.save(npy, ids)
        out[split] = ids
    val_bytes = len(open(os.path.join(PART2_DATA, "val.txt"), encoding="utf-8").read().encode("utf-8"))
    return out["train"], out["val"], tok, len(out["val"]) / val_bytes


def batch(ids, block_size, batch_size, rng):
    ix = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[i:i + block_size] for i in ix])
    y = np.stack([ids[i + 1:i + block_size + 1] for i in ix])
    return x, y
