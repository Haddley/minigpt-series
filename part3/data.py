"""Load the Part 2 TinyStories slice and tokenizer, and serve integer-id batches.

Part 3 reuses everything Part 2 prepared: run part2/prepare_data.py and
part2/tokenizers_setup.py first.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part2"))
from tokenizer import load_tokenizer  # noqa: E402

PART2_DATA = os.path.join(os.path.dirname(__file__), "..", "part2", "data")


def load_split(name):
    """Return (train_ids, val_ids, tokenizer, tokens_per_byte-on-val)."""
    tok = load_tokenizer(name)
    train_text = open(os.path.join(PART2_DATA, "train.txt"), encoding="utf-8").read()
    val_text = open(os.path.join(PART2_DATA, "val.txt"), encoding="utf-8").read()
    train_ids = np.array(tok.encode(train_text), dtype=np.int32)
    val_ids = np.array(tok.encode(val_text), dtype=np.int32)
    tokens_per_byte = len(val_ids) / len(val_text.encode("utf-8"))
    return train_ids, val_ids, tok, tokens_per_byte


def batch(ids, block_size, batch_size, rng):
    ix = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[i:i + block_size] for i in ix])
    y = np.stack([ids[i + 1:i + block_size + 1] for i in ix])
    return x, y
