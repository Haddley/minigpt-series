"""Run a Qwen3 teacher over the token stream once and cache its top-k next-token
logits, so the student training runs need no teacher in the loop.

This is how distillation is done at scale: the teacher forward is an offline
pass, not part of every training step. Here it also lets both student sizes
reuse one cache.

The stream is processed in non-overlapping BLOCK-token chunks; the student is
trained on the same aligned chunks (train_qwen.py), so at every position the
teacher and student have seen exactly the same context.
"""
import argparse
import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

from qwen_data import load_split

CACHE = os.path.join(os.path.dirname(__file__), "data")
TEACHERS = {"qwen8b": "mlx-community/Qwen3-8B-Base-bf16"}
BLOCK = 256


def cache_split(model, ids, k, batch, tag, split, max_chunks=None):
    n_chunks = len(ids) // BLOCK
    if max_chunks:
        n_chunks = min(n_chunks, max_chunks)
    idx_out = np.zeros((n_chunks * BLOCK, k), dtype=np.uint32)
    val_out = np.zeros((n_chunks * BLOCK, k), dtype=np.float16)
    t0 = time.time()
    for b0 in range(0, n_chunks, batch):
        b1 = min(b0 + batch, n_chunks)
        chunk = np.stack([ids[c * BLOCK:(c + 1) * BLOCK] for c in range(b0, b1)])
        logits = model(mx.array(chunk))                       # [B, BLOCK, V]
        top_idx = mx.argpartition(-logits, k, axis=-1)[..., :k].astype(mx.int32)
        top_val = mx.take_along_axis(logits, top_idx, axis=-1).astype(mx.float32)
        mx.eval(top_idx, top_val)
        rows = slice(b0 * BLOCK, b1 * BLOCK)
        idx_out[rows] = np.array(top_idx).reshape(-1, k).astype(np.uint32)
        val_out[rows] = np.array(top_val).reshape(-1, k).astype(np.float16)
        if b0 % (batch * 10) == 0:
            print(f"  {split} {b1}/{n_chunks}  {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(CACHE, f"teacher_{tag}_{split}_idx.npy"), idx_out)
    np.save(os.path.join(CACHE, f"teacher_{tag}_{split}_val.npy"), val_out)


def teacher_bpb(model, val_ids, tpb, n=40, batch=8):
    """The teacher's own bits per byte, from a small separate pass."""
    rng = np.random.default_rng(0)
    nc = len(val_ids) // BLOCK
    tot = 0.0
    for _ in range(n):
        c = rng.integers(0, nc, size=batch)
        x = np.stack([val_ids[i * BLOCK:(i + 1) * BLOCK] for i in c])
        y = np.stack([val_ids[i * BLOCK + 1:(i + 1) * BLOCK + 1] for i in c])
        lg = model(mx.array(x))
        tot += nn.losses.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), mx.array(y).reshape(-1), reduction="mean").item()
    ce = tot / n
    return ce, ce / math.log(2) * tpb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen8b", choices=list(TEACHERS))
    p.add_argument("--k", type=int, default=48)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--train-chunks", type=int, default=10**9,
                   help="cap the number of 256-token chunks cached (default: the whole stream)")
    args = p.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    train_ids, val_ids, _, tpb = load_split()
    model, _ = load(TEACHERS[args.teacher])
    model.freeze()
    mx.eval(model.parameters())
    print(f"teacher {TEACHERS[args.teacher]}  k={args.k}  batch={args.batch}")

    val_ce, val_bpb = teacher_bpb(model, val_ids, tpb)
    print(f"teacher {args.teacher} val CE {val_ce:.4f}  bits/byte {val_bpb:.4f}", flush=True)
    cache_split(model, train_ids, args.k, args.batch, args.teacher, "train", args.train_chunks)
    with open(os.path.join(CACHE, f"teacher_{args.teacher}_meta.json"), "w") as f:
        json.dump({"k": args.k, "train_chunks": args.train_chunks,
                   "val_ce": val_ce, "val_bpb": val_bpb}, f, indent=2)
    print(f"cached teacher_{args.teacher}_* to {CACHE}")


if __name__ == "__main__":
    main()
