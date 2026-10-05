"""Train a MiniGPT on the Qwen3 vocabulary, optionally distilling from a cached
Qwen3-4B/8B teacher (see cache_teacher.py).

  --student tiny    dim 384, 6 layers   (~69M params, ~84% embedding table)
  --student scaled  dim 512, 8 layers   (~103M params, ~75% embedding table)
  --teacher none | 4B | 8B              (reads data/teacher_<n>_* top-k logits)

Training samples non-overlapping 256-token chunks, the same ones the teacher
cache was built from, so teacher and student see identical context everywhere.
"""
import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part4-mlx"))
from model_mlx import MiniGPT  # noqa: E402

from qwen_data import load_split  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "runs")
CACHE = os.path.join(os.path.dirname(__file__), "data")
BLOCK = 256
STUDENTS = {"tiny":   dict(n_embd=384,  n_layer=6,  n_head=6),
            "scaled": dict(n_embd=512,  n_layer=8,  n_head=8),
            "large":  dict(n_embd=1024, n_layer=16, n_head=16),
            "xl":     dict(n_embd=1280, n_layer=20, n_head=16)}
TEACHERS = {"qwen8b": "mlx-community/Qwen3-8B-Base-bf16"}


def lr_at(step, warmup, total, lo, hi):
    if step < warmup:
        return hi * (step + 1) / warmup
    if step > total:
        return lo
    r = (step - warmup) / (total - warmup)
    return lo + 0.5 * (1 + math.cos(math.pi * r)) * (hi - lo)


def peak_gb():
    return (mx.get_peak_memory() if hasattr(mx, "get_peak_memory")
            else mx.metal.get_peak_memory()) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--student", default="tiny", choices=list(STUDENTS))
    p.add_argument("--teacher", default="none", choices=["none", "qwen8b"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-chunks", type=int, default=10**9,
                   help="cap training chunks (default: the whole stream)")
    p.add_argument("--no-ckpt", action="store_true", help="skip checkpoint saving (memory)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids, tok, tpb = load_split()
    n_train = min(len(train_ids) // BLOCK, args.train_chunks)
    n_val = len(val_ids) // BLOCK
    print(f"training on the first {n_train:,} chunks ({n_train*BLOCK:,} tokens)")

    cfg = STUDENTS[args.student]
    student = MiniGPT(tok.vocab_size, block_size=BLOCK, **cfg)
    mx.eval(student.parameters())
    n_params = sum(v.size for _, v in tree_flatten(student.parameters()))
    n_emb = student.tok_emb.weight.size
    print(f"student {args.student}: {n_params:,} params "
          f"({n_emb:,} = {100*n_emb/n_params:.0f}% embedding)  vocab {tok.vocab_size:,}")

    t_idx = t_val = None
    if args.teacher != "none":
        t_idx = np.load(os.path.join(CACHE, f"teacher_{args.teacher}_train_idx.npy"))
        t_val = np.load(os.path.join(CACHE, f"teacher_{args.teacher}_train_val.npy"))
        n_train = min(n_train, t_idx.shape[0] // BLOCK)   # only train where the teacher was cached
        print(f"teacher cache {args.teacher}: {t_idx.shape}, k={t_idx.shape[1]}  -> {n_train:,} chunks")

    def get_chunks(ids, n_chunks, want_teacher):
        c = rng.integers(0, n_chunks, size=args.batch_size)
        x = np.stack([ids[i * BLOCK:(i + 1) * BLOCK] for i in c])
        y = np.stack([ids[i * BLOCK + 1:(i + 1) * BLOCK + 1] for i in c])
        if not want_teacher:
            return mx.array(x), mx.array(y), None, None
        rows = np.concatenate([np.arange(i * BLOCK, (i + 1) * BLOCK) for i in c])
        ti = mx.array(t_idx[rows].astype(np.int32)).reshape(args.batch_size, BLOCK, -1)
        tv = mx.array(t_val[rows].astype(np.float32)).reshape(args.batch_size, BLOCK, -1)
        return mx.array(x), mx.array(y), ti, tv

    def loss_fn(model, x, y, ti, tv):
        s = model(x)
        ce = nn.losses.cross_entropy(s.reshape(-1, s.shape[-1]), y.reshape(-1), reduction="mean")
        if ti is None:
            return ce
        s_top = mx.take_along_axis(s, ti, axis=-1)               # student logits at teacher's top-k
        t_lp = nn.log_softmax(tv / args.temp, axis=-1)
        s_lp = nn.log_softmax(s_top / args.temp, axis=-1)
        kl = (mx.exp(t_lp) * (t_lp - s_lp)).sum(-1).mean()
        return args.alpha * ce + (1 - args.alpha) * (args.temp ** 2) * kl

    loss_and_grad = nn.value_and_grad(student, loss_fn)
    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.99], weight_decay=0.1)

    def val_bpb(n=40):
        student.eval()
        tot = 0.0
        for _ in range(n):
            x, y, _, _ = get_chunks(val_ids, n_val, False)
            s = student(x)
            tot += nn.losses.cross_entropy(
                s.reshape(-1, s.shape[-1]), y.reshape(-1), reduction="mean").item()
        student.train()
        ce = tot / n
        return ce, ce / math.log(2) * tpb

    hpath = os.path.join(OUT, f"history_{args.tag}.json")

    def dump(history, elapsed, crashed=False):
        with open(hpath, "w") as f:
            json.dump({"history": history, "elapsed_sec": elapsed, "peak_gb": peak_gb(),
                       "tokens_per_byte": tpb, "params": n_params, "emb_frac": n_emb / n_params,
                       "student": args.student, "teacher": args.teacher, "crashed": crashed}, f, indent=2)

    history, best = [], float("inf")
    t0 = time.time()
    try:
        for step in range(args.iters + 1):
            opt.learning_rate = lr_at(step, 100, args.iters, 1e-4, 1e-3)
            if step % args.eval_interval == 0:
                ce, bpb = val_bpb()
                history.append({"step": step, "val": ce, "bpb": bpb})
                print(f"step {step:5d}  val {ce:.4f}  val bpb {bpb:.4f}", flush=True)
                dump(history, time.time() - t0)                 # incremental — survive a crash
                if ce < best and not args.no_ckpt:
                    best = ce
                    mx.save_safetensors(os.path.join(OUT, f"ckpt_{args.tag}.safetensors"),
                                        dict(tree_flatten(student.parameters())))
            x, y, ti, tv = get_chunks(train_ids, n_train, args.teacher != "none")
            loss, grads = loss_and_grad(student, x, y, ti, tv)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(student, grads)
            mx.eval(student.state, opt.state)
    except RuntimeError as e:
        print(f"CRASHED at step {step}: {e}", flush=True)
        dump(history, time.time() - t0, crashed=True)
        raise

    best = min((p["val"] for p in history), default=float("inf"))
    print(f"done in {(time.time()-t0)/60:.1f} min  best val {best:.4f}  peak {peak_gb():.2f} GB")
    dump(history, time.time() - t0)


if __name__ == "__main__":
    main()
