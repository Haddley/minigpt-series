"""Train the MLX MiniGPT on the Part 2 TinyStories slice.

The training step is built around three MLX ideas:

  nn.value_and_grad   - a functional gradient: hand it the model and a loss
                        function, get back a function returning (loss, grads)
  mx.compile          - fuses the whole step (forward, backward, clip, update)
                        into one graph; inputs/outputs=state so the compiled
                        graph is allowed to mutate the model and optimizer
  lazy evaluation     - nothing actually runs until mx.eval; one mx.eval per
                        step, on the updated state, is what drives execution
"""
import argparse
import json
import math
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from data import batch, load_split
from model_mlx import MiniGPT

OUT = os.path.join(os.path.dirname(__file__), "runs")


def lr_at(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step > total:
        return lr_min
    ratio = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr_max - lr_min)


def peak_gb():
    for fn in ("get_peak_memory", "get_peak_memory_gb"):
        if hasattr(mx, fn):
            v = getattr(mx, fn)()
            return v / 1e9 if v > 1e6 else v
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / 1e9
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="bpe8k", choices=["char", "gpt2", "bpe8k"])
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids, tok, tokens_per_byte = load_split(args.tokenizer)
    print(f"tokenizer {args.tokenizer}  vocab {tok.vocab_size:,}  "
          f"train tokens {len(train_ids):,}  tokens/byte {tokens_per_byte:.4f}")

    model = MiniGPT(tok.vocab_size, block_size=args.block_size)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"parameters {n_params:,}")

    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.99], weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, MiniGPT.loss)

    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    def eval_loss(ids, n=40):
        model.eval()
        total = 0.0
        for _ in range(n):
            xb, yb = batch(ids, args.block_size, args.batch_size, rng)
            total += MiniGPT.loss(model, mx.array(xb), mx.array(yb)).item()
        model.train()
        return total / n

    history, best_val = [], float("inf")
    t0 = time.time()
    for step in range(args.iters + 1):
        opt.learning_rate = lr_at(step, 100, args.iters, 1e-3, 1e-4)

        if step % args.eval_interval == 0:
            tr, va = eval_loss(train_ids), eval_loss(val_ids)
            bpb = va / math.log(2) * tokens_per_byte
            history.append({"step": step, "train": tr, "val": va, "bpb": bpb})
            print(f"step {step:5d}  train {tr:.4f}  val {va:.4f}  val bpb {bpb:.4f}")
            if va < best_val:
                best_val = va
                mx.save_safetensors(
                    os.path.join(OUT, f"ckpt_{args.tokenizer}.safetensors"),
                    dict(tree_flatten(model.parameters())),
                )

        xb, yb = batch(train_ids, args.block_size, args.batch_size, rng)
        loss = train_step(mx.array(xb), mx.array(yb))
        mx.eval(state)

    elapsed = time.time() - t0
    tok_per_s = args.iters * args.batch_size * args.block_size / elapsed
    print(f"done in {elapsed / 60:.2f} min  best val {best_val:.4f}  "
          f"{tok_per_s:,.0f} tok/s  peak {peak_gb():.2f} GB")
    with open(os.path.join(OUT, f"history_{args.tokenizer}.json"), "w") as f:
        json.dump({"history": history, "elapsed_sec": elapsed,
                   "tokens_per_sec": tok_per_s, "peak_gb": peak_gb(),
                   "tokens_per_byte": tokens_per_byte, "params": n_params,
                   "framework": "mlx"}, f, indent=2)


if __name__ == "__main__":
    main()
