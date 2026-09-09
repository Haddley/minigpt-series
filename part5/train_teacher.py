"""Train a larger MiniGPT on the same TinyStories data, to act as the teacher.

This mirrors what Meta actually did: the Llama 3.2 1B/3B students were distilled
from Llama 3.1 8B/70B - same family, same data, more capacity. Here the teacher
is a ~55M MiniGPT (dim 512, 8 layers) against the ~30M student, both on the
GPT-2 tokenizer so their logits line up.
"""
import argparse
import json
import math
import os
import sys
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part3"))
from data import batch, load_split  # noqa: E402
from model_mlx import MiniGPT  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "runs")


def lr_at(step, warmup, total, lo, hi):
    if step < warmup:
        return hi * (step + 1) / warmup
    if step > total:
        return lo
    r = (step - warmup) / (total - warmup)
    return lo + 0.5 * (1 + math.cos(math.pi * r)) * (hi - lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids, tok, tpb = load_split("gpt2")
    model = MiniGPT(tok.vocab_size, n_layer=args.layers, n_head=args.heads,
                    n_embd=args.dim, block_size=args.block_size)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"teacher dim {args.dim} layers {args.layers}  params {n_params:,}")

    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.99], weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, MiniGPT.loss)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    def val():
        model.eval()
        t = 0.0
        for _ in range(40):
            xb, yb = batch(val_ids, args.block_size, args.batch_size, rng)
            t += MiniGPT.loss(model, mx.array(xb), mx.array(yb)).item()
        model.train()
        return t / 40

    best = float("inf")
    t0 = time.time()
    for s in range(args.iters + 1):
        opt.learning_rate = lr_at(s, 200, args.iters, 1e-4, 1e-3)
        if s % args.eval_interval == 0:
            v = val()
            print(f"step {s:5d}  val {v:.4f}  bpb {v / math.log(2) * tpb:.4f}")
            if v < best:
                best = v
                mx.save_safetensors(os.path.join(OUT, "teacher.safetensors"),
                                    dict(tree_flatten(model.parameters())))
        xb, yb = batch(train_ids, args.block_size, args.batch_size, rng)
        step(mx.array(xb), mx.array(yb))
        mx.eval(state)

    print(f"teacher done in {(time.time() - t0) / 60:.1f} min  best val {best:.4f}")
    json.dump({"dim": args.dim, "layers": args.layers, "heads": args.heads,
               "block_size": args.block_size, "params": n_params, "best_val": best},
              open(os.path.join(OUT, "teacher_config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
