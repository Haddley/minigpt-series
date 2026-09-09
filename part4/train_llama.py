"""Train MiniLlama on the Part 2 TinyStories slice (8k BPE tokenizer).

Reuses the Part 3 MLX training loop - lazy eval, nn.value_and_grad, mx.compile -
and adds ablation flags so the modern block can be turned on one piece at a time:

    python train_llama.py --tag modern
    python train_llama.py --tag gelu   --mlp gelu
    python train_llama.py --tag mha    --gqa-off
    python train_llama.py --tag learned --pos learned
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

from model_llama import Config, MiniLlama  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "runs")


def lr_at(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step > total:
        return lr_min
    r = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (1 + math.cos(math.pi * r)) * (lr_max - lr_min)


def peak_gb():
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / 1e9
    return mx.metal.get_peak_memory() / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--tokenizer", default="bpe8k")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--norm", default="rms", choices=["rms", "layer"])
    p.add_argument("--pos", default="rope", choices=["rope", "learned"])
    p.add_argument("--mlp", default="swiglu", choices=["swiglu", "gelu"])
    p.add_argument("--gqa-off", action="store_true", help="use full MHA (n_kv_heads = n_heads)")
    p.add_argument("--window", type=int, default=0, help="sliding-window attention width (0 = full)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids, tok, tokens_per_byte = load_split(args.tokenizer)
    cfg = Config(vocab_size=tok.vocab_size, block_size=args.block_size,
                 norm=args.norm, pos=args.pos, mlp=args.mlp, window=args.window,
                 n_kv_heads=6 if args.gqa_off else 2)
    model = MiniLlama(cfg)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"tag {args.tag}  norm {cfg.norm}  pos {cfg.pos}  mlp {cfg.mlp}  "
          f"kv_heads {model.blocks[0].attn.n_kv_heads}  params {n_params:,}")

    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.99], weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, MiniLlama.loss)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    def eval_loss(ids, n=40):
        model.eval()
        tot = 0.0
        for _ in range(n):
            xb, yb = batch(ids, args.block_size, args.batch_size, rng)
            tot += MiniLlama.loss(model, mx.array(xb), mx.array(yb)).item()
        model.train()
        return tot / n

    history, best = [], float("inf")
    t0 = time.time()
    for step in range(args.iters + 1):
        opt.learning_rate = lr_at(step, 100, args.iters, 1e-3, 1e-4)
        if step % args.eval_interval == 0:
            va = eval_loss(val_ids)
            tr = eval_loss(train_ids)
            bpb = va / math.log(2) * tokens_per_byte
            history.append({"step": step, "train": tr, "val": va, "bpb": bpb})
            print(f"step {step:5d}  train {tr:.4f}  val {va:.4f}  val bpb {bpb:.4f}")
            if va < best:
                best = va
                mx.save_safetensors(os.path.join(OUT, f"ckpt_{args.tag}.safetensors"),
                                    dict(tree_flatten(model.parameters())))
        xb, yb = batch(train_ids, args.block_size, args.batch_size, rng)
        train_step(mx.array(xb), mx.array(yb))
        mx.eval(state)

    elapsed = time.time() - t0
    print(f"done in {elapsed / 60:.2f} min  best val {best:.4f}  peak {peak_gb():.2f} GB")
    with open(os.path.join(OUT, f"history_{args.tag}.json"), "w") as f:
        json.dump({"history": history, "elapsed_sec": elapsed, "peak_gb": peak_gb(),
                   "tokens_per_byte": tokens_per_byte, "params": n_params,
                   "config": vars(cfg)}, f, indent=2)


if __name__ == "__main__":
    main()
