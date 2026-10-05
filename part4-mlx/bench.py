"""Time a fixed number of training steps in one framework and report tokens/sec
and peak GPU memory. Run once per framework (separate processes keep the memory
numbers clean):

    python bench.py --framework torch
    python bench.py --framework mlx

Same model shape, same batch, same data. This is a throughput and memory probe,
not a full training run: a short warmup, then STEPS timed steps.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part3-tokenisers"))
from data import batch, load_split  # noqa: E402

STEPS = 200
WARMUP = 20


def bench_torch(train_ids, vocab_size, bs, block, seed):
    import torch
    from model import MiniGPT  # part3-tokenisers/model.py

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MiniGPT(vocab_size, block_size=block).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def one():
        xb, yb = batch(train_ids, block, bs, rng)
        x = torch.tensor(xb, dtype=torch.long, device=dev)
        y = torch.tensor(yb, dtype=torch.long, device=dev)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(WARMUP):
        one()
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        one()
    torch.mps.synchronize()
    dt = time.time() - t0
    peak = torch.mps.driver_allocated_memory() / 1e9
    return STEPS * bs * block / dt, peak


def bench_mlx(train_ids, vocab_size, bs, block, seed, compiled=True):
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from model_mlx import MiniGPT
    from train_mlx import peak_gb

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = MiniGPT(vocab_size, block_size=block)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, MiniGPT.loss)
    state = [model.state, opt.state]

    def _step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        return loss

    step = partial(mx.compile, inputs=state, outputs=state)(_step) if compiled else _step

    def one():
        xb, yb = batch(train_ids, block, bs, rng)
        step(mx.array(xb), mx.array(yb))
        mx.eval(state)

    for _ in range(WARMUP):
        one()
    t0 = time.time()
    for _ in range(STEPS):
        one()
    dt = time.time() - t0
    return STEPS * bs * block / dt, peak_gb()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", required=True, choices=["torch", "mlx", "mlx-nocompile"])
    p.add_argument("--tokenizer", default="bpe8k", choices=["char", "bpe8k", "gpt2"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    train_ids, _, tok, _ = load_split(args.tokenizer)
    if args.framework == "torch":
        tps, peak = bench_torch(train_ids, tok.vocab_size, args.batch_size, args.block_size, args.seed)
        label = "PyTorch-MPS"
    else:
        compiled = args.framework == "mlx"
        tps, peak = bench_mlx(train_ids, tok.vocab_size, args.batch_size, args.block_size,
                              args.seed, compiled=compiled)
        label = "MLX" if compiled else "MLX (no compile)"
    print(f"{label:<17} {tps:>10,.0f} tok/s   peak {peak:5.2f} GB   "
          f"(vocab {tok.vocab_size:,}, batch {args.batch_size}, block {args.block_size})")


if __name__ == "__main__":
    main()
