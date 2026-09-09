"""Train MiniGPT on the TinyStories slice with one of the three tokenizers.

Everything except the tokenizer is held fixed, so the runs are comparable:
6 layers, 6 heads, 384-dim, 256-token context, batch 32, AdamW with a short
warmup and cosine decay, gradient clipping at 1.0, best-validation checkpoint.

Cross-entropy is not comparable across vocabularies - a bit of loss over 8k
tokens is not a bit of loss over 50k. So the run is also scored in
bits-per-byte, which normalises by how many raw UTF-8 bytes each tokenizer
packs into a token.
"""
import argparse
import json
import math
import os
import time

import torch

from model import MiniGPT
from tokenizer import load_tokenizer

DATA = os.path.join(os.path.dirname(__file__), "data")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, dev):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(dev), y.to(dev)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, dev, iters=100):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, block_size, batch_size, dev)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def lr_at(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step > total:
        return lr_min
    ratio = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr_max - lr_min)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, choices=["char", "gpt2", "bpe8k"])
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = pick_device()
    tok = load_tokenizer(args.tokenizer)
    print(f"device {dev}  tokenizer {args.tokenizer}  vocab {tok.vocab_size:,}")

    train_text = open(os.path.join(DATA, "train.txt"), encoding="utf-8").read()
    val_text = open(os.path.join(DATA, "val.txt"), encoding="utf-8").read()
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    val_bytes = len(val_text.encode("utf-8"))
    tokens_per_byte = len(val_ids) / val_bytes
    print(f"train tokens {len(train_ids):,}  val tokens {len(val_ids):,}  "
          f"tokens/byte {tokens_per_byte:.4f}")

    model = MiniGPT(tok.vocab_size, block_size=args.block_size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_params:,}  "
          f"(embedding {model.tok_emb.weight.numel():,}, "
          f"rest {n_params - model.tok_emb.weight.numel():,})")

    decay, no_decay = [], []
    for _, pp in model.named_parameters():
        (decay if pp.dim() >= 2 else no_decay).append(pp)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3, betas=(0.9, 0.99),
    )

    history, best_val = [], float("inf")
    t0 = time.time()
    for step in range(args.iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, 100, args.iters, 1e-3, 1e-4)

        if step % args.eval_interval == 0:
            tr = estimate_loss(model, train_ids, args.block_size, args.batch_size, dev)
            va = estimate_loss(model, val_ids, args.block_size, args.batch_size, dev)
            bpb = va / math.log(2) * tokens_per_byte
            history.append({"step": step, "train": tr, "val": va, "bpb": bpb})
            print(f"step {step:5d}  train {tr:.4f}  val {va:.4f}  val bpb {bpb:.4f}")
            if va < best_val:
                best_val = va
                torch.save(
                    {"model": model.state_dict(), "step": step,
                     "tokenizer": args.tokenizer, "block_size": args.block_size,
                     "vocab_size": tok.vocab_size},
                    os.path.join(DATA, f"ckpt_{args.tokenizer}.pt"),
                )

        x, y = get_batch(train_ids, args.block_size, args.batch_size, dev)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    elapsed = time.time() - t0
    print(f"done in {elapsed / 60:.2f} min  best val {best_val:.4f}")
    with open(os.path.join(DATA, f"history_{args.tokenizer}.json"), "w") as f:
        json.dump({"history": history, "elapsed_sec": elapsed,
                   "tokens_per_byte": tokens_per_byte,
                   "vocab_size": tok.vocab_size, "params": n_params}, f, indent=2)


if __name__ == "__main__":
    main()
