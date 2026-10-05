"""Measure a teacher's own bits per byte on the TinyStories validation split,
so the distillation table can show the advantage each teacher actually had."""
import argparse
import json
import math
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part4-mlx"))
from data import load_split  # noqa: E402
from model_mlx import MiniGPT  # noqa: E402

HF_ALIASES = {"gpt2": "openai-community/gpt2", "gpt2-xl": "openai-community/gpt2-xl"}


def make_forward(teacher, tok):
    if teacher.startswith("torch:"):
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(teacher[6:], dtype=torch.float32)
        m = m.to("mps" if torch.backends.mps.is_available() else "cpu").eval()

        def fwd(x_np):
            with torch.no_grad():
                lg = m(torch.tensor(x_np, device=m.device)).logits
            return mx.array(lg.float().cpu().numpy())
        return fwd

    if teacher.endswith(".safetensors"):
        cfg = json.load(open(os.path.join(os.path.dirname(teacher), "teacher_config.json")))
        m = MiniGPT(tok.vocab_size, n_layer=cfg["layers"], n_head=cfg["heads"],
                    n_embd=cfg["dim"], block_size=cfg["block_size"])
        m.load_weights(list(mx.load(teacher).items()))
        m.eval()
        return lambda x_np: m(mx.array(x_np))

    from mlx_lm import load
    m, _ = load(HF_ALIASES.get(teacher, teacher))
    return lambda x_np: m(mx.array(x_np))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batches", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    _, val_ids, tok, tpb = load_split("gpt2")
    rng = np.random.default_rng(0)
    fwd = make_forward(args.teacher, tok)

    tot = 0.0
    for _ in range(args.batches):
        i = rng.integers(0, len(val_ids) - args.block_size - 1, size=args.batch_size)
        x = np.stack([val_ids[j:j + args.block_size] for j in i])
        y = np.stack([val_ids[j + 1:j + args.block_size + 1] for j in i])
        lg = fwd(x)
        tot += nn.losses.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), mx.array(y).reshape(-1), reduction="mean").item()
    ce = tot / args.batches
    print(f"{args.teacher}:  val CE {ce:.4f}  bits/byte {ce / math.log(2) * tpb:.4f}")


if __name__ == "__main__":
    main()
