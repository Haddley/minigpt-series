"""Sample from a distillation checkpoint (student is the Part 3 MiniGPT, GPT-2 vocab)."""
import argparse
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part3"))
from data import load_split  # noqa: E402
from model_mlx import MiniGPT  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "runs")


def sample(logits, temperature, top_k):
    logits = logits / temperature
    if top_k:
        kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
        logits = mx.where(logits < kth, -mx.inf, logits)
    return mx.random.categorical(logits)[:, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)          # baseline | gpt2 | big
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mx.random.seed(args.seed)
    _, _, tok, _ = load_split("gpt2")
    model = MiniGPT(tok.vocab_size)
    model.load_weights(list(mx.load(os.path.join(OUT, f"ckpt_{args.tag}.safetensors")).items()))
    model.eval()

    idx = mx.array([tok.encode(args.prompt)])
    for _ in range(args.max_new_tokens):
        logits = model(idx[:, -model.block_size:])[:, -1, :]
        idx = mx.concatenate([idx, sample(logits, args.temperature, args.top_k)], axis=1)
        mx.eval(idx)
    print(tok.decode(idx[0].tolist()))


if __name__ == "__main__":
    main()
