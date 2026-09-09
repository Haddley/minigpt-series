"""Sample from a trained MLX checkpoint."""
import argparse
import os

import mlx.core as mx

from data import load_split
from model_mlx import MiniGPT

OUT = os.path.join(os.path.dirname(__file__), "runs")


def sample(logits, temperature, top_k):
    logits = logits / temperature
    if top_k:
        kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
        logits = mx.where(logits < kth, -mx.inf, logits)
    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(mx.log(probs))[:, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="bpe8k", choices=["char", "gpt2", "bpe8k"])
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mx.random.seed(args.seed)
    _, _, tok, _ = load_split(args.tokenizer)
    model = MiniGPT(tok.vocab_size)
    weights = mx.load(os.path.join(OUT, f"ckpt_{args.tokenizer}.safetensors"))
    model.load_weights(list(weights.items()))
    model.eval()

    idx = mx.array([tok.encode(args.prompt)])
    for _ in range(args.max_new_tokens):
        logits = model(idx[:, -model.block_size:])[:, -1, :]
        idx = mx.concatenate([idx, sample(logits, args.temperature, args.top_k)], axis=1)
        mx.eval(idx)
    print(tok.decode(idx[0].tolist()))


if __name__ == "__main__":
    main()
