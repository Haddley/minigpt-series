"""Sample from a Part 8 checkpoint (MiniGPT on the Qwen3 vocabulary)."""
import argparse
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part4-mlx"))
from model_mlx import MiniGPT  # noqa: E402

from qwen_data import QwenTokenizer  # noqa: E402
from train_qwen import STUDENTS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "runs")


def sample(logits, temperature, top_k):
    logits = logits / temperature
    if top_k:
        kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
        logits = mx.where(logits < kth, -mx.inf, logits)
    return mx.random.categorical(logits)[:, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--student", default="tiny", choices=list(STUDENTS))
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mx.random.seed(args.seed)
    tok = QwenTokenizer()
    model = MiniGPT(tok.vocab_size, block_size=256, **STUDENTS[args.student])
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
