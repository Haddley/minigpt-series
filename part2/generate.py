"""Sample from a trained checkpoint."""
import argparse
import os

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, choices=["char", "gpt2", "bpe8k"])
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    args = p.parse_args()

    dev = pick_device()
    tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(os.path.join(DATA, f"ckpt_{args.tokenizer}.pt"), map_location=dev)
    model = MiniGPT(ckpt["vocab_size"], block_size=ckpt["block_size"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"checkpoint from step {ckpt['step']}")

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=dev)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
