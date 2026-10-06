"""What is a long row for? A test that TinyStories alone cannot give.

Each test row hides a secret word early on, " Lily's secret word is apple.", fills the gap with
ordinary stories, and ends by asking for it, " Lily's secret word is". Half of every training batch
is rows like these; the other half is ordinary stories. Half the secret rows have a short gap, so that every machine can learn the trick; then each machine
is asked for the word from 100 to 970 pieces back.

    python secret_word.py --tag secret256    --block-size 256  --batch-size 128
    python secret_word.py --tag secret1024   --block-size 1024 --batch-size 32
    python secret_word.py --tag secretwin256 --block-size 1024 --batch-size 32 --window 256
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "part4-mlx"))
sys.path.insert(0, os.path.join(HERE, "..", "part5-modern-block"))
from data import batch, load_split  # noqa: E402
from model_llama import Config, MiniLlama  # noqa: E402
from train_llama import lr_at  # noqa: E402

OUT = os.path.join(HERE, "runs")
NAMES = "Lily Tom Sue Ben Mia Max Anna Sam".split()
WORDS = ("apple ball cake bird boat hat star tree fish moon kite frog "
         "cup sock duck shoe drum bell rose leaf egg coin key lamp").split()
ROW = 1025   # 1,024 inputs and the 1,024 next pieces


def secret_row(ids, tok, rng, gap=None):
    """One row: stories, the secret, `gap` pieces of stories, then the question and the word."""
    name, word = rng.choice(NAMES), rng.choice(WORDS)
    secret = tok.encode(f" {name}'s secret word is {word}.")
    question = tok.encode(f" {name}'s secret word is") + tok.encode(f" {word}")
    if gap is None:   # half short gaps, so every machine can learn the trick; half long ones
        longest = ROW - len(secret) - len(question) - 1
        gap = int(rng.integers(20, 200)) if rng.random() < 0.5 else int(rng.integers(200, longest))
    before = ROW - len(secret) - gap - len(question)
    s = int(rng.integers(0, len(ids) - ROW))
    row = np.concatenate([ids[s:s + before], secret, ids[s + before:s + before + gap], question])
    return row.astype(np.int64), tok.encode(f" {word}")[0]


def mixed_batch(ids, tok, block, size, rng):
    """Half secret rows, half ordinary stories, cut to the machine's row length."""
    xs, ys = [], []
    for k in range(size):
        if k % 2 == 0:
            row, _ = secret_row(ids, tok, rng)
            xs.append(row[-block - 1:-1]); ys.append(row[-block:])   # the end of the row, where the question is
        else:
            x, y = batch(ids, block, 1, rng)
            xs.append(x[0]); ys.append(y[0])
    return np.stack(xs), np.stack(ys)


def accuracy(model, ids, tok, block, gap, n=400, seed=0):
    rng = np.random.default_rng(seed)
    right, chance = 0, 0.0
    word_ids = [tok.encode(f" {w}")[0] for w in WORDS]
    for _ in range(n):
        row, word = secret_row(ids, tok, rng, gap=gap)
        x = row[:-1][-block:]                          # the machine sees at most its own row length
        logits = model(mx.array(x[None]))[0, -1]
        p = mx.softmax(logits[mx.array(word_ids)], axis=-1)   # its chances among the 24 words
        right += int(word_ids[int(mx.argmax(p).item())] == word)
        chance += p[word_ids.index(word)].item()
    return right / n, chance / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-n", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids, tok, _ = load_split("bpe8k")
    model = MiniLlama(Config(vocab_size=tok.vocab_size, block_size=args.block_size, window=args.window))
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"tag {args.tag}  row {args.block_size}  window {args.window or 'none'}  params {n_params:,}", flush=True)

    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.99], weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, MiniLlama.loss)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    t0 = time.time()
    for step in range(args.iters):
        opt.learning_rate = lr_at(step, 100, args.iters, 1e-3, 1e-4)
        x, y = mixed_batch(train_ids, tok, args.block_size, args.batch_size, rng)
        loss = train_step(mx.array(x), mx.array(y))
        mx.eval(state)
        if step % 300 == 0:
            print(f"step {step:5d}  loss {loss.item():.3f}", flush=True)
    minutes = (time.time() - t0) / 60

    model.eval()
    result = {"tag": args.tag, "block_size": args.block_size, "window": args.window,
              "params": n_params, "minutes": minutes}
    mx.save_safetensors(os.path.join(OUT, f"ckpt_{args.tag}.safetensors"), dict(tree_flatten(model.parameters())))
    for gap in (100, 200, 400, 700, 970):
        acc, p = accuracy(model, val_ids, tok, args.block_size, gap, n=args.eval_n)
        result[f"gap_{gap}"] = {"right": acc, "chance_on_right_word": p}
        print(f"word {gap} pieces back: right {acc:.1%}, average chance on the right word {p:.1%}", flush=True)
    print(f"done in {minutes:.1f} min", flush=True)
    json.dump(result, open(os.path.join(OUT, f"secret_{args.tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
