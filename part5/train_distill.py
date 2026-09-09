"""Knowledge-distillation pre-training, at toy scale.

Meta trained Llama 3.2 1B and 3B with logit distillation: the small model learns
from the token-probability distributions of a much larger model, not just the
one-hot next token. This does the same trick with models small enough to run on
a Mac:

  teacher   GPT-2 small (124M), frozen, loaded through mlx-lm
  student   the Part 3 MiniGPT (~30M with the GPT-2 tokenizer)
  shared    the GPT-2 byte-level BPE vocabulary, so the teacher's logits line up
            index-for-index with the student's

Loss = alpha * cross_entropy(student, next token)
     + (1 - alpha) * T^2 * KL(softmax(teacher / T) || softmax(student / T))

Run --alpha 1.0 for the no-teacher baseline (teacher is not loaded).
"""
import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part3"))
from data import batch, load_split  # noqa: E402
from model_mlx import MiniGPT  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "runs")
TEACHER_ID = "openai-community/gpt2"


def lr_at(step, warmup, total, lo, hi):
    if step < warmup:
        return hi * (step + 1) / warmup
    if step > total:
        return lo
    r = (step - warmup) / (total - warmup)
    return lo + 0.5 * (1 + math.cos(math.pi * r)) * (hi - lo)


def peak_gb():
    return (mx.get_peak_memory() if hasattr(mx, "get_peak_memory")
            else mx.metal.get_peak_memory()) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--teacher", default="none",
                   help="'none' | 'gpt2' (mlx-lm GPT-2 small) | path to a MiniGPT .safetensors")
    p.add_argument("--alpha", type=float, default=0.3, help="weight on the hard-label loss")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids, tok, tpb = load_split("gpt2")
    student = MiniGPT(tok.vocab_size, block_size=args.block_size)
    mx.eval(student.parameters())
    n_params = sum(v.size for _, v in tree_flatten(student.parameters()))
    print(f"tag {args.tag}  alpha {args.alpha}  temp {args.temp}  student params {n_params:,}")

    # --teacher: 'none' | 'gpt2' | 'gpt2-xl' | a Hugging Face id with a '/' (mlx-lm)
    #          | 'torch:<hf id>' for an arch mlx-lm lacks (e.g. GPT-Neo / TinyStories)
    #          | path to a local MiniGPT .safetensors
    HF_ALIASES = {"gpt2": "openai-community/gpt2", "gpt2-xl": "openai-community/gpt2-xl"}
    teacher, torch_teacher = None, None
    if args.teacher.startswith("torch:"):
        import torch
        from transformers import AutoModelForCausalLM
        torch_dev = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher[6:], dtype=torch.float32).to(torch_dev).eval()
        print(f"teacher {args.teacher[6:]} loaded (torch, {torch_dev}) and frozen")
    elif args.teacher != "none" and not args.teacher.endswith(".safetensors"):
        from mlx_lm import load
        model_id = HF_ALIASES.get(args.teacher, args.teacher)
        teacher, _ = load(model_id)
        teacher.freeze()
        mx.eval(teacher.parameters())
        print(f"teacher {model_id} loaded (mlx-lm) and frozen")
    elif args.teacher != "none":
        cfg = json.load(open(os.path.join(os.path.dirname(args.teacher), "teacher_config.json")))
        teacher = MiniGPT(tok.vocab_size, n_layer=cfg["layers"], n_head=cfg["heads"],
                          n_embd=cfg["dim"], block_size=cfg["block_size"])
        teacher.load_weights(list(mx.load(args.teacher).items()))
        teacher.freeze()
        teacher.eval()
        mx.eval(teacher.parameters())
        print(f"teacher {args.teacher} ({cfg['params']:,} params) loaded and frozen")

    distilling = teacher is not None or torch_teacher is not None

    def teacher_logits(x_np):
        if torch_teacher is not None:
            import torch
            with torch.no_grad():
                tl = torch_teacher(torch.tensor(x_np, device=torch_teacher.device)).logits
            return mx.array(tl.float().cpu().numpy())
        return mx.stop_gradient(teacher(mx.array(x_np)))

    def loss_fn(model, x, y, t_logits):
        s = model(x)
        V = s.shape[-1]
        ce = nn.losses.cross_entropy(s.reshape(-1, V), y.reshape(-1), reduction="mean")
        if t_logits is None:
            return ce
        t_logp = nn.log_softmax(t_logits / args.temp, axis=-1)
        s_logp = nn.log_softmax(s / args.temp, axis=-1)
        kl = (mx.exp(t_logp) * (t_logp - s_logp)).sum(-1).mean()
        return args.alpha * ce + (1 - args.alpha) * (args.temp ** 2) * kl

    loss_and_grad = nn.value_and_grad(student, loss_fn)
    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.99], weight_decay=0.1)

    def hard_val(ids, n=40):
        student.eval()
        tot = 0.0
        for _ in range(n):
            xb, yb = batch(ids, args.block_size, args.batch_size, rng)
            s = student(mx.array(xb))
            tot += nn.losses.cross_entropy(
                s.reshape(-1, s.shape[-1]), mx.array(yb).reshape(-1), reduction="mean").item()
        student.train()
        return tot / n

    history, best = [], float("inf")
    t0 = time.time()
    for step in range(args.iters + 1):
        opt.learning_rate = lr_at(step, 100, args.iters, 1e-4, 1e-3)
        if step % args.eval_interval == 0:
            va, tr = hard_val(val_ids), hard_val(train_ids)
            bpb = va / math.log(2) * tpb
            history.append({"step": step, "train": tr, "val": va, "bpb": bpb})
            print(f"step {step:5d}  train {tr:.4f}  val {va:.4f}  val bpb {bpb:.4f}")
            if va < best:
                best = va
                mx.save_safetensors(os.path.join(OUT, f"ckpt_{args.tag}.safetensors"),
                                    dict(tree_flatten(student.parameters())))
        xb, yb = batch(train_ids, args.block_size, args.batch_size, rng)
        tl = teacher_logits(xb) if distilling else None
        loss, grads = loss_and_grad(student, mx.array(xb), mx.array(yb), tl)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(student, grads)
        mx.eval(student.state, opt.state)

    elapsed = time.time() - t0
    print(f"done in {elapsed / 60:.2f} min  best val {best:.4f}  peak {peak_gb():.2f} GB")
    with open(os.path.join(OUT, f"history_{args.tag}.json"), "w") as f:
        json.dump({"history": history, "elapsed_sec": elapsed, "peak_gb": peak_gb(),
                   "tokens_per_byte": tpb, "params": n_params,
                   "alpha": args.alpha, "temp": args.temp}, f, indent=2)


if __name__ == "__main__":
    main()
