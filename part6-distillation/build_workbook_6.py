"""Builds minigpt_follow_along_6.ipynb, the follow-along notebook for MiniGPT (Part 6).

    python build_workbook_6.py [ITERS]

MLX needs Apple Silicon, so this notebook is for Jupyter on a Mac, not Colab.
"""
import sys

import nbformat

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
POST = "https://haddley.github.io/posts/minigpt5/"
nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
cells = nb.cells
def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbformat.v4.new_code_cell(s.strip("\n")))

md(f"""
# MiniGPT learns from a teacher: follow along

This notebook goes with my post [MiniGPT (Part 6)]({POST}). It looks at a real teacher's wheel of chances, works out how different two wheels are, and trains a small MiniGPT with and without GPT-2 as its teacher.

**It needs a Mac with Apple Silicon**, because MLX only runs there. Open it in Jupyter from a clone of [minigpt-series](https://github.com/Haddley/minigpt-series), with the packages from its `requirements.txt` installed. The first run downloads GPT-2 (about 500 MB) from Hugging Face.
""")
code("""import os, sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from notebook_setup import setup
run = setup("part6-distillation")""")

md("## 1. The answer, and the teacher's wheel\n\nAfter \"Tim gave his dog a\", the guessing game only knows the one right answer. GPT-2, as a teacher, has a chance for every one of its 50,257 pieces.")
code("""import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

teacher, tokenizer = load("openai-community/gpt2")
ids = tokenizer.encode("Tim gave his dog a")
logits = teacher(mx.array([ids]))[0, -1]
wheel = mx.softmax(logits)
top = mx.argsort(-wheel)[:8].tolist()
print("the teacher's wheel, biggest slices:")
for i in top:
    print(f"  {tokenizer.decode([i])!r:12} {wheel[i].item():6.1%}")
print("\\nthe one-hot answer, if the story said ' bone':")
print(f"  {' bone'!r:12} 100.0%, and every other piece 0%")""")

md("## 2. How different are two wheels?\n\nKL divergence is 0 for identical wheels, and grows as they differ. Softening both with a temperature of 2 first, as the post does, makes the near misses count.")
code("""def kl(teacher_logits, student_logits, temp=2.0):
    t = nn.log_softmax(teacher_logits / temp)
    s = nn.log_softmax(student_logits / temp)
    return (mx.exp(t) * (t - s)).sum().item()

even = mx.zeros_like(logits)                  # a student who knows nothing: an even wheel
print(f"teacher against itself:      {kl(logits, logits):.4f}")
print(f"teacher against an even wheel: {kl(logits, even):.4f}")
print(f"softened wheel's top slice: {mx.softmax(logits / 2)[top[0]].item():.1%}, against {wheel[top[0]].item():.1%} unsoftened")""")

md(f"""
## 3. Train with and without a teacher

The same 30-million-number student, {ITERS:,} steps each, on Part 3's stories with GPT-2's pieces. On my Mac Studio, 3,000 steps took about 13 minutes without a teacher, and 25 minutes with GPT-2 as the teacher.
""")
code(f"""run("train_distill.py", "--tag", "baseline", "--teacher", "none", "--alpha", "1.0", "--iters", "{ITERS}")
run("train_distill.py", "--tag", "gpt2", "--teacher", "gpt2", "--alpha", "0.5", "--iters", "{ITERS}")""")
code("""import json
for tag in ("baseline", "gpt2"):
    h = json.load(open(f"runs/history_{tag}.json"))
    best = min(h["history"], key=lambda p: p["val"])
    print(f"{tag:9} best bits per byte {best['bpb']:.4f}, peak memory {h['peak_gb']:.1f} GB, {h['elapsed_sec'] / 60:.1f} minutes")""")

md("""
## 4. Why the weaker teacher helped more

The post's capacity-gap test: how far each student's wheels ended up from each teacher's (KL divergence, 0 = identical), and how often their biggest slices match. These are the saved results of `run_gap.sh`, which retrains the teacher and three students and then runs `measure_gap.py`: about 95 minutes, so run it from a terminal with `./run_gap.sh` if you want to repeat it.
""")
code("""import json
gap = json.load(open("runs/capacity_gap.json"))
print(f"{'student':28} {'from MiniGPT-512':>22} {'from TinyStories-33M':>24}")
for student in ("no teacher", "taught by MiniGPT-512", "taught by TinyStories-33M"):
    a, b = gap[f"{student} | MiniGPT-512"], gap[f"{student} | TinyStories-33M"]
    print(f"{student:28} {a['kl']:10.2f} ({a['top1_agree']:.1%})  {b['kl']:12.2f} ({b['top1_agree']:.1%})")""")

nbformat.write(nb, "minigpt_follow_along_6.ipynb")
print(len(nb.cells), "cells")
