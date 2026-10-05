"""Builds minigpt_follow_along_5.ipynb, the follow-along notebook for MiniGPT (Part 5).

    python build_workbook_5.py [ITERS]

MLX needs Apple Silicon, so this notebook is for Jupyter on a Mac, not Colab.
"""
import sys

import nbformat

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
POST = "https://haddley.github.io/posts/minigpt4/"
nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
cells = nb.cells
def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbformat.v4.new_code_cell(s.strip("\n")))

md(f"""
# MiniGPT with the modern block: follow along

This notebook goes with my post [MiniGPT (Part 5)]({POST}). It builds the 2017 block and Llama's modern block, counts what each change costs, shows rotary positions (RoPE) at work, trains the modern block, and writes a story.

**It needs a Mac with Apple Silicon**, because MLX only runs there. Open it in Jupyter from a clone of [minigpt-series](https://github.com/Haddley/minigpt-series), with the packages from its `requirements.txt` installed.
""")
code("""import os, sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from notebook_setup import setup
run = setup("part5-modern-block")""")

md("## 1. Six machines, and what each change costs\n\nThe new block, then each change turned back off on its own. Every machine has the same 8,192 token cards.")
code("""import mlx.core as mx
from mlx.utils import tree_flatten
from model_llama import Config, MiniLlama

variants = {
    "new block, all four changes": {},
    "... but the old MLP":        {"mlp": "gelu"},
    "... but full keys and values": {"n_kv_heads": 6},
    "... but the old normalise":  {"norm": "layer"},
    "... but position cards":     {"pos": "learned"},
}
for name, change in variants.items():
    m = MiniLlama(Config(vocab_size=8192, **change))
    mx.eval(m.parameters())
    print(f"{name:32} {sum(v.size for _, v in tree_flatten(m.parameters())):>12,} numbers")""")

md("## 2. RoPE: the match only feels the distance\n\nTurn the same query card and key card for two pairs of positions with the same gap: (7, 5) and (107, 105). With RoPE, the two matches come out the same.")
code("""import mlx.nn as nn
rope = nn.RoPE(64, traditional=False, base=10000)
mx.random.seed(0)
q = mx.random.normal((1, 1, 1, 64))
k = mx.random.normal((1, 1, 1, 64))

def turned(card, position):
    return rope(card, offset=position)

for qp, kp in ((7, 5), (107, 105), (7, 3)):
    match = (turned(q, qp) * turned(k, kp)).sum().item()
    print(f"query at {qp:3}, key at {kp:3} (gap {qp - kp}): match {match:8.4f}")""")

md(f"## 3. Train the new block\n\n{ITERS:,} steps on Part 3's stories, checking bits per byte every 300 steps. On my Mac Studio, 3,000 steps took about 7.5 minutes. `run_ablation.sh` trains all five variants, about 40 minutes in all.")
code(f"""run("train_llama.py", "--tag", "modern", "--iters", "{ITERS}", "--eval-interval", "300")""")

md("## 4. What it writes")
code("""run("generate_llama.py", "--tag", "modern", "--prompt", "Once upon a time", "--max-new-tokens", "150")""")

nbformat.write(nb, "minigpt_follow_along_5.ipynb")
print(len(nb.cells), "cells")
