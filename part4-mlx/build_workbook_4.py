"""Builds minigpt_follow_along_4.ipynb, the follow-along notebook for MiniGPT (Part 4).

    python build_workbook_4.py [ITERS]

MLX needs Apple Silicon, so this notebook is for Jupyter on a Mac, not Colab.
"""
import sys

import nbformat

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
POST = "https://haddley.github.io/posts/minigpt3/"
nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
cells = nb.cells
def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbformat.v4.new_code_cell(s.strip("\n")))

md(f"""
# MiniGPT on a faster engine: follow along

This notebook goes with my post [MiniGPT (Part 4)]({POST}). It rebuilds Part 3's machine in Apple's MLX, checks that it is the same machine, trains it, races MLX against PyTorch, and writes a story.

**It needs a Mac with Apple Silicon**, because MLX only runs there. Open it in Jupyter from a clone of [minigpt-series](https://github.com/Haddley/minigpt-series), with the packages from its `requirements.txt` installed.
""")
code("""import os, sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from notebook_setup import setup
run = setup("part4-mlx")""")

md("## 1. The same machine, in two engines\n\nBuild Part 3's PyTorch machine and the MLX one with the same 8,192 token embeddings, and count their numbers.")
code("""import importlib.util
import mlx.core as mx
from mlx.utils import tree_flatten
from model_mlx import MiniGPT as MLXMiniGPT

spec = importlib.util.spec_from_file_location("torch_model", "../part3-tokenisers/model.py")
torch_model = importlib.util.module_from_spec(spec); spec.loader.exec_module(torch_model)

pt = torch_model.MiniGPT(8192)
mlx_model = MLXMiniGPT(8192)
mx.eval(mlx_model.parameters())
print("PyTorch:", f"{sum(p.numel() for p in pt.parameters()):,}", "numbers")
print("MLX:    ", f"{sum(v.size for _, v in tree_flatten(mlx_model.parameters())):,}", "numbers")""")

md("## 2. Lazy evaluation\n\nMLX writes calculations down and only runs them when a result is needed. Building a big multiplication is instant; asking for its result is where the work happens.")
code("""import time
a = mx.random.normal((4096, 4096))
b = mx.random.normal((4096, 4096))
mx.eval(a, b)

t = time.time(); c = a @ b @ a @ b;  print(f"writing the steps down: {(time.time() - t) * 1000:8.2f} ms")
t = time.time(); mx.eval(c);         print(f"calculating them:       {(time.time() - t) * 1000:8.2f} ms")""")

md(f"## 3. Train the MLX machine\n\n{ITERS:,} training steps on Part 3's stories, checking bits per byte every 300 steps. On my Mac Studio, 3,000 steps took about 8 minutes.")
code(f"""run("train_mlx.py", "--tokenizer", "bpe8k", "--iters", "{ITERS}", "--eval-interval", "300")""")

md("## 4. The race\n\nThe same 200 timed training steps in each engine, each in its own fresh program, so the memory figures stay clean.")
code("""for framework in ("torch", "mlx-nocompile", "mlx"):
    run("bench.py", "--framework", framework)""")

md("## 5. What it writes")
code("""run("generate_mlx.py", "--prompt", "Once upon a time", "--max-new-tokens", "150")""")

nbformat.write(nb, "minigpt_follow_along_4.ipynb")
print(len(nb.cells), "cells")
