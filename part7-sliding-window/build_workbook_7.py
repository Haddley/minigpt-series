"""Builds minigpt_follow_along_7.ipynb, the follow-along notebook for MiniGPT (Part 7).

    python build_workbook_7.py

MLX needs Apple Silicon, so this notebook is for Jupyter on a Mac, not Colab.
"""
import nbformat

POST = "https://haddley.github.io/posts/minigpt6/"
nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
cells = nb.cells
def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbformat.v4.new_code_cell(s.strip("\n")))

md(f"""
# MiniGPT reads further: follow along

This notebook goes with my post [MiniGPT (Part 7)]({POST}). It counts how fast attention's matches grow, checks that the chunked sliding window gives exactly the right answers, and races full attention against the window for memory.

**It needs a Mac with Apple Silicon**, because MLX only runs there. Open it in Jupyter from a clone of [minigpt-series](https://github.com/Haddley/minigpt-series), with the packages from its `requirements.txt` installed.
""")
code("""import os, sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from notebook_setup import setup
run = setup("part7-sliding-window")
sys.path.insert(0, os.path.abspath("../part5-modern-block"))""")

md("## 1. Counting the matches\n\nEvery working card matches its query against every position up to its own. Each doubling of the row roughly quadruples the matches. With a 256-position window, they grow only in step with the row.")
code("""def full_matches(T):
    return T * (T + 1) // 2

def window_matches(T, w=256):
    return sum(min(i + 1, w) for i in range(T))

for T in (4, 8, 1024, 2048, 4096):
    print(f"row {T:>5}: full attention {full_matches(T):>12,} matches, 256-position window {window_matches(T):>10,}")""")

md("## 2. The chunked window gives the right answers\n\nWith a window as long as the whole row, `chunked_swa` must give exactly full attention's numbers. With a real window, it must match the masked version.")
code("""import mlx.core as mx
from model_llama import chunked_swa, sliding_window_mask

mx.random.seed(0)
B, H, T, D = 1, 2, 512, 64
q, k, v = (mx.random.normal((B, H, T, D)) for _ in range(3))
scale = D ** -0.5

full = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
print("window = whole row, biggest difference from full attention:",
      mx.abs(chunked_swa(q, k, v, T, scale) - full).max().item())

masked = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=sliding_window_mask(T, 64, q.dtype))
print("window = 64, biggest difference from the masked version:   ",
      mx.abs(chunked_swa(q, k, v, 64, scale) - masked).max().item())""")

md("## 3. A short memory race\n\nThe same few training steps as `mem_sweep.py`, at three row lengths. The full sweep, up to 16,384 positions, is `python mem_sweep.py`, and needs a 64 GB Mac.")
code("""from mem_sweep import run as race
for T in (512, 1024, 2048):
    for mode in ("full", "naive", "chunked"):
        peak, dt = race(T, mode)
        print(f"row {T:>5}  {mode:8}  most memory {peak:6.2f} GB   {dt * 1000:7.1f} ms per step")""")

md("""
## 4. The secret word: what is a long row for?

The post's secret-word test: three machines asked for a word hidden 100 to 970 pieces back. These are the saved results of `run_secret.sh`, which trains the three machines: about 45 minutes, so run it from a terminal with `./run_secret.sh` if you want to repeat it.
""")
code("""import json
names = {"secret256": "256-position row", "secret1024": "1,024 positions, full attention",
         "secretwin256": "1,024 positions, 256-position window"}
gaps = (100, 200, 400, 700, 970)
print(f"{'machine':38}" + "".join(f"{g:>7} back" for g in gaps))
for tag, name in names.items():
    r = json.load(open(f"runs/secret_{tag}.json"))
    print(f"{name:38}" + "".join(f"{r[f'gap_{g}']['right']:>12.1%}" for g in gaps))""")

nbformat.write(nb, "minigpt_follow_along_7.ipynb")
print(len(nb.cells), "cells")
