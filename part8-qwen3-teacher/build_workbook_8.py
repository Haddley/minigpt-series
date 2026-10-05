"""Builds minigpt_follow_along_8.ipynb, the follow-along notebook for MiniGPT (Part 8).

    python build_workbook_8.py

MLX needs Apple Silicon, so this notebook is for Jupyter on a Mac, not Colab. It needs
only Qwen3's tokeniser, not the 16 GB teacher model.
"""
import nbformat

POST = "https://haddley.github.io/posts/minigpt7/"
nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
cells = nb.cells
def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbformat.v4.new_code_cell(s.strip("\n")))

md(f"""
# MiniGPT and a frontier teacher: follow along

This notebook goes with my post [MiniGPT (Part 8)]({POST}). It compares Qwen3's pieces with GPT-2's and my 8k ones, counts what Qwen3's token cards cost each student, and works out how much text the big students would have needed.

**It needs a Mac with Apple Silicon**, because MLX only runs there. Open it in Jupyter from a clone of [minigpt-series](https://github.com/Haddley/minigpt-series), with the packages from its `requirements.txt` installed. It downloads only Qwen3's tokeniser, not the 16 GB teacher; training with the teacher is in the post's command-line steps.
""")
code("""import os, sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from notebook_setup import setup
run = setup("part8-qwen3-teacher")
sys.path.insert(0, os.path.abspath("../part3-tokenisers"))
import transformers
transformers.logging.set_verbosity_error()   # whole-file tokenising is fine here""")

md("## 1. Three supplies of pieces, on the same test stories")
code("""from tokenizer import load_tokenizer
from qwen_data import QwenTokenizer

val = open("../part3-tokenisers/data/val.txt", encoding="utf-8").read()
val_bytes = len(val.encode("utf-8"))
for name, tok in (("my 8k BPE", load_tokenizer("bpe8k")), ("GPT-2", load_tokenizer("gpt2")), ("Qwen3", QwenTokenizer())):
    print(f"{name:10} {tok.vocab_size:>8,} token cards   {len(tok.encode(val)) / val_bytes:.3f} tokens per byte")""")

md("## 2. What 151,936 token cards cost each student")
code("""import mlx.core as mx
from mlx.utils import tree_flatten
from model_mlx import MiniGPT
from train_qwen import STUDENTS

for name, size in STUDENTS.items():
    m = MiniGPT(151936, n_layer=size["n_layer"], n_head=size["n_head"], n_embd=size["n_embd"])
    total = sum(v.size for _, v in tree_flatten(m.parameters()))
    cards = m.tok_emb.weight.size
    print(f"{name:7} {total / 1e6:6.0f} million numbers, {cards / total:4.0%} token cards, "
          f"{(total - cards) / 1e6:5.0f} million in the blocks")""")

md("## 3. How much text would they need?\n\nThe Chinchilla rule of thumb: about 20 tokens of text for every number in the machine.")
code("""stories_tokens = len(QwenTokenizer().encode(open("../part3-tokenisers/data/train.txt", encoding="utf-8").read()))
print(f"the practice stories: {stories_tokens / 1e6:.1f} million Qwen3 tokens")
for name, numbers, batch in (("scaled", 103e6, 16), ("large", 357e6, 12), ("xl", 588e6, 8)):
    seen = 3000 * batch * 256
    print(f"{name:7} needs about {numbers * 20 / 1e9:4.1f} billion tokens, "
          f"{numbers * 20 / stories_tokens:6,.0f} times the stories; in 3,000 steps it saw {seen / stories_tokens:.1f} passes")""")

nbformat.write(nb, "minigpt_follow_along_8.ipynb")
print(len(nb.cells), "cells")
