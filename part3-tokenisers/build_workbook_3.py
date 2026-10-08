"""Builds minigpt_follow_along_3.ipynb, the follow-along workbook for MiniGPT (Part 3).

    python build_workbook_3.py [ITERS]

The workbook clones this repository and runs this folder's own scripts, so it always
matches the code the post describes.
"""
import sys

import nbformat

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
POST = "https://haddley.github.io/posts/minigpt2/"
nb = nbformat.v4.new_notebook()
nb.metadata = {"colab": {"name": "minigpt_follow_along_3.ipynb", "provenance": []},
               "accelerator": "GPU",
               "kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}}
cells = nb.cells
def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbformat.v4.new_code_cell(s.strip("\n")))

md(f"""
# MiniGPT, three ways to cut text: follow along

This workbook goes with my post [MiniGPT (Part 3)]({POST}). It builds the three tokenisers, checks every number in the post, trains the same machine three times, once with each tokeniser, and compares them fairly, in bits per byte.

It runs this repository's own `part3-tokenisers` scripts, so it always matches the code the post describes. In Colab, choose **Runtime → Change runtime type → GPU** first: the three training runs need a GPU. Everything before the training runs works on any runtime.
""")

md("## 1. Get the code and the stories")
code("""import os, subprocess, sys

def run(*args):
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    print(result.stdout + result.stderr)
    result.check_returncode()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "tiktoken==0.14.0", "tokenizers==0.23.2"], check=True)
if not os.path.exists("minigpt-series"):
    subprocess.run(["git", "clone", "-q", "https://github.com/Haddley/minigpt-series.git"], check=True)
os.chdir("minigpt-series/part3-tokenisers")
sys.path.insert(0, os.getcwd())
print(os.getcwd())""")
md("`prepare_data.py` downloads about 22 MB of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and locks the last 10% away as the test stories.")
code("""run("prepare_data.py")""")

md("## 2. Three tokenisers\n\n`tokenizers_setup.py` builds the letters tokeniser, loads GPT-2's, and trains my own 8,192-piece BPE, then prints its first fifteen glues (merges). `Ġ` stands for a space attached to the front of a piece.")
code("""run("tokenizers_setup.py")""")
md("The post's sentence, cut three ways:")
code("""from tokenizer import load_tokenizer
toks = {name: load_tokenizer(name) for name in ("char", "bpe8k", "gpt2")}
sentence = "Once upon a time, there was a little dog named Spot."
for name, tok in toks.items():
    ids = tok.encode(sentence)
    print(f"{name:6} {tok.vocab_size:>6,} token embeddings  {len(ids):>2} tokens: " + "|".join(tok.decode([i]) for i in ids))
    if name != "char":
        print(" " * 8 + "IDs:", ids)""")

md("## 3. BPE by hand\n\nThe post's exercise, done in code: glue the most common neighbouring pair three times, taking the first pair when two tie.")
code("""from collections import Counter
pieces = list("the cat sat on the mat")
print(len(pieces), "pieces to start")
for step in range(1, 4):
    pairs = list(zip(pieces, pieces[1:]))
    counts = Counter(pairs)
    best = max(counts.values())
    a, b = next(p for p in pairs if counts[p] == best)
    glued, i = [], 0
    while i < len(pieces):
        if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == (a, b):
            glued.append(a + b); i += 2
        else:
            glued.append(pieces[i]); i += 1
    pieces = glued
    print(f"glue {step}: {a!r} + {b!r}, {best} times -> {len(pieces)} pieces")
print(pieces)""")

md("## 4. What bigger pieces buy: more text per position")
code("""val = open("data/val.txt", encoding="utf-8").read()
val_bytes = len(val.encode("utf-8"))
for name, tok in toks.items():
    print(f"{name:6} {len(tok.encode(val)) / val_bytes:.4f} tokens per byte")

stories = [s.strip() for s in val.split("<|endoftext|>") if s.strip()]
lengths = sorted(len(s) for s in stories)
words = val.split()
print(f"\\n{len(stories):,} test stories; the middle-sized one is {lengths[len(lengths) // 2]} letters long")
print(f"{len(val) / len(words):.2f} letters per word, counting its space: 256 letters is about {256 / (len(val) / len(words)):.0f} words")
story = next(s for s in stories if 900 < len(s) < 1100)
print(f"a {len(story)}-letter story needs {len(toks['bpe8k'].encode(story))} of the 256 positions with my 8k BPE")""")

md("## 5. What bigger pieces cost: token embeddings")
code("""from model import MiniGPT
for name, tok in toks.items():
    m = MiniGPT(tok.vocab_size)
    total = sum(p.numel() for p in m.parameters())
    emb = m.tok_emb.weight.numel()
    print(f"{name:6} {tok.vocab_size:>6,} embeddings: {emb:>10,} numbers in the embeddings, "
          f"{total - emb:,} in the blocks, {total / 1e6:.1f} million in all ({emb / total:.0%} embeddings)")""")

md(f"""
## 6. The race

Train the same machine three times, {ITERS:,} steps each, measuring bits per byte every 300 steps. On my Mac Studio's GPU, the three runs took about 15, 15, and 40 minutes. On a Colab GPU, expect them to take a similar time.
""")
code(f"""ITERS = {ITERS}
for name in ("char", "bpe8k", "gpt2"):
    run("train.py", "--tokenizer", name, "--iters", str(ITERS), "--eval-interval", "300")""")
code("""run("compare.py")
from IPython.display import Image, display
display(Image("data/bpb_comparison.png"))""")

md("## 7. Bits per byte, worked out\n\nFrom the best checkpoint of each run: surprise per token, turned into halvings (÷ 0.693) and spread over the bytes each token covers.")
code("""import json, math
for name in ("char", "bpe8k", "gpt2"):
    h = json.load(open(f"data/history_{name}.json"))
    best = min(h["history"], key=lambda p: p["val"])
    bits = best["val"] / math.log(2)
    print(f"{name:6} surprise {best['val']:.4f} per token = {bits:.3f} bits per token "
          f"x {h['tokens_per_byte']:.4f} tokens per byte = {bits * h['tokens_per_byte']:.4f} bits per byte")""")

md("## 8. What it writes")
code("""for name in ("char", "bpe8k"):
    print(f"===== {name} =====")
    run("generate.py", "--tokenizer", name, "--prompt", "Once upon a time", "--max-new-tokens", "300" if name == "char" else "120")""")

nbformat.write(nb, "minigpt_follow_along_3.ipynb")
print(len(nb.cells), "cells")
