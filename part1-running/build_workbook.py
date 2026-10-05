"""Builds minigpt_follow_along.ipynb, the follow-along workbook for MiniGPT (Part 1).

    python build_workbook.py
"""
import nbformat
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minigpt_notebook import part1_cells  # noqa: E402

config_src, attn_src, ffn_src, block_src, model_src = part1_cells()

POST = "https://haddley.github.io/posts/minigpt/"
SITE = sys.argv[1] if len(sys.argv) > 1 else "https://haddley.github.io/minigpt-demo"   # the trained model files
REPO = sys.argv[2] if len(sys.argv) > 2 else "https://raw.githubusercontent.com/Haddley/minigpt-series/main/part1-running"
nb = nbformat.v4.new_notebook()
nb.metadata = {"colab": {"name": "minigpt_follow_along.ipynb", "provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}}
cells = nb.cells
def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbformat.v4.new_code_cell(s.strip("\n")))

md(f"""
# MiniGPT, running: follow along

This workbook goes with my post [MiniGPT (Part 1)]({POST}). It runs my trained MiniGPT model, the *exhibit*, and reproduces every number in the post, step by step, from the letter IDs to the wheel of chances. It only *runs* the model; how the model was trained is the subject of [Part 2](https://haddley.github.io/posts/minigpt-grown/).

Run the cells from top to bottom. A CPU is enough: nothing here needs a GPU.

The model code in section 1 is copied from Jibin Joseph's [MiniGPT notebook](https://github.com/jibin10/MiniGPT), sections 1.1 to 1.5, under its MIT licence (notice at the end).
""")

md("## 1. The model code\n\nThese cells are the notebook's own classes, unchanged: the settings (`GPTConfig`), attention (`CausalSelfAttention`), the MLP (`FeedForward`), one block (`TransformerBlock`), and the whole model (`MiniGPT`). They define the machine, but contain no trained numbers.")
code("""import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F""")
for s in (config_src, attn_src, ffn_src, block_src, model_src):
    code(s)

md(f"""
## 2. Load my trained numbers

The checkpoint `exhibit.pt` holds all 826,433 fixed numbers. `chars` is the vocabulary: the 65 letters of Tiny Shakespeare, sorted, so that each letter's ID is its place in the list.
""")
code(f"""import urllib.request
urllib.request.urlretrieve("{SITE}/exhibit.pt", "exhibit.pt")

chars = list("\\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
stoi = {{ch: i for i, ch in enumerate(chars)}}

model = MiniGPT(GPTConfig())
model.load_state_dict(torch.load("exhibit.pt", map_location="cpu"))
model.eval()                 # running, not training: dropout off
model.requires_grad_(False)  # and no records for training

print(sum(p.numel() for p in model.parameters()), "fixed numbers loaded")""")

md("A small helper: the chances for the next letter after any text, and a way to print the top few.")
code("""def chances(text, temperature=1.0):
    idx = torch.tensor([[stoi[ch] for ch in text]])
    logits, _ = model(idx[:, -model.config.block_size:])
    return torch.softmax(logits[0, -1] / temperature, dim=-1)

def show_top(p, n=5):
    for prob, i in zip(*torch.topk(p, n)):
        print(f"{chars[i]!r:6} {prob.item():6.1%}")""")

md("## 3. The guessing game\n\nThe opening example, exactly as the live demo starts: the speaker's name, a new line, and the line with its last letter hidden.")
code("""show_top(chances("KING RICHARD III:\\nA horse! a horse! my kingdom for a hors"), 3)""")
md("After `go`, the chances are spread out. One more letter, and they are not:")
code("""show_top(chances("go"), 4)
print()
show_top(chances("goo"), 4)""")

md("## 4. Step 1: letters to numbers")
code("""ids = [stoi[ch] for ch in "goo"]
ids""")

md("## 5. Step 2: letter cards, position cards, and working cards\n\nThe letter cards and position cards are fixed tables. Adding a letter card to its position card makes a working card, one per position.")
code("""g_card   = model.token_embedding.weight[stoi["g"]]   # the g letter card
pos1     = model.position_embedding.weight[0]         # the position 1 card (Python counts from 0)
working1 = g_card + pos1                              # working card 1, before block 1
print("g letter card:    ", g_card[:4])
print("position 1 card:  ", pos1[:4])
print("working card 1:   ", working1[:4])""")
md("All three working cards for `goo`, exactly as `MiniGPT.forward` makes them:")
code("""idx = torch.tensor([ids])
x = model.token_embedding(idx) + model.position_embedding(torch.arange(3))
print(x.shape)   # 1 text, 3 working cards, 128 numbers each""")
md("Neighbouring position cards end up alike, and far-apart ones point the opposite way (the post's position ruler):")
code("""P = F.normalize(model.position_embedding.weight, dim=1)
sim = P @ P.T
print("neighbours:     ", round(sim.diagonal(1).mean().item(), 2))
print("100 apart:      ", round(sim.diagonal(100).mean().item(), 2))""")

md("## 6. Step 3, inside block 1: attention\n\nNormalise the working cards, then make the query, key, and value cards with block 1's three fixed recipes.")
code("""block1 = model.blocks[0]
xn = block1.ln1(x)                      # normalised working cards
q = block1.attn.query(xn)[0]            # query cards: 3 x 128
k = block1.attn.key(xn)[0]
v = block1.attn.value(xn)[0]
print("working card 3, normalised:", xn[0, 2, :3])
print("its query card, head 1:   ", q[2, :3])""")
md("Every number on a query card is a weighted mix of all 128 numbers on the working card, plus a bias. Here is the second number of working card 3's query card, worked out by hand:")
code("""W, b = block1.attn.query.weight, block1.attn.query.bias
terms = W[1] * xn[0, 2]                 # 128 weight x number products
print("first three terms:", [round(t, 3) for t in terms[:3].tolist()])
print("sum of 128 terms + bias:", round((terms.sum() + b[1]).item(), 2))""")
md("Head 1 uses the first 32 numbers of each card. Match working card 3's query against every key, shrink by √32, and share out with softmax:")
code("""hd = 32
scores = q[2, :hd] @ k[:, :hd].T        # query x key, added up
shrunk = scores / math.sqrt(hd)
shares = torch.softmax(shrunk, dim=-1)
for name, row in [("query x key", scores), ("divided by sqrt(32)", shrunk), ("share of attention", shares)]:
    print(f"{name:20}", [round(t, 2) for t in row.tolist()])""")
md("All four heads of block 1, for working card 3 (each head uses its own 32-number piece):")
code("""for h in range(4):
    s = slice(h * hd, (h + 1) * hd)
    sh = torch.softmax(q[2, s] @ k[:, s].T / math.sqrt(hd), dim=-1)
    print(f"head {h + 1}:", [f"{t:.1%}" for t in sh.tolist()])""")

md("## 7. Four blocks in a row\n\nEach block adds to the working cards: working card out = working card in + what attention adds + what the MLP adds. Here is what each block does to working card 3.")
code("""x0 = x.clone()
x_run = x.clone()
for n, blk in enumerate(model.blocks, start=1):
    a = blk.attn(blk.ln1(x_run)); y = x_run + a
    d = blk.mlp(blk.ln2(y));      z = y + d
    print(f"block {n}: size in {x_run[0,2].norm():.2f}, attention adds {a[0,2].norm():.2f}, "
          f"MLP adds {d[0,2].norm():.2f}, size out {z[0,2].norm():.2f}, "
          f"in-vs-out likeness {F.cosine_similarity(x_run[0,2], z[0,2], dim=0):.2f}")
    x_run = z
print("likeness to the input embedding after block 4:", round(F.cosine_similarity(x0[0,2], x_run[0,2], dim=0).item(), 2))""")
md("Stopping early (the *logit lens*): send the last working card to the answer cards after each block.")
code("""text = "First Citizen:\\nBefore we proceed any further, hear me spea"
idx2 = torch.tensor([[stoi[ch] for ch in text]])
h = model.token_embedding(idx2) + model.position_embedding(torch.arange(idx2.shape[1]))
def peek(h):
    p = torch.softmax(model.lm_head(model.final_ln(h[0, -1])), dim=-1)
    return ", ".join(f"{chars[i]} {p[i]:.0%}" for i in torch.topk(p, 3).indices)
print("before block 1:", peek(h))
for n, blk in enumerate(model.blocks, start=1):
    h = blk(h)
    print(f"after block {n}: ", peek(h))""")

md("## 8. Step 4: the answer cards\n\nNormalise the last working card once more, then score it against all 65 answer cards: multiply, add up, add the bias.")
code("""last = model.final_ln(x_run[0, 2])
scores = model.lm_head.weight @ last + model.lm_head.bias
for ch in "dkrs":
    print(repr(ch), "score", round(scores[stoi[ch]].item(), 2))
print("lowest:", repr(chars[scores.argmin()]), round(scores.min().item(), 2))
p = torch.softmax(scores, dim=-1)
print("d chance:", f"{p[stoi['d']]:.1%}")""")
md("Is the guess just the closest letter card? Not in this model: compare the last working card with the letter cards and with the answer cards.")
code("""def rank_of_d(table):
    sim = F.cosine_similarity(last[None], table, dim=1)
    return (sim.argsort(descending=True) == stoi["d"]).nonzero().item() + 1
print("d's rank among the letter cards:", rank_of_d(model.token_embedding.weight))
print("d's rank among the answer cards:", rank_of_d(model.lm_head.weight))""")

md("## 9. Step 5: temperature, trimming, and spinning the wheel")
code("""for t in (0.5, 1.0, 2.0):
    p = chances("good m", temperature=t)
    print(f"temperature {t}:", ", ".join(f"{ch} {p[stoi[ch]]:.1%}" for ch in "yea"))""")
md("Top-p 90%: keep the biggest slices until they add up to 90%, then stretch them to fill the wheel.")
code("""p = chances("good m")
order = p.argsort(descending=True)
kept, total = [], 0.0
for i in order.tolist():
    kept.append(i); total += p[i].item()
    if total >= 0.9: break
print(len(kept), "slices kept:", [chars[i] for i in kept], f"holding {total:.1%}")""")
md("Spin the wheel 200 times. The first line fixes the spins, so you get the same text as the post; remove it for a different text each time.")
code("""torch.manual_seed(1)
idx = torch.tensor([[stoi[ch] for ch in "ROMEO:\\n"]])
for _ in range(200):
    logits, _ = model(idx[:, -model.config.block_size:])
    probs = torch.softmax(logits[0, -1] / 0.8, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_id.view(1, 1)], dim=1)
print("".join(chars[i] for i in idx[0].tolist()))""")

md("## 10. Where every fixed number lives")
code("""for name, t in model.state_dict().items():
    if name.startswith("blocks.") and not name.startswith("blocks.0."):
        continue
    if "causal_mask" in name:
        continue
    print(f"{name:32} {str(tuple(t.shape)):12} {t.numel():>7,}")
print("total:", f"{sum(p.numel() for p in model.parameters()):,}")""")

md(f"""
## 11. The same model in llama.cpp, via GGUF

[llama.cpp](https://github.com/ggml-org/llama.cpp) runs models stored as a single GGUF file. My [export script](https://github.com/Haddley/minigpt-series/blob/main/part1-running/export_gguf.py) maps MiniGPT onto llama.cpp's GPT-2 design. This cell exports the file, then gets llama.cpp: the Homebrew one if you have it (`brew install llama.cpp` on a Mac), or else the official Linux build, which is what Colab uses.
""")
code(f"""import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "gguf"], check=True)
urllib.request.urlretrieve("{REPO}/export_gguf.py", "export_gguf.py")
result = subprocess.run([sys.executable, "export_gguf.py", "exhibit.pt", "exhibit.gguf"],
                        check=True, capture_output=True, text=True)
print(result.stdout.strip(), "-", os.path.getsize("exhibit.gguf"), "bytes")""")
code("""import os, shutil, tarfile
llama = shutil.which("llama-completion")
if llama is None:   # Colab and other Linux machines: download the official build
    tag = "b11406"
    urllib.request.urlretrieve(
        f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-x64.tar.gz", "llama.tar.gz")
    tarfile.open("llama.tar.gz").extractall()
    llama = os.path.abspath(f"llama-{tag}/llama-completion")
    os.environ["LD_LIBRARY_PATH"] = os.path.dirname(llama) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
print(llama)""")
code("""out = subprocess.run([llama, "-m", "exhibit.gguf", "-p", "ROMEO:\\n", "-n", "120",
                      "--temp", "0.8", "--top-k", "0", "--top-p", "1.0", "--min-p", "0"],
                     capture_output=True, text=True)
print(out.stdout)""")

md("""
---
### Licence notice for the model code in section 1

MiniGPT, Copyright (c) 2026 Jibin Joseph, https://github.com/jibin10/MiniGPT. The code is licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of the software and associated documentation files covered by the MIT License (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
""")

nbformat.write(nb, "minigpt_follow_along.ipynb")
print(len(nb.cells), "cells")
