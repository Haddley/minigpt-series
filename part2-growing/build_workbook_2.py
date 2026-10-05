# Builds minigpt_follow_along_2.ipynb: the follow-along workbook for MiniGPT (Part 2),
# which grows the exhibit model from random numbers and reproduces the post's numbers.
import nbformat
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minigpt_notebook import part1_cells  # noqa: E402

SITE = sys.argv[1] if len(sys.argv) > 1 else "https://haddley.github.io/minigpt-demo"
config_src, attn_src, ffn_src, block_src, model_src = part1_cells()
POST = "https://haddley.github.io/posts/minigpt-grown/"

nb = nbformat.v4.new_notebook()
nb.metadata = {"colab": {"name": "minigpt_follow_along_2.ipynb", "provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}}
cells = nb.cells
def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbformat.v4.new_code_cell(s.strip("\n")))

md(f"""
# MiniGPT, grown: follow along

This workbook goes with my post [MiniGPT (Part 2)]({POST}). It grows my MiniGPT *exhibit* model from random numbers, using the guessing game, and reproduces the post's numbers on the way: the surprise score, one training step in slow motion, and the exhibit growing, step by step. At the end, it compares what it grew with my published exhibit, number by number.

Run the cells from top to bottom on a CPU runtime. Everything is quick except the growing itself, which is 3,000 training steps: about 6 minutes on my Mac Studio's CPU, and longer on a free Colab CPU.

The model code in section 1 is copied from Jibin Joseph's [MiniGPT notebook](https://github.com/jibin10/MiniGPT), sections 1.1 to 1.5, under its MIT licence (notice at the end). The companion workbook for running the finished model is [minigpt_follow_along.ipynb](https://colab.research.google.com/github/Haddley/minigpt-series/blob/main/part1-running/minigpt_follow_along.ipynb).
""")

md("## 1. The model code\n\nThe notebook's own classes, unchanged. They define the machine; training fills in its numbers.")
code("""import math, time, urllib.request
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F""")
for s in (config_src, attn_src, ffn_src, block_src, model_src):
    code(s)

md("""
## 2. The practice text

Tiny Shakespeare, about 1.1 million letters, from the same address the notebook uses. The vocabulary is every letter in it, sorted, and the last 10% is locked away as the exam text, which training never sees.
""")
code("""urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "input.txt")
text = open("input.txt", encoding="utf-8").read()

chars = sorted(set(text))
V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
def decode(ids): return "".join(chars[i] for i in ids)

data = torch.tensor([stoi[ch] for ch in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"{len(text):,} letters, {V} different letters")
print(f"practice text: {len(train_data):,} letters; locked-away exam text: {len(val_data):,} letters")""")

md("## 3. Starting from nothing\n\nA freshly built machine has random numbers on every dial, so its wheel is almost even: every letter gets a chance close to 1 in 65.")
code("""torch.manual_seed(42)
untrained = MiniGPT(GPTConfig(vocab_size=V))
untrained.eval()
with torch.no_grad():
    logits, _ = untrained(torch.tensor([[stoi[ch] for ch in "goo"]]))
p = torch.softmax(logits[0, -1], dim=-1)
print(f"chances after 'goo': between {p.min():.1%} and {p.max():.1%}; an even wheel would give {1/V:.1%} each")

torch.manual_seed(0)
idx = torch.tensor([[stoi[ch] for ch in "ROMEO:"]])
with torch.no_grad():
    for _ in range(80):
        logits, _ = untrained(idx)
        idx = torch.cat([idx, torch.multinomial(torch.softmax(logits[0, -1], -1), 1).view(1, 1)], dim=1)
print(decode(idx[0].tolist()))""")

md(f"""
## 4. The surprise score

The surprise score is ln(1 ÷ chance): 0 for a chance of 100%, and 0.69 more for every halving. To look at real chances, load my trained exhibit model.
""")
code("""for halvings in range(7):
    chance = 0.5 ** halvings
    print(f"chance {chance:7.2%}   surprise score {math.log(1 / chance):.2f}")
print(f"\\nan even wheel, 1 in 65: surprise score {math.log(V):.2f}")""")
code(f"""urllib.request.urlretrieve("{SITE}/exhibit.pt", "exhibit.pt")
exhibit = MiniGPT(GPTConfig(vocab_size=V))
exhibit.load_state_dict(torch.load("exhibit.pt", map_location="cpu"))
exhibit.eval()
exhibit.requires_grad_(False)

logits, _ = exhibit(torch.tensor([[stoi[ch] for ch in "good m"]]))
p = torch.softmax(logits[0, -1], dim=-1)
for ch in "yea":
    print(f"after 'good m', {{ch!r}} gets {{p[stoi[ch]]:.1%}}: surprise score {{math.log(1 / p[stoi[ch]].item()):.2f}}")""")

md("""
## 5. One training step, as six steps

Growing uses Part 1's five steps, with step 5 changed and a step 6 added. Here is the very first step of growing the exhibit, in slow motion, with a fresh copy of the starting machine.
""")
code("""torch.manual_seed(42)
step_model = MiniGPT(GPTConfig(block_size=128, vocab_size=V, n_layer=4, n_head=4, n_embd=128, dropout=0.1))
step_opt = torch.optim.AdamW(step_model.parameters(), lr=3e-4)
rng = torch.Generator().manual_seed(1337)

def d_after_goo(m):
    m.eval()
    with torch.no_grad():
        logits, _ = m(torch.tensor([[stoi[ch] for ch in "goo"]]))
    m.train()
    return torch.softmax(logits[0, -1], -1)[stoi["d"]].item()

before = d_after_goo(step_model)
g_before = step_model.token_embedding.weight[stoi["g"], :4].tolist()

# step 1: 32 snippets of 128 letters, and the real next letter at every position
ix = torch.randint(len(train_data) - 128, (32,), generator=rng)
x = torch.stack([train_data[i:i + 128] for i in ix])
y = torch.stack([train_data[i + 1:i + 129] for i in ix])
print("one snippet begins:", repr(decode(x[0, :20].tolist())))

# steps 2 to 4: cards, the blocks, and a wheel for every position
logits, loss = step_model(x, y)
chance = torch.softmax(logits[0, 0], -1)[y[0, 0]].item()
print(f"its first wheel gives the real next letter, {chars[y[0, 0]]!r}, a chance of {chance:.2%}")

# step 5: check the answer
print(f"that guess's surprise score: {math.log(1 / chance):.2f}; the average over all {y.numel():,} guesses: {loss.item():.2f}")

# step 6: nudge every number
step_opt.zero_grad()
loss.backward()
step_opt.step()
g_after = step_model.token_embedding.weight[stoi["g"], :4].tolist()
print("the g card's first numbers moved from", [round(v, 5) for v in g_before], "to", [round(v, 5) for v in g_after])
print(f"the chance of d after goo went from {before:.3%} to {d_after_goo(step_model):.3%}")""")

md("""
## 6. Following the blame back, by hand

Backpropagation's key moves, worked by hand for the last link of the chain: the exhibit's guess after `good m`, where the real next letter is `y`. Three rules carry the blame from the surprise score back to the answer cards. Then PyTorch's `loss.backward()` does the same for every link, and the two should agree.
""")
code("""cap = {}
hook = exhibit.final_ln.register_forward_hook(lambda mod, inp, out: cap.update(card=out))
exhibit.requires_grad_(True)
logits, _ = exhibit(torch.tensor([[stoi[ch] for ch in "good m"]]))
hook.remove()
card = cap["card"][0, -1].detach()          # the last working card, after the final normalisation
scores = logits[0, -1]
target = stoi["y"]
loss = F.cross_entropy(scores[None], torch.tensor([target]))
print(f"surprise score: {loss.item():.3f}")

p = F.softmax(scores, dim=-1).detach()                 # the wheel: 65 chances
blame_scores = p.clone()
blame_scores[target] -= 1                              # rule 1: chance, minus 1 for the real letter
blame_cards = torch.outer(blame_scores, card)          # rule 2: each answer card's blame, scaled by the working card
blame_biases = blame_scores                            # rule 3: added, so passed on unchanged
blame_card = exhibit.lm_head.weight.detach().T @ blame_scores   # rule 2 again: back to the working card

for ch in "yeaoz":
    print(f"{ch!r}: chance {p[stoi[ch]]:.2%}, blame on its score {blame_scores[stoi[ch]]:+.3f}, "
          f"blame on its answer card's first number {blame_cards[stoi[ch], 0]:+.3f}")

exhibit.zero_grad()
loss.backward()                                        # PyTorch walks the whole chain
print("biggest difference from loss.backward():", (blame_cards - exhibit.lm_head.weight.grad).abs().max().item())
exhibit.zero_grad()
exhibit.requires_grad_(False)""")

md("""
## 7. Growing the exhibit

This is exactly how I grew the exhibit: the notebook's model and settings, 3,000 steps of 32 snippets of 128 letters, with every random choice fixed in advance. Five times along the way, it stops to sit the exam, write from `ROMEO:`, and report the start of the `g` card and the chance of `d` after `goo`.
""")
code("""def exam_score(model, stride=64):   # every 64th snippet of the locked-away exam text
    scores = []
    with torch.no_grad():
        for k in range(0, len(val_data) - 129, 128 * stride):
            ix = torch.arange(k, min(k + 128 * stride, len(val_data) - 129), 128)
            x = torch.stack([val_data[i:i + 128] for i in ix])
            y = torch.stack([val_data[i + 1:i + 129] for i in ix])
            scores.append(model(x, y)[1].item())
    return sum(scores) / len(scores)

print(f"the published exhibit's exam score: {exam_score(exhibit):.2f}")""")
code("""torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)

torch.manual_seed(42)
model = MiniGPT(GPTConfig(block_size=128, vocab_size=V, n_layer=4, n_head=4, n_embd=128, dropout=0.1))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
batch_rng = torch.Generator().manual_seed(1337)

def snapshot(step):
    model.eval()
    rng = torch.random.get_rng_state()          # so that snapshots do not change the training run
    with torch.no_grad():
        score = exam_score(model)
        sample_rng = torch.Generator().manual_seed(5)
        idx = torch.tensor([[stoi[ch] for ch in "ROMEO:"]])
        for _ in range(150):
            logits, _ = model(idx[:, -128:])
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, -1), 1, generator=sample_rng)], 1)
        logits, _ = model(torch.tensor([[stoi[ch] for ch in "goo"]]))
        d = F.softmax(logits[0, -1], -1)[stoi["d"]].item()
    torch.random.set_rng_state(rng)
    model.train()
    g_card = ", ".join(f"{v:.3f}" for v in model.token_embedding.weight[stoi["g"], :4].tolist())
    print(f"--- step {step}: exam score {score:.2f}; g card begins {g_card}; d after goo {d:.1%}")
    print(decode(idx[0].tolist()), "\\n")

t = time.time()
for step in range(3001):
    if step in (0, 100, 300, 1000, 3000):
        snapshot(step)
    if step == 3000:
        break
    ix = torch.randint(len(train_data) - 128, (32,), generator=batch_rng)
    x = torch.stack([train_data[i:i + 128] for i in ix])
    y = torch.stack([train_data[i + 1:i + 129] for i in ix])
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"grown in {(time.time() - t) / 60:.1f} minutes")""")

md("""
## 8. The payoff: is it the exhibit?

Compare every one of the 826,433 grown numbers with my published exhibit. On my Mac Studio's CPU they match exactly. A different computer does some of its arithmetic in a slightly different order, so on Colab expect a machine that is very close, but not identical to the last digit.
""")
code("""grown = model.state_dict()
mine = torch.load("exhibit.pt", map_location="cpu")
same = sum(torch.equal(grown[k], mine[k]) for k in mine)
biggest = max((grown[k].float() - mine[k].float()).abs().max().item() for k in mine)
print(f"{same} of {len(mine)} named sets of numbers identical; biggest difference {biggest:.2g}")

model.eval()
with torch.no_grad():
    logits, _ = model(torch.tensor([[stoi[ch] for ch in "goo"]]))
print(f"grown model: d after goo {torch.softmax(logits[0, -1], -1)[stoi['d']]:.1%}; "
      f"exam score {exam_score(model):.2f}")""")

md("""
---
### Licence notice for the model code in section 1

MiniGPT, Copyright (c) 2026 Jibin Joseph, https://github.com/jibin10/MiniGPT. The code is licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of the software and associated documentation files covered by the MIT License (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
""")

nbformat.write(nb, "minigpt_follow_along_2.ipynb")
print(len(nb.cells), "cells")
