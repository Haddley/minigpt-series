"""Grow the MiniGPT exhibit model from random numbers, exactly as in MiniGPT (Part 2).

3,000 steps of 32 snippets of 128 letters, with the notebook's model and settings
and every random choice fixed in advance. Five times along the way it reports the
exam score, the start of the g card, and the chance of d after "goo", and writes
from "ROMEO:". On a Mac's CPU the result is bit-for-bit identical to the published
exhibit; other machines do some arithmetic in a different order and come very close.

    python grow_exhibit.py                      # writes exhibit.pt
    python grow_exhibit.py --compare exhibit.pt # also compares with a reference
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minigpt_notebook import load_model_classes, tiny_shakespeare  # noqa: E402

load_model_classes(globals())

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="exhibit.pt")
ap.add_argument("--compare", default=None, help="a reference exhibit.pt to compare with")
args = ap.parse_args()

torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)

text = tiny_shakespeare()
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def exam_score(model, stride=64):
    scores = []
    with torch.no_grad():
        for k in range(0, len(val_data) - 129, 128 * stride):
            ix = torch.arange(k, min(k + 128 * stride, len(val_data) - 129), 128)
            x = torch.stack([val_data[i:i + 128] for i in ix])
            y = torch.stack([val_data[i + 1:i + 129] for i in ix])
            scores.append(model(x, y)[1].item())
    return sum(scores) / len(scores)


torch.manual_seed(42)
model = MiniGPT(GPTConfig(block_size=128, vocab_size=len(chars), n_layer=4, n_head=4, n_embd=128, dropout=0.1))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
batch_rng = torch.Generator().manual_seed(1337)


def snapshot(step):
    model.eval()
    rng = torch.random.get_rng_state()   # snapshots must not change the training run
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
    print("".join(chars[i] for i in idx[0].tolist()), "\n", flush=True)


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
print(f"grown in {(time.time() - t) / 60:.1f} minutes")
torch.save(model.state_dict(), args.out)
print("wrote", args.out)

if args.compare:
    ref = torch.load(args.compare, map_location="cpu")
    grown = model.state_dict()
    same = sum(torch.equal(grown[k], ref[k]) for k in ref)
    biggest = max((grown[k].float() - ref[k].float()).abs().max().item() for k in ref)
    print(f"{same} of {len(ref)} named sets of numbers identical; biggest difference {biggest:.2g}")
