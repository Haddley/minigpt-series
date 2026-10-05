"""How close did each student get to each teacher? Tests the capacity-gap explanation.

For every (teacher, student) pair, on the same test batches: the KL divergence from the
teacher's wheel to the student's (0 = identical wheels), and how often their top slice agrees.
"""
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part4-mlx"))
from data import load_split  # noqa: E402
from model_mlx import MiniGPT  # noqa: E402
from eval_teacher import make_forward  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "runs")
TEACHERS = {"MiniGPT-512": os.path.join(OUT, "teacher.safetensors"),
            "TinyStories-33M": "torch:roneneldan/TinyStories-33M"}
STUDENTS = {"no teacher": "gap_baseline", "taught by MiniGPT-512": "gap_big",
            "taught by TinyStories-33M": "gap_ts33m"}


def main():
    _, val_ids, tok, _ = load_split("gpt2")
    rng = np.random.default_rng(0)
    xs = []
    for _ in range(30):
        i = rng.integers(0, len(val_ids) - 257, size=8)
        xs.append(np.stack([val_ids[j:j + 256] for j in i]))
    teachers = {name: make_forward(path, tok) for name, path in TEACHERS.items()}
    students = {}
    for sname, tag in STUDENTS.items():
        m = MiniGPT(tok.vocab_size, block_size=256)
        m.load_weights(list(mx.load(os.path.join(OUT, f"ckpt_{tag}.safetensors")).items()))
        m.eval()
        students[sname] = m
    sums = {(s, t): [0.0, 0.0] for s in students for t in teachers}
    for x in xs:                                   # one batch at a time, to keep memory small
        t_lp = {t: nn.log_softmax(f(x), axis=-1) for t, f in teachers.items()}
        for sname, m in students.items():
            s_lp = nn.log_softmax(m(mx.array(x)), axis=-1)
            for tname, tl in t_lp.items():
                sums[(sname, tname)][0] += (mx.exp(tl) * (tl - s_lp)).sum(-1).mean().item()
                sums[(sname, tname)][1] += (mx.argmax(tl, -1) == mx.argmax(s_lp, -1)).mean().item()
    results = {}
    for (sname, tname), (kl, agree) in sums.items():
        results[f"{sname} | {tname}"] = {"kl": kl / len(xs), "top1_agree": agree / len(xs)}
        print(f"{sname:28} vs {tname:16}  KL {kl / len(xs):.4f}   same top slice {agree / len(xs):.1%}")
    json.dump(results, open(os.path.join(OUT, "capacity_gap.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
