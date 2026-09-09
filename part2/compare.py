"""Plot the three runs on one bits-per-byte axis and print a summary table.

Writes data/bpb_comparison.png (the curves) and prints a table that
../tools/render_term.py can turn into a figure.
"""
import json
import os

import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "data")
RUNS = ["char", "bpe8k", "gpt2"]
COLORS = {"char": "#888888", "bpe8k": "#1f77b4", "gpt2": "#d62728"}


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for name in RUNS:
        path = os.path.join(DATA, f"history_{name}.json")
        if not os.path.exists(path):
            print(f"skipping {name} - no history file")
            continue
        h = json.load(open(path))
        steps = [p["step"] for p in h["history"]]
        bpb = [p["bpb"] for p in h["history"]]
        style = dict(marker="o", markersize=3)
        if name == "gpt2":
            style.update(linestyle="--", dashes=(4, 3))
        ax.plot(steps, bpb, label=name, color=COLORS[name], **style)
        best = min(h["history"], key=lambda p: p["val"])
        rows.append((name, h["vocab_size"], h["tokens_per_byte"], h["params"],
                     best["step"], best["val"], best["bpb"], h["elapsed_sec"] / 60))

    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("MiniGPT on TinyStories - three tokenizers, one model")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(DATA, "bpb_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.join(DATA, 'bpb_comparison.png')}\n")

    hdr = f"{'tokenizer':<10}{'vocab':>8}{'tok/byte':>10}{'params':>13}{'best step':>11}{'val loss':>10}{'val bpb':>9}{'minutes':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, vocab, tpb, params, step, val, bpb, mins in rows:
        print(f"{name:<10}{vocab:>8,}{tpb:>10.3f}{params:>13,}{step:>11,}{val:>10.4f}{bpb:>9.4f}{mins:>9.1f}")


if __name__ == "__main__":
    main()
