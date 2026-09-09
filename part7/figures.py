"""Part 7 figures: Qwen3-tokeniser student, with and without the Qwen3-8B-Base teacher."""
import json
import math
import os

import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(__file__), "runs")
DATA = os.path.join(os.path.dirname(__file__), "data")
SERIES = [
    ("tiny, no teacher",     "history_tiny_base.json",     "#aaaaaa", "o-"),
    ("tiny, Qwen3-8B-Base",   "history_tiny_qwen8b.json",   "#1f77b4", "o-"),
    ("scaled, no teacher",    "history_scaled_base.json",   "#888844", "s--"),
    ("scaled, Qwen3-8B-Base", "history_scaled_qwen8b.json", "#d62728", "s--"),
]


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rows = []
    for label, fname, color, ls in SERIES:
        path = os.path.join(RUNS, fname)
        if not os.path.exists(path):
            print("skip", fname)
            continue
        d = json.load(open(path))
        h = d["history"]
        ax.plot([p["step"] for p in h], [p["bpb"] for p in h], ls,
                label=label, color=color, markersize=4)
        best = min(h, key=lambda p: p["val"])
        rows.append((label, best["bpb"], best["step"], d["elapsed_sec"] / 60,
                     d.get("peak_gb"), d["params"], d["emb_frac"]))

    meta = os.path.join(DATA, "teacher_qwen8b_meta.json")
    if os.path.exists(meta):
        tb = json.load(open(meta))["val_bpb"]
        ax.axhline(tb, ls=":", color="#1f77b4", alpha=0.6,
                   label=f"Qwen3-8B-Base teacher ({tb:.3f})")

    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Distilling from Qwen3-8B-Base into a Qwen3-vocabulary MiniGPT")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(RUNS, "qwen_curves.png"), dpi=150, bbox_inches="tight")
    print("wrote qwen_curves.png\n")

    print(f"{'run':<26}{'best bpb':>10}{'@step':>8}{'min':>7}{'peakGB':>8}{'params':>12}{'emb%':>6}")
    for label, bpb, step, mins, peak, params, ef in rows:
        print(f"{label:<26}{bpb:>10.4f}{step:>8}{mins:>7.1f}{(peak or 0):>8.1f}"
              f"{params:>12,}{100*ef:>6.0f}")


if __name__ == "__main__":
    main()
