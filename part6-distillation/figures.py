"""Part 6 figures: distillation bpb curves and a summary."""
import json
import math
import os

import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(__file__), "runs")
SERIES = [
    ("baseline (no teacher)",        "history_baseline.json",       "#888888", "o-"),
    ("GPT-2 small (124M)",           "history_gpt2.json",           "#e377c2", "o--"),
    ("GPT-2 XL (1.5B)",              "history_gpt2xl.json",         "#d62728", "o--"),
    ("TinyStories-33M (on-domain)",  "history_tinystories33m.json", "#2ca02c", "s-"),
    ("MiniGPT-512 (on-domain, 51M)", "history_big.json",            "#1f77b4", "s-"),
]


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rows = []
    for label, fname, color, ls in SERIES:
        d = json.load(open(os.path.join(RUNS, fname)))
        h = [p for p in d["history"] if p["step"] >= 300]
        ax.plot([p["step"] for p in h], [p["bpb"] for p in h], ls,
                label=label, color=color, markersize=4)
        best = min(d["history"], key=lambda p: p["val"])
        rows.append((label, best["bpb"], d["elapsed_sec"] / 60, d.get("peak_gb")))

    tc = json.load(open(os.path.join(RUNS, "teacher_config.json")))
    tpb = json.load(open(os.path.join(RUNS, "history_big.json")))["tokens_per_byte"]
    floor = tc["best_val"] / math.log(2) * tpb
    ax.axhline(floor, ls=":", color="#1f77b4", alpha=0.6, label=f"MiniGPT-512 teacher ({floor:.3f})")

    ax.set_ylim(0.60, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Distillation: the teacher's advantage on the data is what counts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(RUNS, "distill_curves.png"), dpi=150, bbox_inches="tight")
    print("wrote distill_curves.png\n")

    print(f"{'teacher':<30}{'best bpb':>10}{'minutes':>9}{'peak GB':>9}")
    for label, bpb, mins, peak in rows:
        print(f"{label:<30}{bpb:>10.4f}{mins:>9.1f}{(peak or 0):>9.2f}")


if __name__ == "__main__":
    main()
