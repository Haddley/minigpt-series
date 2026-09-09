"""Part 6 figures: peak memory vs context length (full / naive mask / chunked)."""
import json
import os

import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(__file__), "runs")
STYLE = {
    "full":    ("full causal attention", "#d62728", "o-"),
    "naive":   ("naive [T,T] window mask", "#ff9896", "s--"),
    "chunked": ("chunked window (w=256)", "#1f77b4", "D-"),
}


def main():
    rows = json.load(open(os.path.join(RUNS, "mem_sweep.json")))
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, (label, color, ls) in STYLE.items():
        pts = [p for p in by_mode.get(mode, []) if p.get("peak_gb")]
        if pts:
            ax.plot([p["T"] for p in pts], [p["peak_gb"] for p in pts], ls,
                    color=color, label=label, markersize=6, linewidth=2)

    ax.axhline(64, color="#555", ls=":", linewidth=1)
    ax.text(520, 70, "64 GB — this machine", fontsize=8, color="#555")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("peak GPU memory (GB), one training step")
    ax.set_title("Full attention scales with T²; chunked windowed attention with T")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, which="both")
    out = os.path.join(RUNS, "mem_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def train_curves():
    p4 = os.path.join(os.path.dirname(__file__), "..", "part4", "runs")
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, label, color in [("full1024", "full attention", "#d62728"),
                              ("window1024", "256-token sliding window", "#1f77b4")]:
        h = json.load(open(os.path.join(p4, f"history_{tag}.json")))["history"]
        ax.plot([p["step"] for p in h], [p["bpb"] for p in h], "o-",
                label=label, color=color, markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Training at a 1,024-token context: windowed vs full attention")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(RUNS, "train_1024.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
    train_curves()
