"""Part 4 figures: the ablation bpb curves and a summary bar chart."""
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RUNS = os.path.join(HERE, "runs")

SERIES = [
    ("GPT baseline", os.path.join(HERE, "..", "part3", "runs", "history_bpe8k.json"), "#888888"),
    ("modern block", os.path.join(RUNS, "history_modern.json"), "#1f77b4"),
    ("SwiGLU -> GELU", os.path.join(RUNS, "history_rms_rope_gelu.json"), "#2ca02c"),
    ("GQA -> full MHA", os.path.join(RUNS, "history_rms_rope_mha.json"), "#9467bd"),
    ("RMSNorm -> LayerNorm", os.path.join(RUNS, "history_layer_rope.json"), "#17becf"),
    ("RoPE -> learned pos", os.path.join(RUNS, "history_rms_learned.json"), "#d62728"),
]


def load(path):
    return json.load(open(path))


def curves(out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path, color in SERIES:
        h = load(path)["history"]
        ax.plot([p["step"] for p in h], [p["bpb"] for p in h], label=label,
                color=color, marker="o", markersize=3,
                linewidth=2.5 if label in ("GPT baseline", "modern block") else 1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("The modern block, and each feature turned back off")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.62, 1.0)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def bars(out):
    labels, bpb, params = [], [], []
    for label, path, _ in SERIES:
        d = load(path)
        labels.append(label.replace(" -> ", "\n-> ").replace("modern block", "modern"))
        bpb.append(min(p["bpb"] for p in d["history"]))
        params.append(d["params"] / 1e6)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    colors = ["#888888", "#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#d62728"]
    bar = ax.bar(labels, bpb, color=colors)
    ax.set_ylabel("best validation bits per byte")
    ax.set_ylim(0.64, 0.71)
    for b, p in zip(bar, params):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                f"{b.get_height():.4f}\n{p:.1f}M", ha="center", fontsize=9)
    ax.set_title("Best bits per byte and parameter count")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    curves(os.path.join(RUNS, "ablation_curves.png"))
    bars(os.path.join(RUNS, "ablation_bars.png"))
