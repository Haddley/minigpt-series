"""Part 8 figures: the student-size sweep, with and without the Qwen3-8B-Base teacher."""
import json
import os

import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(__file__), "runs")

SIZES = ["tiny", "scaled", "large", "xl"]
LABELS = {"tiny": "tiny\n69M", "scaled": "scaled\n103M", "large": "large\n357M", "xl": "xl\n588M"}


def load(tag):
    f = os.path.join(RUNS, f"history_{tag}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    d["best"] = min(d["history"], key=lambda p: p["val"])["bpb"]
    return d


def sweep(out):
    base = [load(f"{s}_base") for s in SIZES]
    dist = [load(f"{s}_qwen8b") for s in SIZES]
    x = [d["params"] / 1e6 for d in base]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, [d["best"] for d in base], "o-", color="#888888", markersize=8, label="no teacher")
    ax.plot(x, [d["best"] for d in dist], "s-", color="#1f77b4", markersize=8,
            label="distilled from Qwen3-8B-Base")
    for xi, b, d in zip(x, base, dist):
        delta = d["best"] - b["best"]
        ax.annotate(f"{delta:+.3f}", (xi, min(b["best"], d["best"]) - 0.03),
                    ha="center", fontsize=9, color="#d62728" if delta > 0 else "#2ca02c")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SIZES])
    ax.minorticks_off()
    ax.set_xlabel("student size")
    ax.set_ylabel("best validation bits per byte")
    ax.set_title("Distilling from a frontier teacher, across student sizes")
    ax.axhline(0.592, ls=":", color="#1f77b4", alpha=0.5)
    ax.text(x[0], 0.60, "Qwen3-8B-Base teacher (0.592)", fontsize=9, color="#6b7280")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

    print(f"\n{'size':<8}{'params':>13}{'base':>9}{'teacher':>9}{'delta':>9}")
    for s, b, d in zip(SIZES, base, dist):
        if b and d:
            print(f"{s:<8}{b['params']:>13,}{b['best']:>9.4f}{d['best']:>9.4f}{d['best']-b['best']:>+9.4f}")


def curves(out):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"tiny": "#aaaaaa", "scaled": "#888844", "large": "#d62728", "xl": "#7b3294"}
    for s in SIZES:
        for teach, ls in (("base", "-"), ("qwen8b", "--")):
            d = load(f"{s}_{teach}")
            if not d:
                continue
            h = [p for p in d["history"] if p["step"] >= 300]
            ax.plot([p["step"] for p in h], [p["bpb"] for p in h], ls, color=colors[s],
                    label=f"{s}{'  + teacher' if teach == 'qwen8b' else ''}", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Part 8 training curves — four student sizes")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    sweep(os.path.join(RUNS, "qwen_sweep.png"))
    curves(os.path.join(RUNS, "qwen_curves.png"))
