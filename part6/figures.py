"""Part 6 figure: peak memory vs context length, full attention vs sliding window."""
import json
import os

import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(__file__), "runs")


def main():
    rows = json.load(open(os.path.join(RUNS, "mem_sweep.json")))
    modes = {}
    for r in rows:
        modes.setdefault(r["mode"], []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, pts in modes.items():
        ok = [p for p in pts if p.get("peak_gb")]
        ax.plot([p["T"] for p in ok], [p["peak_gb"] for p in ok],
                "o-", label=mode, linewidth=2, markersize=5)
        for p in pts:
            if not p.get("peak_gb"):
                ax.scatter([p["T"]], [ax.get_ylim()[1] * 0.95], marker="x", s=80, color="red")
                ax.annotate("OOM", (p["T"], ax.get_ylim()[1] * 0.95), fontsize=8, color="red")

    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.set_title("One training step: full causal attention vs 256-token sliding window")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(RUNS, "mem_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
