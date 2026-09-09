"""Part 5 figures: distillation bpb curves and a summary bar chart."""
import json
import os

import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(__file__), "runs")
SERIES = [
    ("baseline (no teacher)", "history_baseline.json", "#888888"),
    ("teacher: GPT-2 small", "history_gpt2.json", "#d62728"),
    ("teacher: MiniGPT-512", "history_big.json", "#1f77b4"),
]


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for label, fname, color in SERIES:
        d = json.load(open(os.path.join(RUNS, fname)))
        h = [p for p in d["history"] if p["step"] >= 300]
        ax.plot([p["step"] for p in h], [p["bpb"] for p in h], "o-",
                label=label, color=color, markersize=4)
        best = min(d["history"], key=lambda p: p["val"])
        rows.append((label, best["bpb"], d["elapsed_sec"] / 60, d.get("peak_gb")))
    ax.set_ylim(0.60, 1.05)

    # teacher's own held-out bpb, as a floor line
    try:
        tc = json.load(open(os.path.join(RUNS, "teacher_config.json")))
        floor = tc["best_val"] / 0.6931471805599453 * json.load(
            open(os.path.join(RUNS, "history_big.json")))["tokens_per_byte"]
        ax.axhline(floor, ls=":", color="#1f77b4", alpha=0.7,
                   label=f"MiniGPT-512 teacher ({floor:.3f})")
    except Exception:  # noqa: BLE001
        pass

    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Distillation: student trained against a teacher's token probabilities")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(RUNS, "distill_curves.png"), dpi=150, bbox_inches="tight")
    print("wrote distill_curves.png")

    print(f"\n{'run':<26}{'best bpb':>10}{'minutes':>9}{'peak GB':>9}")
    for label, bpb, mins, peak in rows:
        print(f"{label:<26}{bpb:>10.4f}{mins:>9.1f}{(peak or 0):>9.2f}")


if __name__ == "__main__":
    main()
