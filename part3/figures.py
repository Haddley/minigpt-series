"""Build the Part 3 figures from the run histories and the hard-coded bench results."""
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
P2 = os.path.join(HERE, "..", "part2", "data")
P3 = os.path.join(HERE, "runs")

# from: python bench.py --framework {torch,mlx-nocompile,mlx}  (bpe8k, batch 32, block 256)
BENCH = [
    ("PyTorch\nMPS", 48426, 4.60),
    ("MLX\nno compile", 45106, 3.72),
    ("MLX\nmx.compile", 56136, 3.89),
]
COLORS = ["#ff7f0e", "#7fb3d5", "#1f77b4"]


def loss_curves(out):
    torch_h = json.load(open(os.path.join(P2, "history_bpe8k.json")))["history"]
    mlx_h = json.load(open(os.path.join(P3, "history_bpe8k.json")))["history"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([p["step"] for p in torch_h], [p["bpb"] for p in torch_h],
            "o-", color="#ff7f0e", markersize=3, label="PyTorch - MPS")
    ax.plot([p["step"] for p in mlx_h], [p["bpb"] for p in mlx_h],
            "s--", color="#1f77b4", markersize=3, label="MLX")
    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Same model, same 8k tokenizer, same data - two frameworks")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def bench_bars(out):
    names = [b[0] for b in BENCH]
    tps = [b[1] / 1000 for b in BENCH]
    mem = [b[2] for b in BENCH]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.5))
    a1.bar(names, tps, color=COLORS)
    a1.set_ylabel("thousand tokens / sec")
    a1.set_title("training throughput")
    for i, v in enumerate(tps):
        a1.text(i, v + 0.5, f"{v:.1f}k", ha="center")
    a2.bar(names, mem, color=COLORS)
    a2.set_ylabel("peak GPU memory (GB)")
    a2.set_title("peak memory")
    for i, v in enumerate(mem):
        a2.text(i, v + 0.03, f"{v:.2f}", ha="center")
    fig.suptitle("MiniGPT training step - 13.9M params, batch 32, 256-token context")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(P3, exist_ok=True)
    loss_curves(os.path.join(P3, "loss_curves.png"))
    bench_bars(os.path.join(P3, "bench.png"))
