"""How peak training memory scales with context length, three ways:

  full     full causal attention, the fused MLX kernel
  naive    a [T, T] sliding-window mask handed to the same fused kernel
  chunked  chunked_swa - attention computed in 2w-wide bands, never T x T

The point of the naive row is that it saves nothing: the kernel still builds the
full score matrix. chunked is the one whose memory grows linearly in T.
"""
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part4"))
from model_llama import Config, MiniLlama  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "runs")
# full/naive attention is O(T^2); stop trying them past FULL_MAX so the sweep
# does not thrash swap. chunked_swa is O(T*w) and keeps going.
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
FULL_MAX = 4096
WINDOW = 256
BATCH = 8
VOCAB = 8192
STEPS = 5


MODES = {
    "full":    dict(window=0),
    "naive":   dict(window=WINDOW, naive_window=True),
    "chunked": dict(window=WINDOW, naive_window=False),
}


def run(block_size, mode):
    mx.clear_cache()
    mx.reset_peak_memory()
    model = MiniLlama(Config(vocab_size=VOCAB, block_size=block_size, **MODES[mode]))
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-3)
    lg = nn.value_and_grad(model, MiniLlama.loss)
    rng = np.random.default_rng(0)

    def step():
        x = mx.array(rng.integers(0, VOCAB, (BATCH, block_size)).astype(np.int32))
        y = mx.array(rng.integers(0, VOCAB, (BATCH, block_size)).astype(np.int32))
        loss, grads = lg(model, x, y)
        opt.update(model, grads)
        mx.eval(model.state, opt.state)

    step()
    t0 = time.time()
    for _ in range(STEPS):
        step()
    dt = (time.time() - t0) / STEPS
    peak = (mx.get_peak_memory() if hasattr(mx, "get_peak_memory")
            else mx.metal.get_peak_memory()) / 1e9
    return peak, dt


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for T in LENGTHS:
        for mode in ("full", "naive", "chunked"):
            if mode in ("full", "naive") and T > FULL_MAX:
                print(f"{mode:<9} T={T:<6} skipped (O(T^2), past FULL_MAX)")
                rows.append({"mode": mode, "T": T, "peak_gb": None, "error": "skipped"})
                continue
            try:
                peak, dt = run(T, mode)
                print(f"{mode:<9} T={T:<6} peak {peak:7.2f} GB   {dt * 1000:8.1f} ms/step")
                rows.append({"mode": mode, "T": T, "peak_gb": round(peak, 3),
                             "ms_step": round(dt * 1000, 1)})
            except Exception as e:  # noqa: BLE001
                print(f"{mode:<9} T={T:<6} FAILED: {type(e).__name__}")
                rows.append({"mode": mode, "T": T, "peak_gb": None,
                             "error": f"{type(e).__name__}: {str(e)[:100]}"})
            json.dump(rows, open(os.path.join(OUT, "mem_sweep.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'mem_sweep.json')}")


if __name__ == "__main__":
    main()
