"""Export exhibit.pt for the live demo in MiniGPT (Part 1): one flat file of
little-endian 32-bit floats (weights.bin), and a manifest.json giving each named
set of numbers its shape and offset, plus the 65-letter vocabulary.

    python export_web.py exhibit.pt OUTPUT_FOLDER
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minigpt_notebook import tiny_shakespeare  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "exhibit.pt"
out = sys.argv[2] if len(sys.argv) > 2 else "."
sd = torch.load(src, map_location="cpu")
chars = sorted(set(tiny_shakespeare()))
manifest = {"chars": chars, "n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 128, "tensors": {}}
buf, off = [], 0
for name, t in sd.items():
    if "causal_mask" in name:
        continue
    a = t.float().numpy().astype("<f4").ravel()
    manifest["tensors"][name] = {"shape": list(t.shape), "offset": off}
    buf.append(a)
    off += a.size
np.concatenate(buf).tofile(os.path.join(out, "weights.bin"))
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"))
print(off, "numbers written")
