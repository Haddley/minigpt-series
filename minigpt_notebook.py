"""Shared helpers for part1-running and part2-growing.

Both parts use the model code from Jibin Joseph's MiniGPT notebook
(https://github.com/jibin10/MiniGPT, MIT licence for code). This module downloads
the notebook at a pinned commit, so that every script and workbook here uses
exactly the version the posts were written against, and can run its Part 1 cells
(sections 1.1 to 1.5), which define GPTConfig and the MiniGPT classes.
"""
import json
import os
import urllib.request

COMMIT = "cd4f62a4e34c787dd7cbdc26fff8b3558cd1fd70"
NOTEBOOK_URL = f"https://raw.githubusercontent.com/jibin10/MiniGPT/{COMMIT}/MiniGPT_Notebook.ipynb"
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _fetch(url, name):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def part1_cells():
    """The notebook's code cells for sections 1.1 to 1.5, as source strings."""
    nb = json.load(open(_fetch(NOTEBOOK_URL, "MiniGPT_Notebook.ipynb")))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return code[1:6]   # GPTConfig, CausalSelfAttention, FeedForward, TransformerBlock, MiniGPT


def load_model_classes(namespace):
    """Run the Part 1 cells into `namespace` (for example, globals())."""
    exec("import math\nfrom dataclasses import dataclass\nimport torch\n"
         "import torch.nn as nn\nimport torch.nn.functional as F", namespace)
    for src in part1_cells():
        exec(src, namespace)


def tiny_shakespeare():
    """The Tiny Shakespeare text the notebook trains on (about 1.1 million letters)."""
    return open(_fetch(TINY_SHAKESPEARE_URL, "input.txt"), encoding="utf-8").read()
