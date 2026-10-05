"""Shared first cell for the Mac follow-along notebooks (Parts 4 to 7).

These parts use MLX, which needs a Mac with Apple Silicon, so their notebooks are for
Jupyter on a Mac rather than Colab. Each notebook calls `setup(folder)`, which:
  - finds this repository (or clones it next to the notebook),
  - moves into the part's folder so its scripts can be imported and run,
  - prepares Part 3's TinyStories split and tokenisers if they are missing,
and returns `run`, which runs one of the folder's scripts and prints its output.
"""
import os
import platform
import subprocess
import sys

REPO_URL = "https://github.com/Haddley/minigpt-series.git"


def _repo_root():
    here = os.path.abspath(os.getcwd())
    for path in (here, os.path.dirname(here)):
        if os.path.exists(os.path.join(path, "part3-tokenisers", "tokenizer.py")):
            return path
    if not os.path.exists("minigpt-series"):
        subprocess.run(["git", "clone", "-q", REPO_URL], check=True)
    return os.path.abspath("minigpt-series")


def run(*args):
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    print(result.stdout + result.stderr)
    result.check_returncode()


def setup(folder):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("Warning: MLX needs a Mac with Apple Silicon; the MLX cells will not run here.")
    root = _repo_root()
    data = os.path.join(root, "part3-tokenisers", "data")
    if not os.path.exists(os.path.join(data, "bpe8k.json")):
        for script in ("prepare_data.py", "tokenizers_setup.py"):
            subprocess.run([sys.executable, script], cwd=os.path.join(root, "part3-tokenisers"), check=True)
    os.chdir(os.path.join(root, folder))
    sys.path.insert(0, os.getcwd())
    print("working in", os.getcwd())
    return run
