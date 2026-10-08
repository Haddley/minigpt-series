"""Builds readable_minigpt.ipynb from readable_minigpt.py, and runs it so that it is saved with outputs.

readable_minigpt.py is written in "percent" format: a line "# %%" starts a code cell, and a line
"# %% [markdown]" starts a text cell whose lines are Markdown behind "# ". The .py file runs as an
ordinary script; this builder turns the same cells into a notebook.
"""
import nbformat
from nbclient import NotebookClient

cells = []
kind, lines = None, []


def flush():
    text = "\n".join(lines).strip("\n")
    if not text:
        return
    if kind == "markdown":
        markdown = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.split("\n"))
        cells.append(nbformat.v4.new_markdown_cell(markdown))
    else:
        cells.append(nbformat.v4.new_code_cell(text))


for line in open("readable_minigpt.py").read().split("\n"):
    if line.startswith("# %%"):
        flush()
        kind, lines = ("markdown" if "[markdown]" in line else "code"), []
    else:
        lines.append(line)
flush()

notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(notebook, "readable_minigpt.ipynb")
print("wrote readable_minigpt.ipynb with", len(cells), "cells")
