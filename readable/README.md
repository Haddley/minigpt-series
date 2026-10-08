# MiniGPT, rewritten to be read

An experiment: the code that runs my already-trained MiniGPT exhibit model, rewritten for people to
read. It does not train anything.

- `readable_minigpt.py` is the whole program. It runs as an ordinary script, and it is written in
  "percent" format, so its explanation cells and code cells also make a notebook.
- `readable_minigpt.ipynb` is that notebook, saved with its outputs.
- `build_readable_notebook.py` rebuilds and reruns the notebook from the `.py` file.

Every value has its own descriptive name (no `x = x + ...`), every shape is written down, and every
library call is explained. Section 9 checks the answers against Jibin Joseph's original model code:
they agree to within rounding (about 0.000003).

The post that goes with it: [MiniGPT, rewritten to be read](https://haddley.github.io/posts/minigpt7/).

```bash
cd readable
../.venv/bin/python readable_minigpt.py          # run it as a script
../.venv/bin/python build_readable_notebook.py   # rebuild the notebook
```

The trained numbers, `exhibit.pt`, download from the site the first time it runs.
