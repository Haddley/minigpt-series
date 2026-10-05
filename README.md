# minigpt-series

Companion code for the MiniGPT blog series: from running and growing a small,
character-level GPT, up to a modern MLX model, one change at a time, on a 2022 Mac
Studio (M1 Max, 64 GB). Each folder is one post, and the folder number is the post's
part number.

| Folder | Post | What it covers | Runs on |
|---|---|---|---|
| `part1-running/` | [MiniGPT (Part 1)](https://haddley.github.io/posts/minigpt/) | running the trained *exhibit* model, step by step; exporting it to GGUF for llama.cpp | any CPU; [Colab](https://colab.research.google.com/github/Haddley/minigpt-series/blob/main/part1-running/minigpt_follow_along.ipynb) |
| `part2-growing/` | [MiniGPT (Part 2)](https://haddley.github.io/posts/minigpt-grown/) | growing the exhibit from random numbers, exactly | any CPU; [Colab](https://colab.research.google.com/github/Haddley/minigpt-series/blob/main/part2-growing/minigpt_follow_along_2.ipynb) |
| `part3-tokenisers/` | [MiniGPT (Part 3)](https://haddley.github.io/posts/minigpt2/) | character vs GPT-2 vs a trained 8k BPE tokeniser, on TinyStories, scored in bits per byte | PyTorch: Apple GPU, CUDA, or CPU; [Colab](https://colab.research.google.com/github/Haddley/minigpt-series/blob/main/part3-tokenisers/minigpt_follow_along_3.ipynb) |
| `part4-mlx/` | [MiniGPT (Part 4)](https://haddley.github.io/posts/minigpt3/) | the same model rebuilt in Apple's MLX, benchmarked against PyTorch | Apple Silicon |
| `part5-modern-block/` | [MiniGPT (Part 5)](https://haddley.github.io/posts/minigpt4/) | the Llama 3.2 block (RMSNorm, RoPE, SwiGLU, GQA), each change ablated | Apple Silicon |
| `part6-distillation/` | [MiniGPT (Part 6)](https://haddley.github.io/posts/minigpt5/) | logit distillation from GPT-2 and from a same-data 51M teacher | Apple Silicon |
| `part7-sliding-window/` | [MiniGPT (Part 7)](https://haddley.github.io/posts/minigpt6/) | chunked sliding-window attention; a memory sweep against full attention | Apple Silicon |
| `part8-qwen3-teacher/` | MiniGPT (Part 8), not yet published | rebuilt on Qwen3's 152k vocabulary; distilled from Qwen3-8B-Base via cached logits | Apple Silicon |

Every folder has a follow-along notebook, `minigpt_follow_along_N.ipynb`, built by the `build_workbook_N.py` next to it. Parts 1 to 3 run in Colab; Parts 4 to 8 use MLX, so their notebooks are for Jupyter on a Mac with Apple Silicon, and start from `notebook_setup.py`.

Parts 1 and 2 use the model code from Jibin Joseph's
[MiniGPT notebook](https://github.com/jibin10/MiniGPT). `minigpt_notebook.py` downloads
it at a pinned commit, so every script here uses exactly the version the posts were
written against. The trained exhibit model itself, `exhibit.pt`, is published at
https://haddley.github.io/minigpt-demo/exhibit.pt, next to its GGUF export.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Part 1: run the exhibit, and export it
cd part1-running
curl -O https://haddley.github.io/minigpt-demo/exhibit.pt
python build_workbook.py                    # writes minigpt_follow_along.ipynb
python export_gguf.py exhibit.pt exhibit.gguf
python export_web.py exhibit.pt .           # weights.bin + manifest.json for the live demo

# Part 2: grow the exhibit from random numbers (about 6 minutes on a Mac's CPU)
cd ../part2-growing
curl -O https://haddley.github.io/minigpt-demo/exhibit.pt
python grow_exhibit.py --out grown.pt --compare exhibit.pt   # identical on a Mac
python build_workbook_2.py                  # writes minigpt_follow_along_2.ipynb

# Part 3
cd ../part3-tokenisers
python prepare_data.py && python tokenizers_setup.py
python train.py --tokenizer char            # then gpt2, then bpe8k
python compare.py

# Part 4
cd ../part4-mlx
python train_mlx.py --tokenizer bpe8k
python bench.py --framework torch           # then mlx, mlx-nocompile

# Part 5
cd ../part5-modern-block
python train_llama.py --tag modern          # --mlp gelu / --gqa-off / --norm layer / --pos learned
python figures.py

# Part 6
cd ../part6-distillation
python train_distill.py --tag baseline --teacher none --alpha 1.0
python train_distill.py --tag gpt2 --teacher gpt2 --alpha 0.5
python train_teacher.py --dim 512 --layers 8 --iters 5000
python train_distill.py --tag big --teacher runs/teacher.safetensors --alpha 0.5

# Part 7
cd ../part7-sliding-window
python mem_sweep.py && python figures.py

# Part 8
cd ../part8-qwen3-teacher
./run_all.sh
```

Parts 4 to 8 import code and data from earlier folders, so run Part 3's
`prepare_data.py` and `tokenizers_setup.py` first.

## Key results (M1 Max, 64 GB)

| | result | note |
|---|---|---|
| Part 1: the exhibit after `goo` | `d` 96.6% | the same in PyTorch, the browser demo, and llama.cpp |
| Part 2: regrowing the exhibit | 826,433 of 826,433 numbers identical | deterministic CPU training, about 6 minutes |
| Part 3: character tokeniser | 1.04 bits/byte | lowest raw loss, worst bits/byte |
| Part 3: trained 8k BPE | 0.70 bits/byte | matches GPT-2's 50k vocabulary at under half the parameters |
| Part 4: MLX vs PyTorch on MPS | | `mx.compile` 16% faster, 15% less memory |
| Part 5: modern Llama block | 0.672 bits/byte | vs 0.689 for the GPT block; the gain is entirely RoPE |
| Part 6: distilled from a same-data teacher | 0.694 bits/byte | vs 0.756 baseline; GPT-2 as teacher does not help |
| Part 7: 256-token sliding window at 1,024 context | 0.673 bits/byte | the same as full attention, in less memory |
| Part 8: scaled student, Qwen3-8B-Base teacher | 0.731 bits/byte | vs 0.765 baseline; a frontier teacher gives a small gain |

`tools/render_term.py` renders captured output as terminal-style PNGs for the posts
(the runs are headless, so there is no window to screenshot).
