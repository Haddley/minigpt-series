# minigpt-series

Companion code for the MiniGPT blog series — building a small language model from
a character-level GPT up to a modern MLX model, one change at a time, on a 2022
Mac Studio (M1 Max, 64 GB).

| Part | What changes | Framework |
|---|---|---|
| 1 | Jibin Joseph's [MiniGPT notebook](https://github.com/jibin10/MiniGPT) — character-level GPT from first principles | PyTorch + MPS |
| `part2/` | character vs GPT-2 vs a trained 8k BPE tokenizer, on TinyStories, scored in bits per byte | PyTorch + MPS |
| `part3/` | the same model rebuilt in Apple's MLX; benchmarked against PyTorch-MPS | MLX |
| `part4/` | the Llama 3.2 block — RMSNorm, RoPE, SwiGLU, GQA — each ablated | MLX |
| `part5/` | logit distillation from GPT-2 small and from a same-data 51M teacher | MLX |
| `part6/` | chunked sliding-window attention; a memory sweep vs full attention | MLX |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Parts 3–6 need Apple Silicon (MLX). Parts 2–3 also run on CUDA or CPU.

## Running

```bash
# Part 2
cd part2
python prepare_data.py && python tokenizers_setup.py
python train.py --tokenizer char    # then gpt2, then bpe8k
python compare.py

# Part 3
cd ../part3
python train_mlx.py --tokenizer bpe8k
python bench.py --framework torch    # then mlx, mlx-nocompile

# Part 4
cd ../part4
python train_llama.py --tag modern   # --mlp gelu / --gqa-off / --norm layer / --pos learned
python figures.py

# Part 5
cd ../part5
python train_distill.py --tag baseline --teacher none --alpha 1.0
python train_distill.py --tag gpt2 --teacher gpt2 --alpha 0.5
python train_teacher.py --dim 512 --layers 8 --iters 5000
python train_distill.py --tag big --teacher runs/teacher.safetensors --alpha 0.5

# Part 6
cd ../part6
python mem_sweep.py && python figures.py
```

## Key results (M1 Max, 64 GB)

| | val bits/byte | note |
|---|---|---|
| Part 2 — character tokenizer | 1.04 | lowest raw loss, worst bits/byte |
| Part 2 — trained 8k BPE | 0.70 | matches GPT-2's 50k vocab at <½ the params |
| Part 3 — MLX vs PyTorch-MPS | — | `mx.compile` 16% faster, 15% less memory |
| Part 4 — modern Llama block | 0.672 | vs 0.689 GPT block; the gain is entirely RoPE |
| Part 5 — distilled from a same-data teacher | 0.694 | vs 0.756 baseline; GPT-2 as teacher: no help |
| Part 6 — 256-token sliding window @ 1024 ctx | 0.673 | = full attention, at lower memory |

`tools/render_term.py` renders captured stdout as terminal-style PNGs for the posts
(the runs are headless, so there is no window to screenshot).
