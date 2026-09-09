# minigpt-series

Companion code for the MiniGPT blog series.

- **Part 1** — Jibin Joseph's [MiniGPT notebook](https://github.com/jibin10/MiniGPT), run locally on Apple Silicon. Character-level, PyTorch + MPS.
- **`part2/`** — the same GPT, three tokenizers (character, GPT-2 byte-level BPE, a custom 8k BPE), trained on a slice of TinyStories. PyTorch + MPS. Compared on bits-per-byte.
- **`part3/`** — the Part 2 model ported to Apple's MLX. Same tokenizer, same data. Head-to-head speed and memory against the PyTorch + MPS run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Part 2

```bash
cd part2
python prepare_data.py            # download a slice of TinyStories, split 90/10
python tokenizers_setup.py        # build the char / gpt2 / bpe8k tokenizers
python train.py --tokenizer char
python train.py --tokenizer gpt2
python train.py --tokenizer bpe8k
python compare.py                 # bits-per-byte curves + summary table
python generate.py --tokenizer bpe8k --prompt "Once upon a time"
```

## Part 3

```bash
cd part3
python train_mlx.py --tokenizer bpe8k
python bench.py --framework torch          # PyTorch-MPS: tokens/sec, peak memory
python bench.py --framework mlx            # MLX with mx.compile
python bench.py --framework mlx-nocompile  # MLX without mx.compile
python figures.py                         # loss-curve and benchmark plots
python generate_mlx.py --prompt "Once upon a time"
```

Results on a 2022 Mac Studio (M1 Max, 64 GB), 3,000 iterations, 8k BPE tokenizer:

| | val bits/byte | training step | peak memory |
|---|---|---|---|
| PyTorch + MPS | 0.697 | 48,400 tok/s | 4.60 GB |
| MLX (`mx.compile`) | 0.689 | 56,100 tok/s | 3.89 GB |
