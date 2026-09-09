"""Build the three tokenizers compared in Part 2.

  char   - every unique character is a token (the Part 1 approach)
  gpt2   - OpenAI's byte-level BPE, 50,257 tokens, borrowed unchanged via tiktoken
  bpe8k  - a byte-level BPE trained here on the TinyStories training split, 8,192 tokens

The GPT-2 tokenizer needs no training. The 8k tokenizer is trained with the
Hugging Face `tokenizers` library, which uses the same byte-level BPE algorithm
Karpathy walks through in minbpe (https://github.com/karpathy/minbpe) but runs
in seconds rather than minutes.
"""
import os
import pickle

DATA = os.path.join(os.path.dirname(__file__), "data")


def build_char():
    text = open(os.path.join(DATA, "train.txt"), encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    with open(os.path.join(DATA, "char.pkl"), "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos}, f)
    print(f"char : {len(chars)} tokens")


def build_bpe8k():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=["<unk>", "<|endoftext|>"],
        show_progress=False,
    )
    tok.train([os.path.join(DATA, "train.txt")], trainer)
    tok.save(os.path.join(DATA, "bpe8k.json"))
    print(f"bpe8k: {tok.get_vocab_size()} tokens")

    # Show the first learned merges - the pairs BPE thought were most worth joining.
    import json

    saved = json.load(open(os.path.join(DATA, "bpe8k.json")))
    for merge in saved["model"]["merges"][:15]:
        a, b = merge if isinstance(merge, list) else merge.split(" ")
        print(f"  merge  {a!r} + {b!r}")


def check_gpt2():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    print(f"gpt2 : {enc.n_vocab} tokens")


if __name__ == "__main__":
    build_char()
    check_gpt2()
    build_bpe8k()
