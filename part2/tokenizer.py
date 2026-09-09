"""A single interface over the three Part 2 tokenizers: encode, decode, vocab_size."""
import os
import pickle

DATA = os.path.join(os.path.dirname(__file__), "data")


class CharTokenizer:
    name = "char"

    def __init__(self):
        d = pickle.load(open(os.path.join(DATA, "char.pkl"), "rb"))
        self.stoi, self.itos = d["stoi"], d["itos"]
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


class GPT2Tokenizer:
    name = "gpt2"

    def __init__(self):
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

    def encode(self, s):
        # TinyStories separates stories with a literal <|endoftext|>; let it map
        # to GPT-2's real end-of-text token (50256) rather than raising.
        return self.enc.encode(s, allowed_special={"<|endoftext|>"})

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])


class BPE8kTokenizer:
    name = "bpe8k"

    def __init__(self):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(os.path.join(DATA, "bpe8k.json"))
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, s):
        return self.tok.encode(s).ids

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids])


def load_tokenizer(name):
    return {
        "char": CharTokenizer,
        "gpt2": GPT2Tokenizer,
        "bpe8k": BPE8kTokenizer,
    }[name]()
