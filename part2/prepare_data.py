"""Download a slice of TinyStories and make a 90/10 train/val split.

TinyStories (Eldan & Li, 2023) is a corpus of short synthetic children's stories
written with a small vocabulary. It is large enough that a subword tokenizer has
something to compress, and simple enough that a ~15-30M parameter model can learn
to produce coherent sentences.

The V2 GPT-4 validation file is about 22 MB - a good size for an afternoon on an
M1 Max. Swap in TinyStoriesV2-GPT4-train.txt (~2 GB) for a longer run.
"""
import os
import urllib.request

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
DATA = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(DATA, exist_ok=True)
    raw = os.path.join(DATA, "tinystories.txt")
    if not os.path.exists(raw):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw)

    text = open(raw, encoding="utf-8").read()
    print(f"{len(text):,} characters, {len(text.encode('utf-8')):,} bytes")

    n = int(0.9 * len(text))
    open(os.path.join(DATA, "train.txt"), "w", encoding="utf-8").write(text[:n])
    open(os.path.join(DATA, "val.txt"), "w", encoding="utf-8").write(text[n:])
    print("wrote train.txt and val.txt")


if __name__ == "__main__":
    main()
