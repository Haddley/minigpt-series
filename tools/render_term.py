"""Render captured text output as a terminal-style PNG.

Used to turn the real stdout of the training and setup scripts into figures for
the blog posts, since these runs happen headless (no window to screenshot).

    ../.venv/bin/python ../tools/render_term.py data/log_char.txt out.png --title "train.py --tokenizer char"
"""
import argparse
import re
import sys

from PIL import Image, ImageDraw, ImageFont

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

FONTS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/Library/Fonts/Menlo.ttc",
]
BG = (30, 30, 30)
FG = (220, 220, 220)
DIM = (150, 150, 150)
BAR = (45, 45, 45)
DOTS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]


def load_font(size):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("infile")
    p.add_argument("outfile")
    p.add_argument("--title", default="")
    p.add_argument("--size", type=int, default=26)
    p.add_argument("--width", type=int, default=0, help="wrap/crop columns; 0 = auto")
    p.add_argument("--scale", type=int, default=2)
    args = p.parse_args()

    raw = sys.stdin.read() if args.infile == "-" else open(args.infile).read()
    raw = ANSI.sub("", raw.replace("\r", "\n"))
    lines, blanks = [], 0
    for ln in raw.split("\n"):
        ln = ln.rstrip()
        if not ln:
            blanks += 1
            if blanks > 1:
                continue
        else:
            blanks = 0
        lines.append(ln)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    s = args.scale
    font = load_font(args.size * s)
    pad = 24 * s
    bar_h = 40 * s
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + 6 * s

    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max((tmp.textlength(ln, font=font) for ln in lines), default=200 * s)
    W = int(text_w) + 2 * pad
    H = bar_h + pad + line_h * len(lines) + pad

    dot_w = pad + len(DOTS) * 22 * s
    title_font = load_font(int(args.size * 0.8) * s)
    if args.title:
        W = max(W, int(dot_w + tmp.textlength(args.title, font=title_font)) + pad)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, bar_h], fill=BAR)
    for i, col in enumerate(DOTS):
        cx = pad + i * 22 * s
        d.ellipse([cx, bar_h // 2 - 7 * s, cx + 14 * s, bar_h // 2 + 7 * s], fill=col)
    if args.title:
        d.text((dot_w, bar_h / 2 - args.size * 0.45 * s), args.title,
               font=title_font, fill=DIM)

    y = bar_h + pad
    for ln in lines:
        d.text((pad, y), ln, font=font, fill=DIM if ln.startswith("===") else FG)
        y += line_h

    if s != 1:
        img = img.resize((W // s, H // s), Image.LANCZOS)
    img.save(args.outfile)
    print(f"wrote {args.outfile}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
