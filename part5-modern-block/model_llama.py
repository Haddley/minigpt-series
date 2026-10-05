"""The Part 4 MiniGPT with the modern Transformer block from the Llama papers.

Four changes from the vanilla GPT block, each toggleable so the ablation script
can turn them on one at a time:

  norm = "rms"     RMSNorm instead of LayerNorm (no mean-centring, no bias)
  pos  = "rope"    rotary position embeddings instead of a learned table
  mlp  = "swiglu"  SwiGLU feed-forward (gate * up -> down) instead of GELU
  n_kv_heads < n_heads   grouped-query attention: fewer K/V heads than Q heads

The layout is checked against mlx-lm's models/llama.py. Defaults reproduce the
shape Meta used for Llama 3.2 1B/3B, scaled down to ~13M parameters.
"""
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Config:
    vocab_size: int
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    hidden: int = 1024          # SwiGLU inner size; 384*1024*3 == 384*1536*2 (GELU 4x)
    block_size: int = 256
    rope_base: float = 10000.0
    dropout: float = 0.2
    norm: str = "rms"           # "rms" | "layer"
    pos: str = "rope"           # "rope" | "learned"
    mlp: str = "swiglu"         # "swiglu" | "gelu"
    window: int = 0             # 0 = full causal attention; >0 = sliding-window width (Part 7)
    naive_window: bool = False  # >0 window via a [T, T] mask + fused kernel instead of chunked_swa


def make_norm(cfg):
    return nn.RMSNorm(cfg.dim) if cfg.norm == "rms" else nn.LayerNorm(cfg.dim)


def sliding_window_mask(T, w, dtype):
    """Additive [T, T] mask: a query may attend to the w most recent keys, itself included.

    Passing this to a fused attention kernel gives the sliding-window *behaviour* but
    not its memory saving - the kernel still builds the full T x T score matrix and
    just zeroes most of it. See chunked_swa for the version that is actually O(T * w).
    """
    i = mx.arange(T)[:, None]
    j = mx.arange(T)[None, :]
    keep = (j <= i) & (i - j < w)
    return mx.where(keep, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


def chunked_swa(q, k, v, w, scale):
    """Sliding-window attention that never materialises a T x T matrix.

    The sequence is cut into chunks of w tokens. Chunk i attends to the keys in
    chunks i-1 and i - a 2w-wide band that covers every query's w-token causal
    window - so the largest score tensor is [B, H, T/w, w, 2w], linear in T.
    """
    B, H, T, D = q.shape
    pad = (w - T % w) % w
    if pad:
        z = [(0, 0), (0, 0), (0, pad), (0, 0)]
        q, k, v = mx.pad(q, z), mx.pad(k, z), mx.pad(v, z)
    Tp, nc = T + pad, (T + pad) // w
    kpad = mx.pad(k, [(0, 0), (0, 0), (w, 0), (0, 0)])
    vpad = mx.pad(v, [(0, 0), (0, 0), (w, 0), (0, 0)])

    qc = q.reshape(B, H, nc, w, D)
    kc = mx.stack([kpad[:, :, i * w:i * w + 2 * w] for i in range(nc)], axis=2)
    vc = mx.stack([vpad[:, :, i * w:i * w + 2 * w] for i in range(nc)], axis=2)

    scores = (qc @ kc.transpose(0, 1, 2, 4, 3)) * scale          # [B,H,nc,w,2w]
    qpos = mx.arange(w)[:, None]
    kk = mx.arange(2 * w)[None, :]
    rel = w + qpos - kk                                          # global_q - global_key
    band = (rel >= 0) & (rel < w)                                # causal + window
    ci = mx.arange(nc)[:, None, None]
    in_range = (ci * w + kk[None] - w) >= 0                       # key not before position 0
    mask = band[None] & in_range                                 # [nc, w, 2w]
    scores = mx.where(mask[None, None], scores, -mx.inf)

    out = mx.softmax(scores, axis=-1) @ vc                       # [B,H,nc,w,D]
    return out.reshape(B, H, Tp, D)[:, :, :T, :]


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)
        self.window = cfg.window
        self.naive_window = cfg.naive_window
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_base) \
            if cfg.pos == "rope" else None

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        if self.window <= 0:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        elif self.naive_window:
            m = sliding_window_mask(T, self.window, x.dtype)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=m)
        else:
            reps = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, reps, axis=1)
            v = mx.repeat(v, reps, axis=1)
            out = chunked_swa(q, k, v, self.window, self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.dim, cfg.hidden, bias=False)   # up
        self.w2 = nn.Linear(cfg.hidden, cfg.dim, bias=False)   # down

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class GELUMLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Linear(cfg.dim, 4 * cfg.dim)
        self.proj = nn.Linear(4 * cfg.dim, cfg.dim)

    def __call__(self, x):
        return self.proj(nn.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = make_norm(cfg)
        self.attn = Attention(cfg)
        self.ffn_norm = make_norm(cfg)
        self.ffn = SwiGLU(cfg) if cfg.mlp == "swiglu" else GELUMLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def __call__(self, x):
        x = x + self.drop(self.attn(self.attn_norm(x)))
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x


class MiniLlama(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = make_norm(cfg)

    def __call__(self, idx):
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(mx.arange(idx.shape[1]))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x) @ self.tok_emb.weight.T   # tied head

    def loss(self, idx, targets):
        logits = self(idx)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean"
        )
