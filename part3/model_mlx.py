"""The Part 2 MiniGPT, rebuilt in Apple's MLX.

The architecture is identical - learned embeddings, pre-LayerNorm blocks, GELU
MLP, weight tying. What changes is the framework:

  * no .to(device) - MLX arrays live in unified memory the CPU and GPU share
  * mx.fast.scaled_dot_product_attention replaces the hand-written matmul/mask/softmax
  * the output head is a plain matmul against the token embedding, which ties the
    weights without any parameter-sharing bookkeeping
"""
import mlx.core as mx
import mlx.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_embd // n_head) ** -0.5
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x):
        B, T, C = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x):
        return self.drop(self.proj(nn.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)

    def __call__(self, idx):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(mx.arange(T))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x @ self.tok_emb.weight.T  # tied head

    def loss(self, idx, targets):
        logits = self(idx)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean"
        )
