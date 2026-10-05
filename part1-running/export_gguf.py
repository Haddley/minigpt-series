# Export the MiniGPT exhibit model (exhibit.pt) to GGUF, so llama.cpp can run it.
# MiniGPT maps onto llama.cpp's "gpt2" architecture: learned position embeddings,
# pre-LayerNorm blocks, biases everywhere, a GELU MLP, and a separate output layer.
# Two adjustments are needed:
#   1. llama.cpp's gpt2 graph wants Q, K and V stacked into one attn_qkv tensor.
#   2. It has no output bias, so the bias is folded exactly into the final
#      LayerNorm's bias: W (beta + d) = W beta + b, with d = pinv(W) b
#      (exact because the 65 x 128 output matrix has full row rank).
import sys
import numpy as np
import torch
import gguf

src = sys.argv[1] if len(sys.argv) > 1 else "exhibit.pt"
dst = sys.argv[2] if len(sys.argv) > 2 else "exhibit.gguf"

sd = {k: v.double() for k, v in torch.load(src, map_location="cpu").items()}
chars = list("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
n_layer, n_head, n_embd, n_ctx = 4, 4, 128, 128
assert sd["token_embedding.weight"].shape == (len(chars), n_embd)

# fold the output bias into the final LayerNorm bias
W, b = sd["lm_head.weight"], sd["lm_head.bias"]
assert torch.linalg.matrix_rank(W) == W.shape[0]
final_ln_bias = sd["final_ln.bias"] + torch.linalg.pinv(W) @ b

def f32(t):
    return t.float().numpy().astype(np.float32)

w = gguf.GGUFWriter(dst, "gpt2")
w.add_name("MiniGPT exhibit")
w.add_context_length(n_ctx)
w.add_embedding_length(n_embd)
w.add_feed_forward_length(4 * n_embd)
w.add_block_count(n_layer)
w.add_head_count(n_head)
w.add_layer_norm_eps(1e-5)
w.add_file_type(gguf.LlamaFileType.ALL_F32)

# tokenizer: the 65 letters, written the way GPT-2's byte-level BPE stores them
# (space becomes "Ġ", new line becomes "Ċ"), and no merges that can ever apply, so every letter is one token
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for x in range(256):
        if x not in bs:
            bs.append(x); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

b2u = bytes_to_unicode()
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("default")
w.add_token_list([b2u[ord(c)] for c in chars])
w.add_token_types([gguf.TokenType.NORMAL] * len(chars))
# llama.cpp insists on at least one merge rule; this one joins the two byte symbols
# of "é", which never appears in Tiny Shakespeare, so it never fires
w.add_token_merges(["Ã ©"])
w.add_add_bos_token(False)
# llama.cpp needs an end-of-text token, and MiniGPT has none. Left unset, llama.cpp
# picks token 11, which here is ";", and stops at every semicolon. "$" (token 3)
# appears exactly once in all of Tiny Shakespeare, so it makes a safe stand-in.
w.add_bos_token_id(3)
w.add_eos_token_id(3)

w.add_tensor("token_embd.weight", f32(sd["token_embedding.weight"]))
w.add_tensor("position_embd.weight", f32(sd["position_embedding.weight"]))
for i in range(n_layer):
    p = f"blocks.{i}."
    w.add_tensor(f"blk.{i}.attn_norm.weight", f32(sd[p + "ln1.weight"]))
    w.add_tensor(f"blk.{i}.attn_norm.bias", f32(sd[p + "ln1.bias"]))
    qkv_w = torch.cat([sd[p + f"attn.{n}.weight"] for n in ("query", "key", "value")], 0)
    qkv_b = torch.cat([sd[p + f"attn.{n}.bias"] for n in ("query", "key", "value")], 0)
    w.add_tensor(f"blk.{i}.attn_qkv.weight", f32(qkv_w))
    w.add_tensor(f"blk.{i}.attn_qkv.bias", f32(qkv_b))
    w.add_tensor(f"blk.{i}.attn_output.weight", f32(sd[p + "attn.proj.weight"]))
    w.add_tensor(f"blk.{i}.attn_output.bias", f32(sd[p + "attn.proj.bias"]))
    w.add_tensor(f"blk.{i}.ffn_norm.weight", f32(sd[p + "ln2.weight"]))
    w.add_tensor(f"blk.{i}.ffn_norm.bias", f32(sd[p + "ln2.bias"]))
    w.add_tensor(f"blk.{i}.ffn_up.weight", f32(sd[p + "mlp.fc1.weight"]))
    w.add_tensor(f"blk.{i}.ffn_up.bias", f32(sd[p + "mlp.fc1.bias"]))
    w.add_tensor(f"blk.{i}.ffn_down.weight", f32(sd[p + "mlp.fc2.weight"]))
    w.add_tensor(f"blk.{i}.ffn_down.bias", f32(sd[p + "mlp.fc2.bias"]))
w.add_tensor("output_norm.weight", f32(sd["final_ln.weight"]))
w.add_tensor("output_norm.bias", f32(final_ln_bias))
w.add_tensor("output.weight", f32(W))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", dst)
