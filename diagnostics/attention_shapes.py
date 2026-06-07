import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GPTConfig, CausalSelfAttention

config = GPTConfig(n_embd=128, n_head=4, block_size=16, vocab_size=256, n_layer=2)
attn = CausalSelfAttention(config)
attn.eval()

B, T, C = 2, 16, 128
x = torch.randn(B, T, C)

with torch.no_grad():
    # expect: (2, 16, 384) — three projections fused into one matrix multiply
    qkv = attn.c_attn(x)
    print(f"Fused QKV projection shape: {qkv.shape}")

    q, k, v = qkv.split(config.n_embd, dim=2)
    # expect: (2, 16, 128) each — not yet split across heads
    print(f"Q shape (before head split): {q.shape}")
    print(f"K shape (before head split): {k.shape}")
    print(f"V shape (before head split): {v.shape}")

    nh = config.n_head
    hs = config.n_embd // config.n_head
    q = q.view(B, T, nh, hs).transpose(1, 2)
    k = k.view(B, T, nh, hs).transpose(1, 2)
    v = v.view(B, T, nh, hs).transpose(1, 2)
    # expect: (2, 4, 16, 32) — head_dim = 128/4 = 32
    print(f"Q shape (after head split):  {q.shape}  [B, n_head, T, head_dim]")
    print(f"K shape (after head split):  {k.shape}")
    print(f"V shape (after head split):  {v.shape}")
    # expect: head_dim = 32
    print(f"head_dim = n_embd / n_head = {config.n_embd} / {config.n_head} = {hs}")

    att = (q @ k.transpose(-2, -1)) * (1.0 / (hs ** 0.5))
    # expect: (2, 4, 16, 16) — each token attends to every other token
    print(f"Attention scores (before softmax): {att.shape}  [B, n_head, T, T]")

    att_soft = torch.softmax(att, dim=-1)
    # expect: same shape, each row sums to 1.0
    print(f"Attention scores (after softmax):  {att_soft.shape}")
    print(f"Row sum (should be 1.0): {att_soft[0, 0, 0].sum().item():.4f}")

    y = (att_soft @ v).transpose(1, 2).contiguous().view(B, T, C)
    # expect: (2, 16, 128) — same shape as input after heads are reassembled
    print(f"Output after head reassembly: {y.shape}  [B, T, n_embd]")
