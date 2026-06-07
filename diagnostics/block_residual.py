import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GPTConfig, Block

config = GPTConfig(n_embd=128, n_head=4, block_size=16, vocab_size=256, n_layer=2)
block = Block(config)
block.eval()

B, T, C = 2, 16, 128
x = torch.randn(B, T, C)

# expect: (2, 16, 128) — starting shape
print(f"Input shape:                    {x.shape}  [B, T, n_embd]")

with torch.no_grad():
    # LayerNorm normalizes before attention (pre-norm); shape unchanged
    # expect: (2, 16, 128)
    attn_out = block.attn(block.ln_1(x))
    print(f"After attention (pre-residual): {attn_out.shape}")

    # Residual connection adds back the original x; shape unchanged
    # expect: (2, 16, 128)
    x_mid = x + attn_out
    print(f"After attention residual add:   {x_mid.shape}")

    # MLP expands to 4×n_embd internally then contracts back; output shape unchanged
    # expect: (2, 16, 128)
    mlp_out = block.mlp(block.ln_2(x_mid))
    print(f"After MLP (pre-residual):       {mlp_out.shape}")

    # Second residual connection; shape unchanged
    # expect: (2, 16, 128)
    x_out = x_mid + mlp_out
    print(f"After MLP residual add (final): {x_out.shape}")

    # Verify via full block forward — must match
    # expect: (2, 16, 128)
    x_block = block(torch.randn(B, T, C))
    print(f"Full block forward output:      {x_block.shape}")

    all_same = all(s.shape == (B, T, C) for s in [attn_out, x_mid, mlp_out, x_out, x_block])
    # expect: True — shape is invariant through the entire block
    print(f"\nAll shapes == (B={B}, T={T}, C={C}): {all_same}")
