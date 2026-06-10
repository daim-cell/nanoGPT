import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GPT

print("Loading gpt2 pretrained weights from HuggingFace...")
model = GPT.from_pretrained('gpt2')
model.eval()

wte    = model.transformer.wte.weight
lm_head = model.lm_head.weight

# expect: (50257, 768) — vocab_size × n_embd
print(f"\nwte.weight shape:     {wte.shape}  [vocab_size, n_embd]")
# expect: (50257, 768) — same shape, same tensor
print(f"lm_head.weight shape: {lm_head.shape}  [vocab_size, n_embd]")

# data_ptr() returns the raw memory address — same address means same tensor object
# expect: True — wte and lm_head.weight are literally the same Parameter in memory
same_ptr = wte.data_ptr() == lm_head.data_ptr()
print(f"\nSame memory address (weight tying confirmed): {same_ptr}")

# Mutating one should be visible in the other
with torch.no_grad():
    original = wte[0, 0].item()
    wte[0, 0] = 999.0
    # expect: True — both see the mutation because they point to the same storage
    print(f"Mutation visible through lm_head: {lm_head[0, 0].item() == 999.0}")
    wte[0, 0] = original  # restore
# This reduces parameters which are usually a lot (vocab_size, n_embd) and saves memory, but also means that the input embedding and output projection are forced to be the same. 
print(f"\nvocab_size = {wte.shape[0]}  (GPT-2 uses 50257 tokens)")
print(f"n_embd     = {wte.shape[1]}  (each token maps to a 768-dim vector)")
print(f"Parameters saved by tying: {wte.numel():,}  (one copy instead of two)")
