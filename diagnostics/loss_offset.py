import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

# Replicate get_batch from train.py on a small synthetic token array
block_size = 8
batch_size  = 2
np.random.seed(42)

# Token IDs 0..29 — simple sequence to make the shift obvious
data = np.arange(30, dtype=np.uint16)
print(f"Full token sequence: {data.tolist()}")

# Sample random windows exactly as train.py does
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy(data[i    : i + block_size    ].astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy(data[i + 1: i + block_size + 1].astype(np.int64)) for i in ix])

print(f"\nSampled start indices: {ix.tolist()}")

# expect: y[b] == x[b] shifted left by one — the next-token target for each position
print(f"\n{'':8} x[0]: {x[0].tolist()}")
print(f"{'':8} y[0]: {y[0].tolist()}")
print(f"\n{'':8} x[1]: {x[1].tolist()}")
print(f"{'':8} y[1]: {y[1].tolist()}")

# expect: True — every interior position of y equals the next position of x
shifted_match = (x[:, 1:] == y[:, :-1]).all()
print(f"\nx[:, 1:] == y[:, :-1]: {shifted_match.item()}")

print(f"\nAt each position t: x[t] is the input token, y[t] is the token to predict.")
print(f"y is x shifted left by 1, so the model always predicts the next token.")
