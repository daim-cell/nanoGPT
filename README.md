# nanoGPT

This repository is based on Andrej Karpathy's nanoGPT: a small, readable GPT
training and fine-tuning codebase. The upstream project is now deprecated in
favor of [nanochat](https://github.com/karpathy/nanochat), but this repo is
still useful for studying the internals of GPT training.

The current local version adds educational diagnostics around transformer
shapes, residual connections, embedding weight tying, next-token targets, and
LoRA/QLoRA adapter structure. It also includes an experimental
`finetune_qlora.py` script for trying parameter-efficient fine-tuning with
Hugging Face `transformers` and `peft`.

## Recent Local Changes

Recent commits added and documented the following:

- More explanatory comments in `train.py`, especially around training setup,
  batching, evaluation, optimizer setup, compilation, and gradient updates.
- `diagnostics/attention_shapes.py` to inspect the fused QKV projection,
  per-head attention tensors, attention scores, and final reassembled output.
- `diagnostics/block_residual.py` to show that attention, MLP, and residual
  additions preserve `[batch, time, channels]` shape through a GPT block.
- `diagnostics/embedding_weights.py` to verify GPT-2 input embeddings and the
  language-model head share the same weight tensor.
- `diagnostics/loss_offset.py` to demonstrate why `y` is `x` shifted left by
  one token in next-token prediction.
- `diagnostics/lora_shapes.py` to inspect LoRA adapter matrices and compare
  adapter parameter counts with the frozen base weights.
- `finetune_qlora.py` to test a DistilGPT-2 LoRA/QLoRA-style fine-tuning flow
  on the `tatsu-lab/alpaca` instruction dataset.

## Install

Base nanoGPT dependencies:

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

For the LoRA/QLoRA experiment:

```sh
pip install peft bitsandbytes accelerate safetensors
```

`bitsandbytes` 4-bit quantization requires CUDA. On CPU or Apple Silicon MPS,
`finetune_qlora.py` falls back to LoRA without true 4-bit QLoRA quantization.

## Quick Start

Prepare the tiny Shakespeare character dataset:

```sh
python data/shakespeare_char/prepare.py
```

Train a small character-level GPT:

```sh
python train.py config/train_shakespeare_char.py
```

Sample from the trained checkpoint:

```sh
python sample.py --out_dir=out-shakespeare-char
```

For CPU-only runs, use a smaller configuration and disable compilation:

```sh
python train.py config/train_shakespeare_char.py \
  --device=cpu \
  --compile=False \
  --eval_iters=20 \
  --log_interval=1 \
  --block_size=64 \
  --batch_size=12 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --lr_decay_iters=2000 \
  --dropout=0.0
```

On Apple Silicon, try `--device=mps` if your PyTorch installation supports it.

## Fine-Tuning With nanoGPT

Prepare the GPT-2-tokenized Shakespeare dataset:

```sh
python data/shakespeare/prepare.py
```

Run the standard nanoGPT fine-tuning config:

```sh
python train.py config/finetune_shakespeare.py
```

Sample from the fine-tuned checkpoint:

```sh
python sample.py --out_dir=out-shakespeare
```

## QLoRA / LoRA Experiment

The added `finetune_qlora.py` script explores instruction fine-tuning
`distilgpt2` on a small slice of `tatsu-lab/alpaca`.

Run it with:

```sh
python finetune_qlora.py
```

What the script does:

- Loads `distilgpt2` with Hugging Face `transformers`.
- Uses 4-bit NF4 quantization through `bitsandbytes` when CUDA is available.
- Falls back to float16 LoRA-only training on non-CUDA devices.
- Applies LoRA adapters to GPT-2 style `c_attn` and `c_proj` modules.
- Trains on 500 Alpaca examples with PEFT adapters.
- Saves adapter artifacts under `results/qlora_adapter/`.
- Writes base-vs-fine-tuned generations to `results/comparison.txt` when the
  full run completes.

Important note from the local test:

I could not run true QLoRA locally because I do not have a CUDA GPU available.
The available fallback path used LoRA without 4-bit quantization, and the test
outputs were very poor. Even so, the run was useful because it showed the basic
QLoRA/LoRA workflow: load a base model, prepare quantization when possible,
attach low-rank adapters, train only the adapter weights, save the adapter, and
compare base-model output against fine-tuned output.

The poor result is expected for this rough experiment. Better output will need
proper hyperparameter tuning and model selection, for example:

- Use a stronger base model than `distilgpt2`.
- Train on more examples or a higher-quality instruction dataset.
- Tune LoRA rank, alpha, dropout, learning rate, warmup, batch size, and
  gradient accumulation.
- Run on a CUDA GPU so actual 4-bit QLoRA can be used.
- Evaluate generations across more prompts before trusting the adapter.

## Diagnostics

These scripts are meant for understanding the model internals. Run them from
the repository root.

```sh
python diagnostics/attention_shapes.py
python diagnostics/block_residual.py
python diagnostics/embedding_weights.py
python diagnostics/loss_offset.py
python diagnostics/lora_shapes.py
```

Diagnostic purpose:

- `attention_shapes.py`: prints Q, K, V, attention-score, and attention-output
  shapes.
- `block_residual.py`: shows the shape flow through attention, residual
  addition, MLP, and the full transformer block.
- `embedding_weights.py`: confirms GPT-2 weight tying between `wte.weight` and
  `lm_head.weight`.
- `loss_offset.py`: illustrates next-token prediction targets with a simple
  synthetic token sequence.
- `lora_shapes.py`: loads the saved adapter if available, otherwise creates a
  fresh LoRA model, then prints LoRA A/B matrix shapes and adapter parameter
  counts.


## Acknowledgements

This repo builds on the original
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) project.
