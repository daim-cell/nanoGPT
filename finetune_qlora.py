"""
QLoRA fine-tuning of GPT-2 (124M) on tatsu-lab/alpaca.

Uses peft + bitsandbytes for 4-bit NF4 quantization on CUDA.
On Apple Silicon (MPS) bitsandbytes 4-bit is not supported; the script
falls back to float16 with LoRA adapters only (no quantization).

Usage:
    python finetune_qlora.py
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilgpt2"
MAX_LENGTH   = 512
N_EXAMPLES   = 500
OUTPUT_DIR   = "results/qlora_adapter"
COMPARE_FILE = "results/comparison.txt"

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["c_attn", "c_proj"]

DEVICE   = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
USE_4BIT = DEVICE == "cuda"   # bitsandbytes 4-bit requires CUDA

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about autumn.",
    "What is the capital of France?",
    "Give me a recipe for banana bread.",
    "How does photosynthesis work?",
]

# ── Dataset helpers ───────────────────────────────────────────────────────────
def format_example(ex):
    parts = [ex["instruction"]]
    if ex.get("input"):
        parts.append(ex["input"])
    parts.append(ex["output"])
    return {"text": "\n\n".join(parts)}


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


# ── Inference helper ──────────────────────────────────────────────────────────
def generate(model, tokenizer, prompt, max_new_tokens=150):
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    print(f"Device: {DEVICE} | 4-bit quantization: {USE_4BIT}")


    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Base model — 4-bit NF4 on CUDA, float16 fallback on MPS/CPU
    if USE_4BIT:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_cfg, device_map="auto"
        )
    else:
        print("[info] bitsandbytes 4-bit not available on this device — using float16 + LoRA only")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).to(DEVICE)

    # 3. Prepare for k-bit training (upcasts layer norms to fp32 to avoid underflow)
    if USE_4BIT:
        base_model = prepare_model_for_kbit_training(base_model)

    # 4. LoRA adapters via peft
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # 5. Dataset — alpaca, format as "instruction\n\ninput\n\noutput", tokenize
    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{N_EXAMPLES}]")
    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds = ds.map(lambda ex: tokenize(ex, tokenizer), batched=True, remove_columns=["text"])
    ds = ds.train_test_split(test_size=0.05, seed=42)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # # 6. Training
    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
        fp16=(DEVICE == "cuda")
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 7. Inference comparison: base model vs fine-tuned, 5 fixed prompts
    model.eval()

    base_cmp = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(DEVICE)
    base_cmp.eval()

    lines = []
    for prompt in PROMPTS:
        base_out = generate(base_cmp, tokenizer, prompt)
        ft_out   = generate(model, tokenizer, prompt)
        lines += [
            f"PROMPT: {prompt}",
            f"\nBASE:\n{base_out}",
            f"\nFINETUNED:\n{ft_out}",
            "\n" + "─" * 80 + "\n",
        ]

    with open(COMPARE_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"Comparison saved to {COMPARE_FILE}")


if __name__ == "__main__":
    main()
