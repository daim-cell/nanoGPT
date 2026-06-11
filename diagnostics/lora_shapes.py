"""Run after finetune_qlora.py has been executed at least once."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

ADAPTER_DIR = "results/qlora_adapter"
MODEL_NAME  = "distilgpt2"

# Load either the saved adapter or create a fresh peft model for shape inspection
if os.path.isdir(ADAPTER_DIR):
    print(f"Loading saved adapter from {ADAPTER_DIR}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
else:
    print(f"Adapter not found — creating a fresh peft model for shape inspection")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["c_attn", "c_proj"],
                          bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_cfg)

# Find and inspect LoRA-adapted layers
# peft wraps each targeted Linear as a LoRA module with lora_A and lora_B sub-linears
shown = 0
for name, module in model.named_modules():
    if not hasattr(module, 'lora_A'):
        continue
    for adapter_key in module.lora_A:
        A = module.lora_A[adapter_key].weight  # Linear(in_features, r) → weight is (r, in_features)
        B = module.lora_B[adapter_key].weight  # Linear(r, out_features) → weight is (out_features, r)
        W = module.base_layer.weight           # original frozen weight (out_features, in_features)

        # expect: A is (r, in_features), B is (out_features, r)
        print(f"Layer: {name}")
        print(f"  Frozen weight W:  {W.shape}  [out_features, in_features]")
        print(f"  LoRA A:           {A.shape}  [r, in_features]")
        print(f"  LoRA B:           {B.shape}  [out_features, r]")

        r = A.shape[0]
        # expect: r × (in + out) << out × in — very few trainable params
        adapter_params = A.numel() + B.numel()
        full_params    = W.numel()
        print(f"  Adapter params:   {adapter_params:,}  ({adapter_params / full_params * 100:.1f}% of W)")
        print()
        shown += 1
        if shown >= 2:
            break
    if shown >= 2:
        break
