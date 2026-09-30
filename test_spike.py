import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 1. Grab prompt from terminal arguments or prompt interactively
if len(sys.argv) > 1:
    user_prompt = " ".join(sys.argv[1:])
else:
    user_prompt = input("Enter your test prompt: ")

print(f"\n[+] Testing with prompt: \"{user_prompt}\"")

# 2. Load Tokenizer & Model
print("[+] Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": user_prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

def run_inference(label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            use_cache=False
        )
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(generated.strip())
    print("-" * 60)

# 3. Test 1: Clean Baseline
run_inference("1. CLEAN BASELINE (Pristine Weights)")

# 4. Save original weight value before mutating
INDEX = 512
SPIKE_VALUE = 500.0

with torch.no_grad():
    original_val = model.model.norm.weight.data[INDEX].item()
    print(f"\n[!] Original weight at index {INDEX}: {original_val:.4f}")
    print(f"[!] Spiking index {INDEX} to {SPIKE_VALUE}...")
    model.model.norm.weight.data[INDEX] = SPIKE_VALUE

# 5. Test 2: Spiked Model
run_inference(f"2. MUTATED MODEL (model.model.norm.weight[{INDEX}] = {SPIKE_VALUE})")

# 6. Clean up: Restore weight so the model in memory returns to normal
with torch.no_grad():
    model.model.norm.weight.data[INDEX] = original_val
    print(f"[+] Weight at index {INDEX} successfully restored to {original_val:.4f}.")
