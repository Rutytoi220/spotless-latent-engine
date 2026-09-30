import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    prompt = "Write a Python function 'is_prime(n: int) -> bool' to check if a number is prime. Include docstrings."
    messages = [
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    # Store pristine clone of the final norm weights
    original_norm_weight = model.model.norm.weight.data.clone()

    def generate_and_print(condition):
        print(f"\n{'='*60}\n{condition}\n{'='*60}")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        input_length = inputs.input_ids.shape[1]
        print(tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True))
        print("-" * 60)

    # 1. Condition 1: Baseline Control
    generate_and_print("1. BASELINE CONTROL (Pristine model weights)")

    # 2. Condition 2: Sign Flip / Phase Inversion
    model.model.norm.weight.data.copy_(original_norm_weight)
    model.model.norm.weight.data[:1792].mul_(-1.0)
    generate_and_print("2. SIGN FLIP / PHASE INVERSION (First half *-1.0)")

    # 3. Condition 3: Single-Bit Cosmic Spike
    model.model.norm.weight.data.copy_(original_norm_weight)
    model.model.norm.weight.data[512] = 500.0
    generate_and_print("3. SINGLE-BIT COSMIC SPIKE (Index 512 = 500.0)")

if __name__ == "__main__":
    main()
