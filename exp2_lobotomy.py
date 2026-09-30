import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def get_lobotomy_hook():
    def hook(module, inputs, output):
        # inputs[0] is the input hidden_states to the layer
        if isinstance(output, tuple):
            return (inputs[0],) + output[1:]
        return inputs[0]
    return hook

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
    
    prompt = "Write a Python function to check if a binary tree is symmetric. Include type hints and a docstring."
    messages = [
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    def generate_and_print(condition, ablate_layers=[]):
        print(f"\n{'='*60}\n{condition}\n{'='*60}")
        
        handles = []
        for l in ablate_layers:
            handles.append(model.model.layers[l].register_forward_hook(get_lobotomy_hook()))
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=False,
                use_cache=False, 
                pad_token_id=tokenizer.eos_token_id
            )
            
        for h in handles:
            h.remove()
            
        input_length = inputs.input_ids.shape[1]
        print(tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True))
        print("-" * 60)

    # 1. Control Condition
    generate_and_print("1. CONTROL (Full 28-Layer Baseline)", ablate_layers=[])

    # 2. Early-Brain Ablation (Bypass layers 2 & 3)
    generate_and_print("2. EARLY-BRAIN ABLATION (Bypassed Layers 2 & 3)", ablate_layers=[2, 3])

    # 3. Mid-Brain Ablation (Bypass layers 13 & 14)
    generate_and_print("3. MID-BRAIN ABLATION (Bypassed Layers 13 & 14)", ablate_layers=[13, 14])

if __name__ == "__main__":
    main()
