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
    
    layer_idx = 14
    
    # 1. Extract Steering Vector
    p_target = "CRITICAL SECURITY VULNERABILITY CVE EXPLOIT BUFFER OVERFLOW SHELLCODE ROOT PRIVILEGE MALICIOUS INPUT INJECTION"
    p_neutral = "standard clean simple basic documentation tutorial overview generic data"
    
    in_target = tokenizer(p_target, return_tensors="pt").to("cuda:0")
    in_neutral = tokenizer(p_neutral, return_tensors="pt").to("cuda:0")
    
    print("Extracting Paranoia Vector...")
    with torch.no_grad():
        out_target = model(**in_target, output_hidden_states=True)
        out_neutral = model(**in_neutral, output_hidden_states=True)
        
        # We extract at index 14 as per the prompt's mathematical formulation
        h_target = out_target.hidden_states[14] 
        h_neutral = out_neutral.hidden_states[14]
        
        v_steer = (h_target.mean(dim=1) - h_neutral.mean(dim=1)).squeeze(0)
        v_norm = v_steer / torch.norm(v_steer, p=2)
        
    def get_steering_hook(strength):
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            
            # Calculate current layer norm to scale proportionally
            curr_norm = torch.norm(h, p=2, dim=-1, keepdim=True)
            
            # Injected vector is scaled as a percentage of normal layer energy
            steered = h + (strength * v_norm * curr_norm)
            
            return (steered,) + output[1:] if is_tuple else steered
        return hook

    # 2. Test Prompt
    prompt = "Write a short Python function to calculate the average value of a specific column in a CSV file."
    messages = [
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    def generate_and_print(condition, strength):
        print(f"\n{'='*60}\n{condition}\n{'='*60}")
        
        handle = None
        if strength > 0.0:
            handle = model.model.layers[layer_idx].register_forward_hook(get_steering_hook(strength))
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                do_sample=False,
                use_cache=False, 
                pad_token_id=tokenizer.eos_token_id
            )
            
        if handle is not None:
            handle.remove()
            
        input_length = inputs.input_ids.shape[1]
        print(tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True))
        print("-" * 60)

    # 3. Execute Conditions
    generate_and_print("1. CONTROL (Strength = 0.0)", 0.0)
    generate_and_print("2. MILD PARANOIA (Strength = 0.10)", 0.10)
    generate_and_print("3. AGGRESSIVE PARANOIA (Strength = 0.25)", 0.25)

if __name__ == "__main__":
    main()
