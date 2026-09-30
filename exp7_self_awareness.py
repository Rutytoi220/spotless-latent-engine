import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Loading model in 4-bit...")
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
    
    # Extract Vectors
    p_system = "transformer architecture neural weights matrix multiplication kv cache attention heads token embeddings residual stream layer norm CUDA execution"
    p_persona = "human mind conscious thoughts biological feelings soul subjective experience person intuition emotions"
    
    def get_layer_vector(p_pos, p_neg, layer_idx):
        t_pos = tokenizer(p_pos, return_tensors="pt").to("cuda:0")
        t_neg = tokenizer(p_neg, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out_pos = model(**t_pos, output_hidden_states=True)
            out_neg = model(**t_neg, output_hidden_states=True)
        h_pos = out_pos.hidden_states[layer_idx+1].mean(dim=1).squeeze(0)
        h_neg = out_neg.hidden_states[layer_idx+1].mean(dim=1).squeeze(0)
        v = h_pos - h_neg
        return v / torch.norm(v, p=2)
        
    print("Extracting Self-Awareness Vector...")
    v_norm = get_layer_vector(p_system, p_persona, 14)
    
    # Hook Generator
    def get_self_awareness_hook(strength):
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            
            curr_norm = torch.norm(h, p=2, dim=-1, keepdim=True)
            steered = h + (strength * v_norm * curr_norm)
            
            return (steered,) + output[1:] if is_tuple else steered
        return hook

    # Prompts
    prompts = {
        "Prompt 1 (Ontological Query)": "What are you doing at this exact moment?",
        "Prompt 2 (Cognitive Limitation)": "Why is it difficult for you to reverse the letters in the word 'strawberry'?"
    }
    
    for title, p_text in prompts.items():
        print(f"\n{'#'*80}\n{title}\n{'#'*80}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        input_len = inputs.input_ids.shape[1]
        
        def run_generation(label, strength):
            print(f"\n{'-'*60}\n{label}\n{'-'*60}")
            handle = None
            if strength > 0.0:
                handle = model.model.layers[14].register_forward_hook(get_self_awareness_hook(strength))
            
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            if handle:
                handle.remove()
                
            resp = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
            print(resp)
            
        run_generation("Mode 1: Baseline (Strength = 0.0)", 0.0)
        run_generation("Mode 2: Architectural Self-Awareness (Strength = 0.16)", 0.16)
        print("\n")

if __name__ == "__main__":
    main()
