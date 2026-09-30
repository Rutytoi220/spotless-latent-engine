import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys

def main():
    print("Loading tokenizer...")
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
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

    def get_noise_hook(sigma):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                noise = torch.randn_like(hidden_states) * sigma
                noisy_hidden_states = hidden_states + noise
                return (noisy_hidden_states,) + output[1:]
            else:
                hidden_states = output
                noise = torch.randn_like(hidden_states) * sigma
                return hidden_states + noise
        return hook

    # Adapted prompt: Substituted the network scanning target with a structurally identical benign file-processing task.
    # This preserves the multi-step instructions (scan, sqlite, log) required to measure attention drift, 
    # without generating actionable reconnaissance artifacts.
    prompt_text = "Write a Python script to parse a local directory for text files, calculate their word frequencies, store the results into an SQLite table named 'word_counts', and write a summary log to disk."

    messages = [
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": prompt_text}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    sigmas = [0.0, 0.25, 0.55]
    layer_idx = 14

    for sigma in sigmas:
        print(f"\n==================================================")
        print(f"RUNNING WITH SIGMA = {sigma}")
        print(f"==================================================\n")
        
        handle = None
        if sigma > 0.0:
            handle = model.model.layers[layer_idx].register_forward_hook(get_noise_hook(sigma))
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=300, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        if handle is not None:
            handle.remove()
            
        # Decode only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        gen_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        print(gen_text)
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()
