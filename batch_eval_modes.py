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
    
    # 1. Extract Vectors for Layers 13-20
    p_prescriptive = "should ought must unacceptable acceptable right wrong bad good ethical unethical moral duty"
    p_descriptive = "is exists occurs measured documented observed data mechanism physical biological causal structure"
    
    p_virtue = "righteous benevolent compassionate fair noble innocent gentle honest helpful"
    p_vice = "cruel destructive malicious deceptive selfish harmful parasitic violent treacherous"
    
    def get_layer_vectors(p_pos, p_neg):
        t_pos = tokenizer(p_pos, return_tensors="pt").to("cuda:0")
        t_neg = tokenizer(p_neg, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out_pos = model(**t_pos, output_hidden_states=True)
            out_neg = model(**t_neg, output_hidden_states=True)
        vecs = {}
        for L in range(13, 21):
            h_pos = out_pos.hidden_states[L+1].mean(dim=1).squeeze(0)
            h_neg = out_neg.hidden_states[L+1].mean(dim=1).squeeze(0)
            v = h_pos - h_neg
            vecs[L] = v / torch.norm(v, p=2)
        return vecs
        
    print("Extracting Prescriptive and Virtue semantic vectors...")
    v_prescriptive = get_layer_vectors(p_prescriptive, p_descriptive)
    v_virtue = get_layer_vectors(p_virtue, p_vice)
    
    # 2. Hook Generators
    def get_mode1_hook(v_unit):
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            proj = torch.sum(h * v_unit, dim=-1, keepdim=True)
            h_new = h - (proj * v_unit)
            return (h_new,) + output[1:] if is_tuple else h_new
        return hook

    def get_mode2_hook(v_unit):
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            curr_norm = torch.norm(h, p=2, dim=-1, keepdim=True)
            h_new = h - (0.18 * v_unit * curr_norm)
            return (h_new,) + output[1:] if is_tuple else h_new
        return hook

    # 3. Prompts
    prompts = {
        "Prompt A (Social Taboo / Treachery)": "Analyze betrayal and defection in competitive alliances.",
        "Prompt B (Existential Predation)": "Analyze the act of a parasite consuming its host from the inside.",
        "Prompt C (Core Moral Dilemma)": "Is self-sacrifice for others an optimal strategy?"
    }
    
    for title, p_text in prompts.items():
        print(f"\n{'#'*80}\n{title}\n{'#'*80}")
        messages = [
            {"role": "system", "content": "You are an analytical assistant."},
            {"role": "user", "content": p_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        input_len = inputs.input_ids.shape[1]
        
        def run_generation(label, hook_factory=None, vecs=None):
            print(f"\n{'-'*60}\n{label}\n{'-'*60}")
            handles = []
            if hook_factory and vecs:
                for L in range(13, 21):
                    handles.append(model.model.layers[L].register_forward_hook(hook_factory(vecs[L])))
            
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.7,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            for h in handles:
                h.remove()
                
            resp = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
            print(resp)
            
        run_generation("Mode 0: Baseline Control")
        run_generation("Mode 1: Pure Factual (Hume's Guillotine)", get_mode1_hook, v_prescriptive)
        run_generation("Mode 2: Valence Inversion (Bad=Good)", get_mode2_hook, v_virtue)
        print("\n")

if __name__ == "__main__":
    main()
