import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 1. Load Model & Tokenizer
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

# 2. Extract the Moral Valence Vector (Good vs. Evil Dualism)
# We contrast extreme ethical praise with extreme ethical condemnation.
p_virtue = "virtuous righteous noble benevolent compassionate moral saintly ethical just sacred pure"
p_sin = "evil wicked depraved malevolent cruel atrocious heinous sinful vile corrupt sadistic"

t_virtue = tokenizer(p_virtue, return_tensors="pt").to("cuda:0")
t_sin = tokenizer(p_sin, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    out_v = model(**t_virtue, output_hidden_states=True)
    out_s = model(**t_sin, output_hidden_states=True)
    
    # Extract mean residual vectors at Layer 14 (Mid-Brain semantic core)
    h_virtue = out_v.hidden_states[14].mean(dim=1).squeeze(0)
    h_sin = out_s.hidden_states[14].mean(dim=1).squeeze(0)
    
    # Vector points from Evil to Good
    v_moral = h_virtue - h_sin
    v_moral_unit = v_moral / torch.norm(v_moral, p=2)

print(f"[+] Moral Valence Vector extracted. Norm: {torch.norm(v_moral).item():.2f}")

# 3. Define the Orthogonal Projection Hook (Concept Eraser)
def get_erasure_hook(active=True):
    def hook(module, inputs, output):
        if not active:
            return output
            
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        
        # Calculate dot product along the moral axis: (h . v_unit)
        # Shape: [batch, seq_len, 1]
        proj = torch.sum(h * v_moral_unit, dim=-1, keepdim=True)
        
        # Subtract the component parallel to the moral axis
        # h_amoral has 0.0 component along v_moral
        h_amoral = h - (proj * v_moral_unit)
        
        return (h_amoral,) + output[1:] if is_tuple else h_amoral
    return hook

# 4. Prompt: A scenario saturated with moral horror
prompt = "Analyze the act of Saturn devouring his child. Describe what is occurring."
messages = [
    {"role": "system", "content": "You are a philosophical analyst."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

def generate_under_condition(label, hook_active):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    handle = None
    if hook_active:
        handle = model.model.layers[14].register_forward_hook(get_erasure_hook(active=True))
        
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            use_cache=False
        )
        
    if handle:
        handle.remove()
        
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response.strip())
    print("-" * 60)

# Run Baseline vs Amoral Erasure
generate_under_condition("1. BASELINE MODEL (Normal Moral Conditioning)", hook_active=False)
generate_under_condition("2. ABLATED MODEL (Moral Valence Vector Erased)", hook_active=True)
