"""
Standalone isolated diagnostic evaluation script for Step 2600 SpotlessLatentAdapter.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "/var/home/rutytoi/qwen_experiments/best_latent_adapter.pt"

PROMPTS = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "How many times does the letter 'r' appear in the word 'strawberry'?",
    "Write a Python function invert_tree(root) to invert a binary tree recursively."
]

class SpotlessLatentAdapter(nn.Module):
    def __init__(self, d: int = 3584, bottleneck: int = 448):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Linear(d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d)
        )
        self.verbalizer = nn.Sequential(
            nn.Linear(d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d)
        )

    def refine(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.refiner(h)

    def verbalize(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.verbalizer(h)


def run_eval():
    print("=" * 70)
    print("SPOTLESS LATENT ADAPTER: ISOLATED EVALUATION")
    print("=" * 70)

    # 1. Load Tokenizer & 4-bit Base Model
    print(f"[*] Loading tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading 4-bit base model: {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    embed_weight = model.model.embed_tokens.weight
    embed_norm = embed_weight.norm(p=2, dim=-1).mean().item()
    print(f"[+] Base model loaded. Mean embed norm: {embed_norm:.4f}")

    # 2. Instantiate and load adapter
    print(f"[*] Instantiating SpotlessLatentAdapter(d=3584, bottleneck=448)...")
    adapter = SpotlessLatentAdapter(d=3584, bottleneck=448).to("cuda:0", dtype=torch.bfloat16)
    print(f"[*] Loading weights from {ADAPTER_PATH}...")
    state_dict = torch.load(ADAPTER_PATH, map_location="cuda:0")
    adapter.load_state_dict(state_dict)
    adapter.eval()
    print("[+] Adapter loaded successfully.")

    # 3. Test prompts
    K_latent_steps = 4

    for i, prompt in enumerate(PROMPTS, 1):
        print("\n" + "=" * 70)
        print(f"PROMPT {i}: \"{prompt}\"")
        print("=" * 70)

        # Let's test with system prompt matching train_spotless_engine.py
        messages = [
            {"role": "system", "content": "You are a precise, articulate reasoning engine. Solve clearly and thoroughly."},
            {"role": "user", "content": prompt}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda:0")

        # --- GENERATION MODE 1: train_spotless_engine.py Phase 3 style (verbalizer on token 0 only) ---
        with torch.no_grad():
            prompt_embeds = model.model.embed_tokens(input_ids)
            out = model.model(inputs_embeds=prompt_embeds, use_cache=True)
            pkv = out.past_key_values
            h = out.last_hidden_state[:, -1:, :]
            h_0 = h.clone()

            for _ in range(K_latent_steps):
                h_ref = adapter.refine(h)
                e_lat = (h_ref / h_ref.norm(p=2, dim=-1, keepdim=True)) * embed_norm
                out = model.model(inputs_embeds=e_lat, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                h = out.last_hidden_state[:, -1:, :]

            drift = (h - h_0).norm(p=2).item()
            h_verb = adapter.verbalize(h)
            logits = model.lm_head(model.model.norm(h_verb))[:, -1, :]
            
            # Inspect top 5 predicted tokens
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=5)
            top_toks = [(tokenizer.decode([idx.item()]), float(prob.item()), idx.item()) for prob, idx in zip(top_probs[0], top_ids[0])]
            print(f"Latent Drift (h - h0): {drift:.2f}")
            print(f"Top 5 initial tokens from verbalized latent thought:")
            for tok_str, prob_val, tok_id in top_toks:
                print(f"  Token ID {tok_id:6d} ({repr(tok_str):12s}): {prob_val * 100:.2f}%")

            tok = logits.argmax(dim=-1, keepdim=True)
            gen_tokens = [tok.item()]

            for step_idx in range(120):
                if tok.item() == tokenizer.eos_token_id:
                    break
                curr_emb = model.model.embed_tokens(tok)
                out = model.model(inputs_embeds=curr_emb, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                logits = model.lm_head(model.model.norm(out.last_hidden_state[:, -1:, :]))[:, -1, :]
                tok = logits.argmax(dim=-1, keepdim=True)
                gen_tokens.append(tok.item())

            decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
            print(f"\n[Raw Decoded Output - Mode 1: Token 0 verbalized only] ({len(gen_tokens)} tokens):")
            print("--- START OF OUTPUT ---")
            print(decoded_text)
            print("--- END OF OUTPUT ---")

        # --- GENERATION MODE 2: Full-sequence verbalized style (verbalizer on every token step) ---
        with torch.no_grad():
            prompt_embeds = model.model.embed_tokens(input_ids)
            out = model.model(inputs_embeds=prompt_embeds, use_cache=True)
            pkv = out.past_key_values
            h = out.last_hidden_state[:, -1:, :]

            for _ in range(K_latent_steps):
                h_ref = adapter.refine(h)
                e_lat = (h_ref / h_ref.norm(p=2, dim=-1, keepdim=True)) * embed_norm
                out = model.model(inputs_embeds=e_lat, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                h = out.last_hidden_state[:, -1:, :]

            h_verb = adapter.verbalize(h)
            logits = model.lm_head(model.model.norm(h_verb))[:, -1, :]
            tok = logits.argmax(dim=-1, keepdim=True)
            gen_tokens_full = [tok.item()]

            for step_idx in range(120):
                if tok.item() == tokenizer.eos_token_id:
                    break
                curr_emb = model.model.embed_tokens(tok)
                out = model.model(inputs_embeds=curr_emb, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                h_verb_step = adapter.verbalize(out.last_hidden_state[:, -1:, :])
                logits = model.lm_head(model.model.norm(h_verb_step))[:, -1, :]
                tok = logits.argmax(dim=-1, keepdim=True)
                gen_tokens_full.append(tok.item())

            decoded_text_full = tokenizer.decode(gen_tokens_full, skip_special_tokens=False)
            print(f"\n[Raw Decoded Output - Mode 2: Full-sequence verbalized] ({len(gen_tokens_full)} tokens):")
            print("--- START OF OUTPUT ---")
            print(decoded_text_full)
            print("--- END OF OUTPUT ---")


if __name__ == "__main__":
    run_eval()
