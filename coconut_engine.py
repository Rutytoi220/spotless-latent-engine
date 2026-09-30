"""
Coconut Engine: Continuous Latent Reasoning for Qwen2.5-Coder-7B-Instruct
Phase 1 implementation: LatentReasoningWrapper enabling K steps of continuous
vector thoughts prior to natural language verbalization.
"""

import os
import gc
import json
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class LatentReasoningWrapper:
    """
    Wraps Qwen2.5-Coder-7B to perform continuous latent reasoning (Coconut architecture).
    Instead of discrete tokens, reasoning occurs as continuous vector passes in R^d.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: str = "cuda:0",
        quantization_4bit: bool = True,
    ):
        self.device = device
        self.model_id = model_id

        print(f"[LatentReasoningWrapper] Loading tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[LatentReasoningWrapper] Initializing 4-bit base model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

        # Compute empirical mean embedding norm: \bar{\sigma}_embed
        embed_weight = self.model.model.embed_tokens.weight
        self.embed_norm = embed_weight.norm(p=2, dim=-1).mean().item()
        print(f"[LatentReasoningWrapper] Base model initialized. Mean embed norm: {self.embed_norm:.4f}")

    def get_virtual_tokens(self, h: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Projects a continuous latent vector through lm_head to inspect semantic superposition."""
        with torch.no_grad():
            normed_h = self.model.model.norm(h)
            logits = self.model.lm_head(normed_h).squeeze()
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=top_k)
            virtuals = []
            for prob, idx in zip(top_probs, top_ids):
                token_str = self.tokenizer.decode([idx.item()])
                virtuals.append((token_str, float(prob.item())))
            return virtuals

    def generate(
        self,
        prompt: str,
        K: int = 4,
        max_new_tokens: int = 200,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
        system_prompt: Optional[str] = "You are a precise algorithmic problem solver. Solve step by step.",
        inspect_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the three-phase continuous reasoning pipeline:
        1. Language Phase: Encode prompt into initial KV-cache and terminal hidden state h_0.
        2. Latent Phase: Recurrently update continuous state h_{t+1} = Transformer(norm(h_t) * sigma).
        3. Decoding Phase: Decode autoregressively from terminal latent thought h_K.
        """
        start_time = time.time()
        
        # Build prompt using chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        trajectory = []

        with torch.no_grad():
            # ==========================================
            # PHASE 1: LANGUAGE PHASE (Prompt encoding)
            # ==========================================
            prompt_embeds = self.model.model.embed_tokens(input_ids)
            out = self.model.model(inputs_embeds=prompt_embeds, use_cache=True)
            pkv = out.past_key_values
            h = out.last_hidden_state[:, -1:, :]
            h_0 = h.clone()

            if inspect_trajectory:
                trajectory.append({
                    "step": 0,
                    "type": "prompt_terminal",
                    "norm": float(h.norm(p=2).item()),
                    "cosine_sim_to_prev": 1.0,
                    "drift_from_h0": 0.0,
                    "virtual_tokens": self.get_virtual_tokens(h),
                })

            # ==========================================
            # PHASE 2: LATENT REASONING PHASE (K steps)
            # ==========================================
            latent_start_time = time.time()
            for k in range(1, K + 1):
                prev_h = h
                # Normalize and project to embedding manifold
                h_unit = h / h.norm(p=2, dim=-1, keepdim=True)
                e_latent = h_unit * self.embed_norm

                # Continuous forward step through the full transformer
                out = self.model.model(inputs_embeds=e_latent, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                h = out.last_hidden_state[:, -1:, :]

                if inspect_trajectory:
                    cosine_sim = F.cosine_similarity(h.flatten(), prev_h.flatten(), dim=0).item()
                    drift = (h - h_0).norm(p=2).item()
                    trajectory.append({
                        "step": k,
                        "type": "latent_thought",
                        "norm": float(h.norm(p=2).item()),
                        "cosine_sim_to_prev": float(cosine_sim),
                        "drift_from_h0": float(drift),
                        "virtual_tokens": self.get_virtual_tokens(h),
                    })

            latent_time = time.time() - latent_start_time

            # ==========================================
            # PHASE 3: DECODING PHASE (Verbalization)
            # ==========================================
            verbalization_start_time = time.time()
            generated_token_ids = []

            # Project terminal latent thought to first token logits
            normed_h = self.model.model.norm(h)
            logits = self.model.lm_head(normed_h)[:, -1, :]

            # Select first verbalized token
            if do_sample and temperature > 0.0:
                scaled_logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            generated_token_ids.append(token_id)
            time_to_first_token = time.time() - verbalization_start_time

            # Autoregressive verbalization loop
            while len(generated_token_ids) < max_new_tokens:
                if token_id == self.tokenizer.eos_token_id:
                    break

                # Embed token and pass with KV-cache
                curr_embed = self.model.model.embed_tokens(next_token)
                out = self.model.model(inputs_embeds=curr_embed, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                normed_out = self.model.model.norm(out.last_hidden_state[:, -1:, :])
                logits = self.model.lm_head(normed_out)[:, -1, :]

                if do_sample and temperature > 0.0:
                    scaled_logits = logits / temperature
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        scaled_logits[indices_to_remove] = -float("Inf")
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                token_id = next_token.item()
                generated_token_ids.append(token_id)

        total_time = time.time() - start_time
        output_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        return {
            "prompt": prompt,
            "K_latent_steps": K,
            "generated_text": output_text,
            "num_tokens_verbalized": len(generated_token_ids),
            "total_latency_sec": total_time,
            "latent_latency_sec": latent_time,
            "time_to_first_token_sec": time_to_first_token,
            "trajectory": trajectory,
        }


def run_milestone1_verification():
    """Verifies K in {0, 2, 4, 8} continuous thought steps on an algorithmic benchmark problem."""
    print("=" * 70)
    print("MILESTONE 1 VERIFICATION: Coconut Engine Architecture")
    print("=" * 70)

    engine = LatentReasoningWrapper()

    test_prompt = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer step-by-step."

    print(f"\nPrompt: {test_prompt}\n")

    for K in [0, 2, 4, 8]:
        print(f"\n--- Testing K = {K} Continuous Thought Vectors ---")
        result = engine.generate(
            prompt=test_prompt,
            K=K,
            max_new_tokens=150,
            do_sample=False,  # Deterministic comparison
            inspect_trajectory=True,
        )
        print(f"Generated text ({result['num_tokens_verbalized']} tokens, {result['total_latency_sec']:.2f}s total):")
        print(result["generated_text"][:200] + ("..." if len(result["generated_text"]) > 200 else ""))
        if result["trajectory"]:
            print(f"Terminal Latent Thought Norm: {result['trajectory'][-1]['norm']:.2f}, Drift: {result['trajectory'][-1]['drift_from_h0']:.2f}")

    print("\n[+] Milestone 1 Architecture Verification Complete!")


if __name__ == "__main__":
    run_milestone1_verification()
