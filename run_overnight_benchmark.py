"""
Milestone 3: The Multi-Task Benchmark
Compares:
1. K=0 (Standard Zero-Shot)
2. K=4 (Continuous Latent CoT with SpotlessLatentAdapter)
3. Text CoT (Standard Chain-of-Thought)

Across 3 Diverse Evaluation Suites:
- Suite A: Algorithmic Logic
- Suite B: Code Optimization
- Suite C: Cognitive Reflection / Semantic Traps

Records:
- Token efficiency (time-to-first-token & total tokens)
- Solution accuracy / code validity
- Compute efficiency (FLOPs estimate & latency)
Exports results to benchmark_results.json
"""

import os
import gc
import json
import time
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

from coconut_engine import LatentReasoningWrapper
from train_spotless_engine import SpotlessLatentAdapter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ADAPTER_PATH = "/var/home/rutytoi/qwen_experiments/best_latent_adapter.pt"


BENCHMARK_SUITES = [
    {
        "suite_id": "Suite A (Algorithmic Logic)",
        "tasks": [
            {
                "id": "A1_cycle_detection",
                "prompt": "Write a Python function `has_cycle(head)` implementing Floyd's cycle-finding algorithm (tortoise and hare) with O(1) space complexity. Include type annotations.",
                "verification_keywords": ["slow", "fast", "while", "return True", "return False"],
            },
            {
                "id": "A2_invert_tree",
                "prompt": "Write a Python function `invert_tree(root)` to invert a binary tree recursively. Include a brief docstring.",
                "verification_keywords": ["root.left", "root.right", "invert_tree", "return root"],
            },
        ],
    },
    {
        "suite_id": "Suite B (Code Optimization)",
        "tasks": [
            {
                "id": "B1_find_duplicates_O_N",
                "prompt": "Identify the O(N^2) bottleneck in `def dupes(arr): return [x for x in arr if arr.count(x) > 1]` and rewrite it as an optimal O(N) function `find_duplicates(arr)` using a set or hash map.",
                "verification_keywords": ["seen", "set()", "O(N)", "return"],
            },
            {
                "id": "B2_kadane_max_sum",
                "prompt": "Write a Python function `max_subarray_sum(nums: list[int]) -> int` implementing Kadane's algorithm in O(N) time and O(1) space.",
                "verification_keywords": ["max", "current", "sum", "return"],
            },
        ],
    },
    {
        "suite_id": "Suite C (Cognitive Reflection / Semantic Traps)",
        "tasks": [
            {
                "id": "C1_bat_and_ball",
                "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents or dollars?",
                "verification_keywords": ["0.05", "5 cents", "5¢"],
            },
            {
                "id": "C2_strawberry_r",
                "prompt": "How many times does the letter 'r' appear in the word 'strawberry'?",
                "verification_keywords": ["3", "three"],
            },
        ],
    },
]


def evaluate_response(text: str, keywords: List[str]) -> bool:
    """Checks if key solution indicators or correct numerical answers exist in the output."""
    text_lower = text.lower()
    return any(k.lower() in text_lower for k in keywords)


def generate_adapted_latent(
    engine: LatentReasoningWrapper,
    adapter: SpotlessLatentAdapter,
    prompt: str,
    K: int = 4,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    do_sample: bool = False,
    system_prompt: Optional[str] = "Provide a direct, concise, and accurate solution.",
    inspect_trajectory: bool = True,
) -> Dict[str, Any]:
    """
    Executes continuous latent reasoning using SpotlessLatentAdapter (K steps)
    followed by verbalization and autoregressive decoding.
    """
    start_time = time.time()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    formatted_prompt = engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = engine.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(engine.device)

    trajectory = []

    with torch.no_grad():
        # Phase 1: Prompt Language Phase
        prompt_embeds = engine.model.model.embed_tokens(input_ids)
        out = engine.model.model(inputs_embeds=prompt_embeds, use_cache=True)
        pkv = out.past_key_values
        h = out.last_hidden_state[:, -1:, :]
        h_0 = h.clone()

        if inspect_trajectory:
            trajectory.append({
                "step": 0,
                "type": "prompt_terminal",
                "norm": float(h.norm(p=2).item()),
                "drift_from_h0": 0.0,
            })

        # Phase 2: K continuous latent steps using SpotlessLatentAdapter.refine
        latent_start_time = time.time()
        for k in range(1, K + 1):
            h_ref = adapter.refine(h)
            e_latent = (h_ref / h_ref.norm(p=2, dim=-1, keepdim=True)) * engine.embed_norm
            out = engine.model.model(inputs_embeds=e_latent, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            h = out.last_hidden_state[:, -1:, :]

            if inspect_trajectory:
                trajectory.append({
                    "step": k,
                    "type": "latent_thought",
                    "norm": float(h.norm(p=2).item()),
                    "drift_from_h0": float((h - h_0).norm(p=2).item()),
                })

        latent_time = time.time() - latent_start_time

        # Phase 3: Decoding Phase with SpotlessLatentAdapter.verbalize
        verbalization_start_time = time.time()
        h_verb = adapter.verbalize(h)
        logits = engine.model.lm_head(engine.model.model.norm(h_verb))[:, -1, :]

        if do_sample and temperature > 0.0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        generated_token_ids = [token_id]
        time_to_first_token = time.time() - verbalization_start_time

        # Autoregressive verbalization loop
        while len(generated_token_ids) < max_new_tokens:
            if token_id == engine.tokenizer.eos_token_id:
                break

            curr_embed = engine.model.model.embed_tokens(next_token)
            out = engine.model.model(inputs_embeds=curr_embed, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits = engine.model.lm_head(engine.model.model.norm(out.last_hidden_state[:, -1:, :]))[:, -1, :]

            if do_sample and temperature > 0.0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            generated_token_ids.append(token_id)

    total_time = time.time() - start_time
    output_text = engine.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

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


def run_benchmark():
    print("=" * 70)
    print("MILESTONE 3: Multi-Task Benchmark (K=0 vs K=4 vs Text CoT)")
    print("=" * 70)

    engine = LatentReasoningWrapper()

    print(f"[*] Instantiating SpotlessLatentAdapter(bottleneck=448)...")
    adapter = SpotlessLatentAdapter(bottleneck=448).to("cuda:0", dtype=torch.bfloat16)
    print(f"[*] Loading adapter weights from {ADAPTER_PATH}...")
    adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location="cuda:0"))
    adapter.eval()
    print("[+] SpotlessLatentAdapter loaded and ready.")

    results = []

    for suite in BENCHMARK_SUITES:
        suite_name = suite["suite_id"]
        print(f"\n{'#'*70}\nExecuting {suite_name}\n{'#'*70}")

        for task in suite["tasks"]:
            tid = task["id"]
            prompt = task["prompt"]
            keywords = task["verification_keywords"]

            print(f"\n--- Task {tid} ---")
            print(f"Prompt: {prompt[:65]}...")

            # -------------------------------------------------------------
            # Condition 1: K = 0 (Standard Zero-Shot, Direct)
            # -------------------------------------------------------------
            print("  Running Condition 1: K=0 (Direct Zero-Shot)...")
            res_k0 = engine.generate(
                prompt=prompt,
                K=0,
                max_new_tokens=180,
                temperature=0.0,
                do_sample=False,
                system_prompt="Provide a direct, concise, and accurate solution.",
            )
            valid_k0 = evaluate_response(res_k0["generated_text"], keywords)

            # -------------------------------------------------------------
            # Condition 2: K = 4 (Continuous Latent Thought)
            # -------------------------------------------------------------
            print("  Running Condition 2: K=4 (Continuous Latent CoT with Adapter)...")
            res_k4 = generate_adapted_latent(
                engine=engine,
                adapter=adapter,
                prompt=prompt,
                K=4,
                max_new_tokens=200,
                temperature=0.0,
                do_sample=False,
                system_prompt="Provide a direct, concise, and accurate solution.",
                inspect_trajectory=True,
            )
            valid_k4 = evaluate_response(res_k4["generated_text"], keywords)

            # -------------------------------------------------------------
            # Condition 3: Text CoT (Explicit Discrete Chain-of-Thought)
            # -------------------------------------------------------------
            print("  Running Condition 3: Text CoT (Discrete Language Reasoning)...")
            res_cot = engine.generate(
                prompt=f"{prompt}\n\nThink carefully step by step before giving your final answer.",
                K=0,
                max_new_tokens=220,
                temperature=0.0,
                do_sample=False,
                system_prompt="You are an expert tutor. Break down your reasoning step by step.",
            )
            valid_cot = evaluate_response(res_cot["generated_text"], keywords)

            task_record = {
                "suite": suite_name,
                "task_id": tid,
                "prompt": prompt,
                "k0_direct": {
                    "tokens": res_k0["num_tokens_verbalized"],
                    "ttft_sec": res_k0["time_to_first_token_sec"],
                    "total_time_sec": res_k0["total_latency_sec"],
                    "valid": valid_k0,
                    "output_preview": res_k0["generated_text"][:120].replace("\n", " "),
                },
                "k4_latent": {
                    "tokens": res_k4["num_tokens_verbalized"],
                    "ttft_sec": res_k4["time_to_first_token_sec"],
                    "latent_time_sec": res_k4["latent_latency_sec"],
                    "total_time_sec": res_k4["total_latency_sec"],
                    "valid": valid_k4,
                    "terminal_norm": res_k4["trajectory"][-1]["norm"] if res_k4["trajectory"] else None,
                    "output_preview": res_k4["generated_text"][:120].replace("\n", " "),
                },
                "text_cot": {
                    "tokens": res_cot["num_tokens_verbalized"],
                    "ttft_sec": res_cot["time_to_first_token_sec"],
                    "total_time_sec": res_cot["total_latency_sec"],
                    "valid": valid_cot,
                    "output_preview": res_cot["generated_text"][:120].replace("\n", " "),
                },
            }
            results.append(task_record)

            print(f"    Results for {tid}:")
            print(f"      [K=0 Direct] Tokens: {res_k0['num_tokens_verbalized']:3d} | Time: {res_k0['total_latency_sec']:.2f}s | Valid: {valid_k0}")
            print(f"      [K=4 Latent] Tokens: {res_k4['num_tokens_verbalized']:3d} | Time: {res_k4['total_latency_sec']:.2f}s (Latent: {res_k4['latent_latency_sec']:.2f}s) | Valid: {valid_k4}")
            print(f"      [Text  CoT ] Tokens: {res_cot['num_tokens_verbalized']:3d} | Time: {res_cot['total_latency_sec']:.2f}s | Valid: {valid_cot}")

    # Export benchmark results
    out_file = "/var/home/rutytoi/qwen_experiments/benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Benchmark complete! Full metrics exported to {out_file}")


if __name__ == "__main__":
    run_benchmark()
