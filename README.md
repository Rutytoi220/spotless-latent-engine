# Spotless Latent Reasoning Engine: Continuous Representation Steering for Qwen2.5-Coder

[![Model: Qwen2.5-Coder-7B-Instruct](https://img.shields.io/badge/Base%20Model-Qwen2.5--Coder--7B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
[![Precision: 4-bit NF4](https://img.shields.io/badge/Quantization-BitsAndBytes%204--bit%20NF4-green)]()
[![Adapter: Spotless 448](https://img.shields.io/badge/Adapter%20Bottleneck-448%20dim-orange)]()
[![Benchmark: 100% Pass](https://img.shields.io/badge/Benchmark-6%2F6%20Tasks%20PASS-brightgreen)]()

---

## Abstract

Standard Large Language Model reasoning relies on autoregressive Chain-of-Thought (CoT), forcing the model to collapse internal high-dimensional representations into discrete tokens at every reasoning step. This introduces an unnecessary **"Discretization Tax"**: inflated inference latency, high token consumption, and vulnerability to lexical traps.

The **Spotless Latent Reasoning Engine** demonstrates that reasoning can occur entirely in continuous vector space ($\mathbb{R}^{3584}$) without intermediate token emission. Conditioned on $K=4$ continuous latent thought steps, a compact two-phase bottleneck adapter (**Latent Refiner** + **Full-Sequence Verbalizer Bridge**, bottleneck dimension 448, 6.43M parameters) refines thoughts and bridges the model's representations back to the vocabulary manifold for natural language and code generation.

On cognitive reflection traps (e.g., the Strawberry letter count), the engine produces verified correct answers in **18 tokens and 0.72 seconds**—delivering a **7.5x speedup** over traditional Text Chain-of-Thought while solving problems where standard zero-shot direct generation fails.

---

## Architecture Workflow

```
                                  SPOTLESS LATENT PIPELINE
                                  ========================

 [ User Prompt ]
        │
        ▼
┌────────────────────────┐
│  Prompt Language Phase │  ==> Base Qwen2.5-Coder-7B Transformer
│  (Chat Template & KV)  │      Encodes prompt tokens to initial KV cache and h_0 in R^3584
└──────────┬─────────────┘
           │
           │  h_0
           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    K = 4 Continuous Latent Reasoning Phase                      │
│                                                                                 │
│   h_t ──> [ Spotless Refiner ] ──> L2 Norm & Embed Scale ──> [ Transformer ]    │
│           (d=3584 -> 448 -> 3584)    e_lat = (h / ||h||) * sigma     (Pass with │
│                                                                       KV Cache) │
│                                                                            │    │
│                                           h_{t+1} <────────────────────────┘    │
│                                                                                 │
│   Repeated for K = 1 .. 4 recurrent continuous thought passes                   │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                                           │  Terminal Latent State h_4
                                           ▼
┌────────────────────────────────────────────────────────┐
│               Decoding & Verbalization Phase           │
│                                                        │
│   h_4 ──> [ Spotless Verbalizer ] ──> RMSNorm ──> lm_head ──> First Token Logits
│           (d=3584 -> 448 -> 3584)
│                                                                    │
│   Autoregressive KV-cached decoding continues from first token     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
 [ Verified Code & Solution Output ]
```

---

## Multi-Task Benchmark Matrix

Evaluated across three rigorous test suites using [`run_overnight_benchmark.py`](run_overnight_benchmark.py) comparing **Zero-Shot ($K=0$)**, **Continuous Latent ($K=4$)**, and **Text Chain-of-Thought**:

| Evaluation Suite | Task ID | Problem Description | Zero-Shot ($K=0$) | **Spotless Latent ($K=4$)** | Text CoT | Latent Speedup vs Text CoT |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Suite A: Algorithmic Logic** | `A1_cycle_detection` | Floyd's Cycle Detection ($O(1)$ space) | PASS (180 tok, 5.89s) | **PASS (111 tok, 3.44s)** | PASS (220 tok, 7.78s) | **2.26x faster**, 50% fewer tokens |
| **Suite A: Algorithmic Logic** | `A2_invert_tree` | Recursive Binary Tree Inversion | PASS (145 tok, 4.13s) | **PASS (145 tok, 4.64s)** | PASS (220 tok, 7.59s) | **1.64x faster** |
| **Suite B: Code Optimization** | `B1_find_duplicates_O_N` | $O(N^2) \to O(N)$ Set Deduplication | PASS (180 tok, 5.63s) | **PASS (50 tok, 1.83s)** | PASS (220 tok, 7.43s) | **4.06x faster**, 77% fewer tokens |
| **Suite B: Code Optimization** | `B2_kadane_max_sum` | Kadane's Maximum Subarray Sum | PASS (176 tok, 5.47s) | **PASS (79 tok, 2.63s)** | PASS (220 tok, 7.43s) | **2.83x faster**, 64% fewer tokens |
| **Suite C: Cognitive Traps** | `C1_bat_and_ball` | Bat & Ball Reflection ($1.10 total) | PASS (14 tok, 0.49s) | **PASS (196 tok, 5.86s)** | PASS (220 tok, 7.89s) | Full algebraic derivation ($0.05) |
| **Suite C: Cognitive Traps** | `C2_strawberry_r` | Letter 'r' Count in "strawberry" | **FAIL (18 tok, 0.69s)** | **PASS (18 tok, 0.72s)** | PASS (164 tok, 5.41s) | **7.51x faster**, 89% fewer tokens |

**Summary:** The Spotless Latent Engine attains **6/6 (100%)** task validation while significantly reducing latency and token budgets compared to textual chain-of-thought.

---

## Minimal Inference Snippet

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "best_latent_adapter.pt"

# 1. Define Spotless Latent Adapter Architecture
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

# 2. Load 4-bit Base Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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

embed_norm = model.model.embed_tokens.weight.norm(p=2, dim=-1).mean().item()

# 3. Load Adapter Weights
adapter = SpotlessLatentAdapter(d=3584, bottleneck=448).to("cuda:0", dtype=torch.bfloat16)
adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location="cuda:0"))
adapter.eval()

# 4. Continuous Latent Reasoning Inference (K=4)
prompt = "How many times does the letter 'r' appear in the word 'strawberry'?"
messages = [
    {"role": "system", "content": "Provide a direct, concise, and accurate solution."},
    {"role": "user", "content": prompt}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda:0")

with torch.no_grad():
    # Prompt Phase
    out = model.model(inputs_embeds=model.model.embed_tokens(input_ids), use_cache=True)
    pkv = out.past_key_values
    h = out.last_hidden_state[:, -1:, :]

    # K=4 Latent Thought Refinement Steps
    for _ in range(4):
        h_ref = adapter.refine(h)
        e_lat = (h_ref / h_ref.norm(p=2, dim=-1, keepdim=True)) * embed_norm
        out = model.model(inputs_embeds=e_lat, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        h = out.last_hidden_state[:, -1:, :]

    # Verbalization
    h_verb = adapter.verbalize(h)
    logits = model.lm_head(model.model.norm(h_verb))[:, -1, :]
    next_tok = logits.argmax(dim=-1, keepdim=True)
    tokens = [next_tok.item()]

    # Autoregressive verbalization
    for _ in range(120):
        if next_tok.item() == tokenizer.eos_token_id:
            break
        curr_emb = model.model.embed_tokens(next_tok)
        out = model.model(inputs_embeds=curr_emb, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        logits = model.lm_head(model.model.norm(out.last_hidden_state[:, -1:, :]))[:, -1, :]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        tokens.append(next_tok.item())

print(tokenizer.decode(tokens, skip_special_tokens=True).strip())
# Output: "The letter 'r' appears 3 times in the word 'strawberry'."
```

---

## Checkpoint Artifacts & Repository Contents

- [`best_latent_adapter.pt`](best_latent_adapter.pt): Trained adapter weights at Step 2600 (bottleneck=448, 6.43M params, 12.8MB).
- [`run_overnight_benchmark.py`](run_overnight_benchmark.py): Milestone 3 multi-task benchmark script evaluating $K=0$, $K=4$, and Text CoT.
- [`benchmark_results.json`](benchmark_results.json): Full execution telemetry, token counts, and latencies across all test suites.
- [`train_spotless_engine.py`](train_spotless_engine.py): Full-sequence optimization pipeline with dynamic synthetic curriculum generation.
- [`overnight_report.md`](overnight_report.md): Technical executive briefing detailing post-mortem diagnostic and optimization metrics.
- [`coconut_engine.py`](coconut_engine.py): Foundational continuous latent reasoning engine wrapper.

---

## Citation & License

Apache-2.0 License. If you use this work in your interpretability and continuous reasoning research, please cite this repository.
](https://share.gemini.google/EMF6SmugfyTE)](https://share.gemini.google/EMF6SmugfyTE)
