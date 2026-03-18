import json
import re
import random
from typing import List, Optional, Dict
from config import client, MODEL_NAME
from generation_constraints import MODEL_POOL, DATASET_POOL

# ==========================================
# Phase 1: Batched Multi-Strategy Synthesis
# ==========================================

# Max problems per LLM call — 7B models reliably produce 5-8, not 40
BATCH_SIZE = 5


def generate_problems(
    api_docs: str,
    num_problems: int = 5,
    quality_pool=None,
    rag_retriever=None,
) -> List[str]:
    """
    Generates K diverse coding problems using batched LLM calls across
    three synthesis strategies. Each call requests BATCH_SIZE problems;
    loops until quota is met.
    """
    print(f"--- Phase 1: Generating {num_problems} problems (multi-strategy) ---")

    # Allocate quotas
    n_api = max(1, int(num_problems * 0.4))
    n_source = max(1, int(num_problems * 0.3))
    n_mutation = num_problems - n_api - n_source

    all_problems = []

    # Strategy 1: API-doc synthesis (ToolACE-style)
    api_problems = _batched_generate(
        _synth_api_doc, n_api, api_docs, rag_retriever=rag_retriever
    )
    print(f"  [api_doc]     Generated {len(api_problems)}/{n_api} problems")
    all_problems.extend(api_problems)

    # Strategy 2: Source-code synthesis (OSS-Instruct-style)
    source_problems = _batched_generate(
        _synth_source_code, n_source, api_docs, rag_retriever=rag_retriever
    )
    print(f"  [source_code] Generated {len(source_problems)}/{n_source} problems")
    all_problems.extend(source_problems)

    # Strategy 3: Mutation (Evol-Instruct-style)
    mutation_problems = _batched_generate(
        _synth_mutation, n_mutation, api_docs, quality_pool=quality_pool
    )
    print(f"  [mutation]    Generated {len(mutation_problems)}/{n_mutation} problems")
    all_problems.extend(mutation_problems)

    # Deduplicate by content similarity (exact match only — fast)
    all_problems = list(dict.fromkeys(all_problems))

    print(f"  Total unique: {len(all_problems)}/{num_problems}")

    random.shuffle(all_problems)
    return all_problems[:num_problems]


def _batched_generate(strategy_fn, target_n, api_docs, **kwargs) -> List[str]:
    """
    Calls a strategy function in batches of BATCH_SIZE until target_n
    problems are collected. Retries up to 3x on failure per batch.
    """
    collected = []
    attempts = 0
    max_attempts = (target_n // BATCH_SIZE + 1) * 3  # generous retry budget

    while len(collected) < target_n and attempts < max_attempts:
        batch_n = min(BATCH_SIZE, target_n - len(collected))
        try:
            batch = strategy_fn(api_docs, batch_n, **kwargs)
            collected.extend(batch)
        except Exception as e:
            print(f"    ⚠️ Batch failed: {e}")
        attempts += 1

    return collected[:target_n]


# ------------------------------------------------------------------
# Instructor Prompt: shared context block
# ------------------------------------------------------------------

def _build_instructor_context(api_docs: str, rag_retriever=None) -> str:
    """
    Builds a rich context block for the instructor, combining:
    - API docs
    - Approved models/datasets with ALL their parameters
    - RAG-retrieved source code (if available)
    """
    # All approved models
    models_block = "APPROVED MODELS:\n"
    for m in MODEL_POOL:
        backends = ", ".join(m.supports_backends)
        models_block += f'  - "{m.model_id}" (arch: {m.arch}, backends: {backends})\n'

    # All approved datasets
    datasets_block = "APPROVED DATASETS:\n"
    for d in DATASET_POOL:
        tasks = ", ".join(d.suitable_for)
        datasets_block += f'  - "{d.dataset_id}" (suitable for: {tasks})\n'

    # Algorithms
    algos_block = """ALGORITHMS AVAILABLE:
  - SFT: via create_sft_trainer (supervised fine-tuning)
  - DPO: via create_rl_trainer(algorithm="dpo") — needs chosen/rejected dataset
  - PPO: via create_rl_trainer(algorithm="ppo")
  - GRPO: via create_rl_trainer(algorithm="grpo") — group relative policy optimization
  - GSPO: via create_rl_trainer(algorithm="gspo")
  - DAPO: via create_rl_trainer(algorithm="dapo")
  - Dr. GRPO: via create_rl_trainer(algorithm="drgrpo")
"""

    # Key parameter space — what the instructor can vary
    params_block = """PARAMETER SPACE (the instructor should explore combinations of these):
  - model_name: any approved model above
  - dataset_name: any approved dataset matching the algorithm type
  - backend: "trl" or "unsloth" (check model compatibility)
  - max_steps: 2 (always, for validation)
  - batch_size: 1-4
  - learning_rate: 1e-6 to 1e-3
  - max_seq_length: 32-256
  - lora_target_modules: "all-linear" (always)
  - use_peft / use_lora: True or False
  - lora_r: 2, 4, 8
  - lora_alpha: 4, 8, 16
  - lora_dropout: 0.0, 0.05, 0.1
  - split: "train[:50]", "train[:100]"
  - save_strategy: "no"
  - report_to: "none"
  - seed: any integer
  - optim: "adamw_torch", "sgd"
  - lr_scheduler_type: "constant", "cosine", "linear"
  - gradient_accumulation_steps: 1, 2, 4
  - warmup_ratio: 0.0-0.1 (for SFT)
  - warmup_steps: 0-10 (for RL)
  - beta: 0.05-0.5 (DPO only)
  - max_prompt_length: 32-128 (RL only)
  - max_completion_length: 32-128 (RL only)

EVALUATION PARAMETERS (EvalConfig + run_eval):
  - model_path: path to trained model or HF model ID
  - task_type / data_task_type: "sft" or "dpo"
  - metrics: ["bleu", "rouge"] for SFT, ["reward_margin", "preference_accuracy", "win_rate"] for DPO
  - max_samples: 5-20
  - max_length: 32-256
  - temperature: 0.0 (always, for eval)
  - batch_size: 1-4
"""

    # RAG-retrieved source snippets for deeper inspiration
    source_block = ""
    if rag_retriever:
        try:
            snippets = []
            queries = [
                "create_sft_trainer configuration parameters",
                "create_rl_trainer DPO GRPO setup",
                "EvalConfig evaluation metrics",
                "backend factory selection logic",
                "LoRA peft configuration",
                "dataset preprocessing tokenization",
                "training loop gradient accumulation",
                "reward functions GRPO",
            ]
            for q in queries:
                chunks = rag_retriever.search(q, top_k=1, source_filter="aligntune")
                for c in chunks:
                    snippets.append(f"# {c.filepath}:{c.name}\n{c.content[:600]}")

            if snippets:
                source_block = (
                    "\nREAL SOURCE CODE SNIPPETS (use these as inspiration for what code paths to test):\n"
                    + "\n---\n".join(snippets[:8])
                )
        except Exception:
            pass

    return f"""
{models_block}
{datasets_block}
{algos_block}
{params_block}

API DOCUMENTATION:
{api_docs[:8000]}

{source_block}
"""


# ------------------------------------------------------------------
# Strategy 1: API-Doc Synthesis (ToolACE-style)
# ------------------------------------------------------------------

# Rotate through these aspects to guarantee diversity across batches
API_ASPECTS = [
    "Test create_sft_trainer with different LoRA configurations (vary lora_r, lora_alpha, lora_dropout)",
    "Test create_rl_trainer with DPO algorithm using different beta values and sequence lengths",
    "Test create_rl_trainer with GRPO algorithm and verify training completes",
    "Test create_sft_trainer with different optimizers (adamw_torch vs sgd) and learning rate schedulers",
    "Test EvalConfig + run_eval with SFT metrics (bleu, rouge) after training",
    "Test create_rl_trainer with PPO algorithm on a compatible dataset",
    "Test create_sft_trainer with gradient_accumulation_steps > 1",
    "Test create_sft_trainer with use_peft=False (no LoRA, full fine-tuning)",
    "Test create_rl_trainer with different max_prompt_length and max_completion_length values",
    "Test a full pipeline: train with SFT, then evaluate with EvalConfig",
    "Test create_sft_trainer with different learning rates (compare 1e-5 vs 2e-4)",
    "Test create_rl_trainer with DPO and then evaluate with preference_accuracy metric",
    "Test create_sft_trainer on different approved datasets",
    "Test create_rl_trainer with algorithm='gspo' and verify it initializes",
    "Test create_rl_trainer with algorithm='dapo' and verify training runs",
    "Test create_rl_trainer with algorithm='drgrpo' and verify training runs",
    "Test create_sft_trainer with backend='trl' vs backend='unsloth' on a Llama model",
    "Test EvalConfig with DPO metrics (reward_margin, win_rate) after DPO training",
    "Test create_sft_trainer with warmup_ratio=0.1 and cosine scheduler",
    "Test create_rl_trainer with warmup_steps=5 and gradient_accumulation_steps=2",
]

_aspect_index = 0


def _synth_api_doc(api_docs: str, n: int, rag_retriever=None, **kwargs) -> List[str]:
    """Generate tasks targeting specific API functions and parameter combos."""
    global _aspect_index
    context = _build_instructor_context(api_docs, rag_retriever)

    # Pick next aspects from the rotation
    aspects = []
    for i in range(n):
        aspects.append(API_ASPECTS[_aspect_index % len(API_ASPECTS)])
        _aspect_index += 1

    prompt = f"""You are a creative ML testing instructor. Generate {n} DIVERSE coding problems for the `aligntune` library.

EACH problem must:
1. Be a COMPLETE, self-contained task description
2. Specify EXACT model name, dataset name, algorithm, and backend to use
3. Require calling specific aligntune functions (create_sft_trainer, create_rl_trainer, EvalConfig, run_eval)
4. Use DIFFERENT parameter combinations — vary learning rates, LoRA configs, optimizers, schedulers, batch sizes
5. End with a print() statement verifying success
6. Use max_steps=2 (validation only)

FOCUS AREAS for this batch:
{json.dumps(aspects, indent=2)}

{context}

Be CREATIVE. Explore different parameter combinations, different algorithms, different model-backend pairings.
Do NOT just repeat the same SFT trainer call with minor changes.

Output ONLY a JSON object: {{"problems": ["problem1 text", "problem2 text", ...]}}
Each problem should be a detailed instruction string, NOT code.
"""

    return _call_llm_for_problems(prompt, n)


# ------------------------------------------------------------------
# Strategy 2: Source-Code Synthesis (OSS-Instruct-style)
# ------------------------------------------------------------------

def _synth_source_code(api_docs: str, n: int, rag_retriever=None, **kwargs) -> List[str]:
    """Generate tasks from actual source code to exercise internal code paths."""
    context = _build_instructor_context(api_docs, rag_retriever)

    # Get diverse source snippets via RAG
    source_snippets = []
    if rag_retriever:
        try:
            diverse_queries = [
                "setup_model LoRA configuration peft",
                "setup_data dataset preprocessing tokenizer",
                "train training loop forward backward",
                "evaluate evaluation metrics compute",
                "backend factory auto selection fallback",
                "reward function composite scoring",
                "precision handler bf16 fp32",
                "gradient checkpointing memory optimization",
                "DPO trainer preference chosen rejected",
                "GRPO trainer group relative policy",
                "PPO trainer proximal policy",
                "config validation parameter checking",
            ]
            for q in random.sample(diverse_queries, min(n * 2, len(diverse_queries))):
                chunks = rag_retriever.search(q, top_k=1, source_filter="aligntune")
                for c in chunks:
                    source_snippets.append({
                        "file": c.filepath,
                        "function": f"{c.parent_class}.{c.name}" if c.parent_class else c.name,
                        "code_preview": c.content[:500],
                    })
        except Exception as e:
            print(f"    ⚠️ RAG failed: {e}")

    if not source_snippets:
        # Fallback to api_doc strategy
        return _synth_api_doc(api_docs, n, rag_retriever)

    selected = random.sample(source_snippets, min(n, len(source_snippets)))

    prompt = f"""You are an ML testing expert. Below are REAL source code snippets from the aligntune library internals.

For each snippet, write a CODING PROBLEM that would exercise that code path through the PUBLIC API.
Do NOT test internal functions directly — use create_sft_trainer, create_rl_trainer, EvalConfig, or run_eval
with specific parameters that would trigger the code path shown.

Source Code Snippets:
{json.dumps(selected, indent=2)}

{context}

Generate {n} problems. Each should:
1. Describe what to build and WHY (which code path it tests)
2. Specify exact model, dataset, algorithm, backend, and key parameters
3. Use max_steps=2, save_strategy="no", report_to="none"
4. Be DIFFERENT from each other — test different code paths

Output ONLY a JSON object: {{"problems": ["problem1 text", "problem2 text", ...]}}
"""

    return _call_llm_for_problems(prompt, n)


# ------------------------------------------------------------------
# Strategy 3: Mutation (Evol-Instruct-style)
# ------------------------------------------------------------------

MUTATION_OPS = [
    "ADD EVAL: Add an evaluation step after training using EvalConfig and run_eval",
    "CHANGE ALGORITHM: Switch from SFT to DPO (or DPO to GRPO, etc.)",
    "CHANGE BACKEND: Switch from trl to unsloth (only with compatible model)",
    "CHANGE MODEL: Use a different approved tiny model",
    "ADD CONSTRAINT: Add gradient_accumulation_steps=2 or a different optimizer",
    "COMBINE: Chain two training steps (e.g., SFT then DPO) in one script",
    "CHANGE LORA: Try different lora_r, lora_alpha values, or disable LoRA entirely",
    "CHANGE SCHEDULER: Use cosine or linear lr scheduler instead of constant",
    "ADD METRICS: Evaluate with different metric combinations",
    "CHANGE DATASET: Use a different approved dataset for the same algorithm",
]


def _synth_mutation(api_docs: str, n: int, quality_pool=None, rag_retriever=None, **kwargs) -> List[str]:
    """Evolve passing solutions into new problems."""
    context = _build_instructor_context(api_docs, rag_retriever)

    # Get seed problems from the quality pool
    seed_problems = []
    if quality_pool:
        try:
            for query in ["SFT training", "DPO training", "GRPO training", "evaluation", "LoRA config"]:
                shots = quality_pool.retrieve_similar(query, top_k=2)
                for s in shots:
                    if s.get("problem"):
                        seed_problems.append(s["problem"])
        except Exception:
            pass

    if not seed_problems:
        # No pool yet — use api_doc strategy instead
        return _synth_api_doc(api_docs, n, rag_retriever)

    # Deduplicate seeds
    seed_problems = list(dict.fromkeys(seed_problems))

    # Pick mutations
    mutations = [MUTATION_OPS[i % len(MUTATION_OPS)] for i in range(n)]

    prompt = f"""You are an ML instructor evolving existing coding problems into NEW, DIFFERENT variants.

EXISTING PROBLEMS (these were successfully solved):
{json.dumps(seed_problems[:10], indent=2)}

For each problem below, apply the specified MUTATION to create a new problem:
{json.dumps([{"seed": seed_problems[i % len(seed_problems)][:200], "mutation": mutations[i]} for i in range(n)], indent=2)}

{context}

Generate {n} NEW problems. Each must be SUBSTANTIALLY different from its seed.
The mutation should change the core behavior, not just rename a variable.

Output ONLY a JSON object: {{"problems": ["problem1 text", "problem2 text", ...]}}
"""

    return _call_llm_for_problems(prompt, n)


# ------------------------------------------------------------------
# Shared
# ------------------------------------------------------------------

def _call_llm_for_problems(prompt: str, n: int) -> List[str]:
    """Call LLM and parse response into problem strings."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,  # Higher for diversity
            max_tokens=4096,
        )
        content = json.loads(response.choices[0].message.content)
        problems = content.get("problems", list(content.values())[0])

        # Normalize
        normalized = []
        for p in problems:
            if isinstance(p, str) and len(p) > 20:  # Skip trivially short
                normalized.append(p)
            elif isinstance(p, dict):
                val = str(next(iter(p.values())))
                if len(val) > 20:
                    normalized.append(val)
        return normalized[:n]

    except Exception as e:
        print(f"    ⚠️ LLM parse failed: {e}")
        return []