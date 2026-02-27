import json
from typing import List
from config import client, MODEL_NAME

# ==========================================
# Phase 1: Problem Augmentation (Smoke-Test)
# ==========================================

# Explicit problem categories to ensure coverage across the library surface
PROBLEM_TIERS = [
    "basic_sft",         # Single create_sft_trainer call, train 2 steps
    "basic_dpo",         # Single create_rl_trainer with algorithm="dpo", train 2 steps
    "basic_grpo",        # Single create_rl_trainer with algorithm="grpo", train 2 steps
    "sft_with_lora",     # SFT with explicit LoRA config
    "sft_then_eval",     # Train SFT then run EvalConfig + run_eval
    "dpo_then_eval",     # Train DPO then run EvalConfig + run_eval
    "backend_comparison", # Create same trainer with backend="trl" vs "unsloth"
    "multi_algorithm",   # Train with two different RL algorithms sequentially
]

def generate_problems(api_docs: str, num_problems: int = 5) -> List[str]:
    """Generates K diverse smoke-test coding problems for aligntune."""
    print(f"--- Phase 1: Generating {num_problems} problems ---")

    prompt = f"""You are an expert ML instructor creating VALIDATION TEST problems for the `aligntune` library.

API Documentation:
{api_docs}

CRITICAL RULES:
1. Every problem MUST use tiny models: "sshleifer/tiny-gpt2" or "hf-internal-testing/tiny-random-LlamaForCausalLM"
2. Every problem MUST set max_steps=2 — we are ONLY verifying the code runs, not training real models
3. Every problem MUST use small dataset slices: split="train[:50]"
4. Every problem MUST use save_strategy="no" and report_to="none"
5. Every problem MUST end with a print() statement confirming success
6. Do NOT use quantization (set quantization=None or omit it) — tiny models don't need it
7. Use backend="trl" unless the problem specifically tests unsloth

Generate {num_problems} diverse problems. Each should test a DIFFERENT aspect of the library:
- Some should test create_sft_trainer only
- Some should test create_rl_trainer with different algorithms (dpo, grpo, ppo)
- Some should test the evaluation pipeline (EvalConfig + run_eval)
- Some should combine training + evaluation
- At least one should compare backends

Each problem prompt should clearly state what the user needs to build and which library features to use.
Output ONLY a JSON object with a "problems" key containing a list of problem strings.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,  # Higher for diversity
    )

    try:
        content = json.loads(response.choices[0].message.content)
        problems = content.get("problems", list(content.values())[0])
        return problems[:num_problems]
    except Exception as e:
        print(f"Error parsing generated problems: {e}")
        # Fallback: return hardcoded smoke-test problems
        return _fallback_problems()[:num_problems]


def _fallback_problems() -> List[str]:
    """Hardcoded problems if LLM generation fails."""
    return [
        'Write a Python script that uses aligntune to create an SFT trainer with model "sshleifer/tiny-gpt2" on the "tatsu-lab/alpaca" dataset using the TRL backend. Run 2 training steps and print the final loss.',

        'Write a Python script that uses aligntune to run DPO training with model "sshleifer/tiny-gpt2" on "Anthropic/hh-rlhf" using create_rl_trainer. Use algorithm="dpo", beta=0.1, max_steps=2, and print confirmation that training completed.',

        'Write a Python script that uses aligntune to create a GRPO trainer with model "sshleifer/tiny-gpt2" on "openai/gsm8k". Use create_rl_trainer with algorithm="grpo", run 2 steps, and print the result.',

        'Write a Python script that trains an SFT model using aligntune with "sshleifer/tiny-gpt2" on "tatsu-lab/alpaca" for 2 steps, then evaluates it using EvalConfig and run_eval with metrics=["bleu", "rouge"]. Print both training and evaluation results.',

        'Write a Python script that uses aligntune to create two SFT trainers for "sshleifer/tiny-gpt2" on "tatsu-lab/alpaca" — one with backend="trl" and one with backend="unsloth". Run 2 steps each and print both results to compare.',
    ]