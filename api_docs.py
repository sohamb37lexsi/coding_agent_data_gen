ALIGNTUNE_API_DOCS = """
# AlignTune Library Reference
# pip install aligntune

AlignTune is a production-ready fine-tuning library for LLMs supporting SFT and RL methods 
(DPO, PPO, GRPO, GSPO, DAPO, Dr. GRPO) with TRL and Unsloth backends.


================================================================================
## VALIDATION CONSTRAINTS (MUST FOLLOW)
================================================================================

When writing scripts, ALWAYS use these settings to keep execution fast and lightweight:

### Approved Tiny Models (use ONLY these):
- "sshleifer/tiny-gpt2"           — General purpose, ~600KB
- "hf-internal-testing/tiny-random-LlamaForCausalLM"  — Llama-arch testing

### Approved Small Datasets (use ONLY these):
- "tatsu-lab/alpaca"              — SFT instruction data (split="train[:50]")
- "Anthropic/hh-rlhf"            — DPO preference data (split="train[:50]")
- "openai/gsm8k"                 — Math/reasoning for GRPO (split="train[:50]")

### Required Hyperparameter Constraints:
- max_steps=2                     — ALWAYS set to 2 (just verify training runs)
- batch_size=2                    — Keep small
- max_seq_length=128              — Short sequences
- gradient_accumulation_steps=1
- logging_steps=1
- save_strategy="no"              — Don't save checkpoints during validation
- report_to="none"                — No external logging

### Dataset Splits:
- ALWAYS limit data with split="train[:50]" (50 examples max)
- For eval: split="train[50:60]" (10 examples)

The goal is to verify the code INITIALIZES correctly, runs 1-2 training steps 
without error, and exits cleanly. NOT to produce a trained model.


================================================================================
## Factory Functions (aligntune.core.backend_factory)
================================================================================

### create_sft_trainer(...)
Creates a Supervised Fine-Tuning trainer.

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    # Required
    model_name: str,              # HF model ID (use approved tiny models)
    dataset_name: str,            # HF dataset ID (use approved datasets)
    backend: str,                 # "trl" or "unsloth"

    # Training
    num_epochs: int = 3,
    max_steps: int = 2,           # Use 2 for validation
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "constant",
    max_grad_norm: float = 0.3,
    optim: str = "adamw_torch",

    # Data
    max_seq_length: int = 128,
    max_samples: int = None,
    dataset_text_field: str = "messages",
    split: str = "train[:50]",
    task_type: str = "instruction_following",

    # LoRA
    use_peft: bool = True,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "all-linear",

    # Quantization (optional, set None to skip)
    quantization: dict = None,    # Skip for tiny models

    # Precision
    bf16: bool = False,           # Tiny models work fine in fp32
    tf32: bool = False,
    gradient_checkpointing: bool = False,

    # Output
    output_dir: str = "./output",
    save_strategy: str = "no",
    logging_steps: int = 1,
    report_to: str = "none",
    seed: int = 42,
    trust_remote_code: bool = True,
)
```

Returns: A trainer object.
- trainer.train() -> dict with keys: "model_path", "final_loss", "training_time", "total_steps"
- trainer.evaluate() -> dict with metric scores


### create_rl_trainer(...)
Creates a Reinforcement Learning trainer (DPO, PPO, GRPO, GSPO, DAPO, Dr. GRPO).

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    # Required
    model_name: str,              # HF model ID (use approved tiny models)
    dataset_name: str,            # HF dataset ID (must have chosen/rejected for DPO)
    algorithm: str,               # "dpo", "ppo", "grpo", "gspo", "dapo", "drgrpo"
    backend: str,                 # "trl" or "unsloth"

    # Training
    num_epochs: int = 1,
    max_steps: int = 2,           # Use 2 for validation
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    lr_scheduler_type: str = "constant",
    optim: str = "adamw_torch",

    # Data
    split: str = "train[:50]",
    max_seq_length: int = 128,
    max_prompt_length: int = 64,
    max_completion_length: int = 64,

    # LoRA
    use_lora: bool = True,
    lora_config: dict = {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Quantization (skip for tiny models)
    quantization: dict = None,
    precision: str = "fp32",

    # Algorithm-specific
    beta: float = 0.1,           # DPO temperature parameter

    # Output
    output_dir: str = "./output",
    save_steps: int = 999,       # Effectively never save during validation
    logging_steps: int = 1,
    report_to: str = "none",
    seed: int = 42,
)
```

Returns: A trainer object.
- trainer.train() -> dict with keys: "model_path", "final_loss", "training_time", "total_steps"
- trainer.dataset_dict -> the loaded dataset splits


================================================================================
## Evaluation (aligntune.eval.runner)
================================================================================

```python
from aligntune.eval.runner import EvalConfig, run_eval

config = EvalConfig(
    model_path: str,              # Path to trained model or HF model ID
    output_dir: str,

    # Task
    task_type: str,               # "dpo" or "sft"
    data_task_type: str,          # "dpo" or "sft"

    # Metrics
    metrics: list,                # SFT: ["bleu", "rouge"]  DPO: ["reward_margin", "preference_accuracy", "win_rate"]

    # Data
    dataset_name: str,
    split: str = "train[50:60]",  # Small eval slice
    max_samples: int = 10,

    # Model
    device: str = "cuda",
    batch_size: int = 2,
    max_length: int = 128,
    temperature: float = 0.0,
    trust_remote_code: bool = True,

    # LoRA models
    use_lora: bool = False,
    base_model: str = None,

    # DPO eval
    reference_model_path: str = None,

    # Backend
    use_unsloth: bool = False,
    system_prompt: str = None,
    seed: int = 42,
)

results = run_eval(config)                 # For SFT
results = run_eval(config, dataset_dict)   # For DPO (pass trainer.dataset_dict)
```

Returns: dict with metric scores.


================================================================================
## Supported Algorithms
================================================================================

| Algorithm | Key       | Backends        |
|-----------|-----------|-----------------|
| SFT       | (factory) | trl, unsloth    |
| DPO       | "dpo"     | trl, unsloth    |
| PPO       | "ppo"     | trl, unsloth    |
| GRPO      | "grpo"    | trl, unsloth    |
| GSPO      | "gspo"    | trl, unsloth    |
| DAPO      | "dapo"    | trl, unsloth    |
| Dr. GRPO  | "drgrpo"  | trl, unsloth    |


================================================================================
## Complete Working Examples (Smoke-Test Style)
================================================================================

### Example 1: SFT — Verify trainer creates and runs 2 steps
```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="sshleifer/tiny-gpt2",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    max_steps=2,
    batch_size=2,
    learning_rate=5e-5,
    max_seq_length=128,
    split="train[:50]",
    output_dir="./output/sft_test",
    save_strategy="no",
    report_to="none",
    seed=42,
)
result = trainer.train()
print(f"SFT OK — loss: {result.get('final_loss', 'N/A')}")
```

### Example 2: DPO — Verify preference training initializes
```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="sshleifer/tiny-gpt2",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="dpo",
    backend="trl",
    max_steps=2,
    batch_size=2,
    learning_rate=5e-5,
    max_seq_length=128,
    max_prompt_length=64,
    max_completion_length=64,
    split="train[:50]",
    beta=0.1,
    output_dir="./output/dpo_test",
    logging_steps=1,
    report_to="none",
    seed=42,
)
result = trainer.train()
print(f"DPO OK — loss: {result.get('final_loss', 'N/A')}")
```

### Example 3: GRPO — Verify group policy optimization initializes
```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="sshleifer/tiny-gpt2",
    dataset_name="openai/gsm8k",
    algorithm="grpo",
    backend="trl",
    max_steps=2,
    batch_size=2,
    learning_rate=5e-5,
    max_seq_length=128,
    split="train[:50]",
    output_dir="./output/grpo_test",
    logging_steps=1,
    report_to="none",
    seed=42,
)
result = trainer.train()
print(f"GRPO OK — loss: {result.get('final_loss', 'N/A')}")
```

### Example 4: SFT + Evaluate — Verify end-to-end train-then-eval
```python
from aligntune.core.backend_factory import create_sft_trainer
from aligntune.eval.runner import EvalConfig, run_eval

# Train
trainer = create_sft_trainer(
    model_name="sshleifer/tiny-gpt2",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    max_steps=2,
    batch_size=2,
    max_seq_length=128,
    split="train[:50]",
    output_dir="./output/sft_eval_test",
    save_strategy="no",
    report_to="none",
)
train_result = trainer.train()
print(f"Training OK — loss: {train_result.get('final_loss', 'N/A')}")

# Evaluate
eval_config = EvalConfig(
    model_path="sshleifer/tiny-gpt2",
    output_dir="./eval_results",
    data_task_type="sft",
    dataset_name="tatsu-lab/alpaca",
    split="train[50:60]",
    device="cuda",
    batch_size=2,
    max_length=128,
    max_samples=10,
    metrics=["bleu", "rouge"],
)
eval_result = run_eval(eval_config)
print(f"Eval OK — results: {eval_result}")
```
"""