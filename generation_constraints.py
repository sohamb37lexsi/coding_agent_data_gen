import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# ==========================================
# Generation Constraints & Model Rotation
# ==========================================

# ------------------------------------------------------------------
# 1. Model Pool & Rotation
# ------------------------------------------------------------------

@dataclass
class ModelSpec:
    """A tiny model approved for smoke-testing."""
    model_id: str
    arch: str                          # "gpt2", "llama", "qwen2", "phi"
    lora_target_modules: str           # What works for this architecture
    supports_backends: List[str]       # ["trl"] or ["trl", "unsloth"]
    notes: str = ""

# Approved tiny models — every model here is <500MB and runs in seconds
MODEL_POOL: List[ModelSpec] = [
    # GPT-2 family
    ModelSpec(
        model_id="sshleifer/tiny-gpt2",
        arch="gpt2",
        lora_target_modules="all-linear",  # GPT-2 uses c_attn/c_proj, not q_proj/v_proj
        supports_backends=["trl"],          # Unsloth has compiled cache bugs for GPT-2
        notes="GPT-2 arch uses c_attn/c_proj. Do NOT use q_proj/v_proj.",
    ),
    # Llama family
    ModelSpec(
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        arch="llama",
        lora_target_modules="all-linear",
        supports_backends=["trl", "unsloth"],
        notes="Llama arch supports standard q_proj/v_proj but all-linear is safest.",
    ),
    # Qwen2 family (if available in your env)
    ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        arch="qwen2",
        lora_target_modules="all-linear",
        supports_backends=["trl", "unsloth"],
        notes="Qwen2 arch. Requires trust_remote_code=True.",
    ),
]

# Dataset pool with metadata about what they're suitable for
@dataclass
class DatasetSpec:
    dataset_id: str
    suitable_for: List[str]   # ["sft", "dpo", "grpo", ...]
    split: str = "train[:50]"
    eval_split: str = "train[50:60]"

DATASET_POOL: List[DatasetSpec] = [
    DatasetSpec("tatsu-lab/alpaca", suitable_for=["sft"]),
    DatasetSpec("Anthropic/hh-rlhf", suitable_for=["dpo"]),
    DatasetSpec("openai/gsm8k", suitable_for=["grpo", "gspo", "dapo", "drgrpo"]),
    DatasetSpec("HuggingFaceH4/ultrachat_200k", suitable_for=["ppo", "sft"]),
]


class ModelRotator:
    """Rotates through approved models when one fails due to architecture issues."""

    # Error patterns that indicate a model-specific issue (not a code bug)
    MODEL_ERROR_PATTERNS = [
        r"Target modules .+ not found in the base model",
        r"does not have .+ attribute",
        r"unsloth_compiled_module.*Error",
        r"Direct module loading failed",
        r"not supported for .+ architecture",
        r"KeyError:.*attention",
        r"Expected .+ but got .+",  # Shape mismatches
        r"CUDA out of memory",
    ]

    def __init__(self):
        self._failed_models: Dict[str, List[str]] = {}  # problem_hash -> [model_ids that failed]

    def is_model_error(self, traceback_text: str) -> bool:
        """Returns True if the traceback indicates a model-specific issue."""
        for pattern in self.MODEL_ERROR_PATTERNS:
            if re.search(pattern, traceback_text, re.IGNORECASE):
                return True
        return False

    def get_model(
        self,
        task_type: str,
        backend: str,
        problem_hash: str,
        preferred_model: Optional[str] = None,
    ) -> Optional[ModelSpec]:
        """
        Returns the next model to try, skipping ones that already failed
        for this problem.
        """
        failed = self._failed_models.get(problem_hash, [])

        candidates = [
            m for m in MODEL_POOL
            if m.model_id not in failed
            and backend in m.supports_backends
        ]

        if not candidates:
            # All models exhausted for this backend, try other backend's models
            other_backend = "unsloth" if backend == "trl" else "trl"
            candidates = [
                m for m in MODEL_POOL
                if m.model_id not in failed
                and other_backend in m.supports_backends
            ]

        if not candidates:
            return None

        # Prefer the requested model if it hasn't failed
        if preferred_model:
            for c in candidates:
                if c.model_id == preferred_model:
                    return c

        return candidates[0]

    def mark_failed(self, problem_hash: str, model_id: str):
        """Record that a model failed for this specific problem."""
        if problem_hash not in self._failed_models:
            self._failed_models[problem_hash] = []
        if model_id not in self._failed_models[problem_hash]:
            self._failed_models[problem_hash].append(model_id)
            print(f"  🔄 Marked {model_id} as failed, will try next model")

    def get_dataset(self, task_type: str) -> Optional[DatasetSpec]:
        """Returns an appropriate dataset for the task type."""
        candidates = [d for d in DATASET_POOL if task_type in d.suitable_for]
        return candidates[0] if candidates else None


# ------------------------------------------------------------------
# 2. Parameter Constraints (Hard Boundaries)
# ------------------------------------------------------------------

# These define the ONLY acceptable values for each parameter.
# The LLM prompt will include these, and post-generation validation will enforce them.

SFT_PARAM_CONSTRAINTS: Dict[str, Any] = {
    "model_name": {
        "type": "enum",
        "values": [m.model_id for m in MODEL_POOL],
        "description": "Must be an approved tiny model from the pool",
    },
    "dataset_name": {
        "type": "enum",
        "values": ["tatsu-lab/alpaca"],
        "description": "SFT dataset",
    },
    "backend": {
        "type": "enum",
        "values": ["trl", "unsloth"],
    },
    "max_steps": {
        "type": "int",
        "min": 1, "max": 5,
        "default": 2,
        "description": "Smoke test only — keep between 1-5",
    },
    "batch_size": {
        "type": "int",
        "min": 1, "max": 4,
        "default": 2,
    },
    "learning_rate": {
        "type": "float",
        "min": 1e-6, "max": 1e-3,
        "default": 5e-5,
    },
    "max_seq_length": {
        "type": "int",
        "min": 32, "max": 256,
        "default": 128,
    },
    "split": {
        "type": "enum",
        "values": ["train[:50]", "train[:100]"],
        "default": "train[:50]",
    },
    "save_strategy": {
        "type": "enum",
        "values": ["no"],
        "description": "Never save during validation",
    },
    "report_to": {
        "type": "enum",
        "values": ["none"],
    },
    "logging_steps": {
        "type": "int",
        "min": 1, "max": 5,
        "default": 1,
    },
    "seed": {
        "type": "int",
        "min": 0, "max": 9999,
        "default": 42,
    },
    "lora_target_modules": {
        "type": "enum",
        "values": ["all-linear"],
        "description": "ALWAYS use all-linear. Never specify q_proj/v_proj/etc — they differ per architecture.",
    },
    "quantization": {
        "type": "enum",
        "values": ["None"],
        "description": "Never use quantization for tiny models",
    },
    "bf16": {
        "type": "enum",
        "values": [False],
        "description": "Tiny models work in fp32",
    },
    "gradient_checkpointing": {
        "type": "enum",
        "values": [False],
    },
}

RL_PARAM_CONSTRAINTS: Dict[str, Any] = {
    "model_name": {
        "type": "enum",
        "values": [m.model_id for m in MODEL_POOL],
    },
    "algorithm": {
        "type": "enum",
        "values": ["dpo", "ppo", "grpo", "gspo", "dapo", "drgrpo"],
    },
    "backend": {
        "type": "enum",
        "values": ["trl", "unsloth"],
    },
    "max_steps": {
        "type": "int",
        "min": 1, "max": 5,
        "default": 2,
    },
    "batch_size": {
        "type": "int",
        "min": 1, "max": 4,
        "default": 2,
    },
    "learning_rate": {
        "type": "float",
        "min": 1e-6, "max": 1e-3,
        "default": 5e-5,
    },
    "max_seq_length": {
        "type": "int",
        "min": 32, "max": 256,
        "default": 128,
    },
    "max_prompt_length": {
        "type": "int",
        "min": 16, "max": 128,
        "default": 64,
    },
    "max_completion_length": {
        "type": "int",
        "min": 16, "max": 128,
        "default": 64,
    },
    "split": {
        "type": "enum",
        "values": ["train[:50]", "train[:100]"],
        "default": "train[:50]",
    },
    "beta": {
        "type": "float",
        "min": 0.01, "max": 1.0,
        "default": 0.1,
        "description": "DPO temperature",
    },
    "report_to": {
        "type": "enum",
        "values": ["none"],
    },
    "quantization": {
        "type": "enum",
        "values": ["None"],
    },
    "lora_target_modules": {
        "type": "note",
        "description": "Use all-linear via use_lora=True. Do NOT pass explicit target_modules.",
    },
}

EVAL_PARAM_CONSTRAINTS: Dict[str, Any] = {
    "max_samples": {
        "type": "int",
        "min": 5, "max": 20,
        "default": 10,
    },
    "max_length": {
        "type": "int",
        "min": 32, "max": 256,
        "default": 128,
    },
    "batch_size": {
        "type": "int",
        "min": 1, "max": 4,
        "default": 2,
    },
    "temperature": {
        "type": "float",
        "min": 0.0, "max": 0.0,
        "default": 0.0,
        "description": "Always greedy for eval",
    },
    "metrics": {
        "type": "enum_list",
        "sft": ["bleu", "rouge"],
        "dpo": ["reward_margin", "preference_accuracy", "win_rate"],
    },
    "split": {
        "type": "enum",
        "values": ["train[50:60]", "test"],
        "default": "train[50:60]",
    },
}


def format_constraints_for_prompt() -> str:
    """Builds a human-readable constraints block to inject into LLM prompts."""
    lines = [
        "=" * 60,
        "HARD CONSTRAINTS — VIOLATING THESE WILL CAUSE EXECUTION FAILURE",
        "=" * 60,
        "",
        "APPROVED MODELS (use ONLY these):",
    ]
    for m in MODEL_POOL:
        backends = ", ".join(m.supports_backends)
        lines.append(f'  - "{m.model_id}" (arch: {m.arch}, backends: {backends})')
        lines.append(f'    LoRA: use lora_target_modules="all-linear"')
        if m.notes:
            lines.append(f"    Note: {m.notes}")

    lines.append("")
    lines.append("APPROVED DATASETS:")
    for d in DATASET_POOL:
        tasks = ", ".join(d.suitable_for)
        lines.append(f'  - "{d.dataset_id}" (for: {tasks}, split: "{d.split}")')

    lines.append("")
    lines.append("REQUIRED PARAMETER VALUES:")
    lines.append("  - max_steps = 2 (validation only, never higher)")
    lines.append("  - batch_size = 2")
    lines.append("  - max_seq_length = 128")
    lines.append("  - save_strategy = \"no\"")
    lines.append("  - report_to = \"none\"")
    lines.append("  - quantization = None (skip for tiny models)")
    lines.append("  - bf16 = False (tiny models use fp32)")
    lines.append('  - lora_target_modules = "all-linear" (NEVER use q_proj/v_proj/etc)')
    lines.append("  - split = \"train[:50]\"")
    lines.append("")
    lines.append("FORBIDDEN ACTIONS:")
    lines.append("  - Do NOT install, upgrade, or reinstall any packages")
    lines.append("  - Do NOT use subprocess to run pip")
    lines.append("  - Do NOT use quantization dicts (no load_in_4bit)")
    lines.append("  - Do NOT specify explicit LoRA target modules per architecture")
    lines.append("  - Do NOT use !pip or %magic commands")
    lines.append("  - Do NOT attempt to download or save models to disk")
    lines.append("")
    lines.append("EVERY SCRIPT MUST:")
    lines.append("  - End with a print() confirming success")
    lines.append("  - Actually call trainer.train() to trigger execution")
    lines.append("  - Import only from aligntune.core.backend_factory or aligntune.eval.runner")

    return "\n".join(lines)


# ------------------------------------------------------------------
# 3. Code Validator (Post-Generation Check)
# ------------------------------------------------------------------

class CodeValidator:
    """Validates generated code against hard constraints before sandbox execution."""

    VIOLATION_PATTERNS = [
        (r'load_in_4bit.*True', "Quantization not allowed for tiny models"),
        (r'bnb_4bit', "Quantization not allowed for tiny models"),
        (r'target_modules.*\[.*"q_proj"', 'Do not use explicit target_modules — use "all-linear"'),
        (r'target_modules.*\[.*"v_proj"', 'Do not use explicit target_modules — use "all-linear"'),
        (r'subprocess\.run', "subprocess calls not allowed"),
        (r'pip install', "pip install not allowed"),
        (r'max_steps\s*=\s*(\d+)', None),  # Checked programmatically below
        (r'bf16\s*=\s*True', "bf16=True not allowed for tiny models"),
    ]

    ALLOWED_MODELS = {m.model_id for m in MODEL_POOL}

    @classmethod
    def validate(cls, code: str) -> Tuple[bool, List[str]]:
        """
        Returns (is_valid, list_of_violations).
        Violations are warnings — the sandbox is the final judge, but this
        catches obvious constraint violations before wasting a sandbox run.
        """
        violations = []

        for pattern, msg in cls.VIOLATION_PATTERNS:
            match = re.search(pattern, code)
            if match:
                if pattern == r'max_steps\s*=\s*(\d+)':
                    val = int(match.group(1))
                    if val > 5 and val != -1:
                        violations.append(f"max_steps={val} exceeds limit of 5")
                elif msg:
                    violations.append(msg)

        # Check model is in approved list
        model_match = re.search(r'model_name\s*=\s*"([^"]+)"', code)
        if model_match:
            model = model_match.group(1)
            if model not in cls.ALLOWED_MODELS:
                violations.append(f'Model "{model}" not in approved list: {cls.ALLOWED_MODELS}')

        return len(violations) == 0, violations


# ------------------------------------------------------------------
# 4. Prompt Hardening Templates
# ------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are a code generation assistant for the `aligntune` ML library.
You write SHORT, CORRECT smoke-test scripts that verify library functionality.

RULES YOU MUST NEVER BREAK:
1. Use ONLY approved tiny models — NEVER use large models
2. Use lora_target_modules="all-linear" — NEVER specify per-architecture modules
3. Set max_steps=2 — we are testing, not training
4. Do NOT install packages — everything is pre-installed
5. Do NOT use quantization — tiny models don't need it
6. Every script must call .train() and print results

If an error comes from a library internal (trl, unsloth), do NOT try to fix the library.
Instead: switch to a different model, a different backend, or different parameters.
"""

SOLVER_SYSTEM_PROMPT = """You translate plans into Python scripts for the `aligntune` library.
Output ONLY code inside ```python ``` blocks. No explanations.

CRITICAL: Your code must follow these exact patterns. Do not deviate.
- Import from: aligntune.core.backend_factory (create_sft_trainer, create_rl_trainer)
- Import from: aligntune.eval.runner (EvalConfig, run_eval)
- Always use: max_steps=2, batch_size=2, max_seq_length=128
- Always use: save_strategy="no", report_to="none"
- Always use: lora_target_modules="all-linear" (for SFT) or omit target_modules entirely (for RL)
- Always end with: print("SUCCESS: <description>")
"""