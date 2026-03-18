import re
import ast
from dataclasses import dataclass, field, asdict
from typing import Optional, List

# ==========================================
# Multi-Dimensional Quality Scoring
# ==========================================

# Weights aligned with lib-agent's eval harness
SCORE_WEIGHTS = {
    "syntax": 0.10,
    "api_compliance": 0.20,
    "exec": 0.35,
    "outcome": 0.35,
}

# AlignTune API surface — imports and calls that count as "using the library"
ALIGNTUNE_API_PATTERNS = [
    r"from\s+aligntune",
    r"import\s+aligntune",
    r"create_sft_trainer",
    r"create_rl_trainer",
    r"EvalConfig",
    r"run_eval",
]

# Outcome patterns — evidence that training/eval actually ran
OUTCOME_PATTERNS = [
    r"(?:loss|final_loss)\s*[:=]\s*[\d.]+",
    r"(?:SFT|DPO|GRPO|PPO|Training|Eval)\s+OK",
    r"SUCCESS",
    r"completed\s+successfully",
    r"results?\s*[:=]\s*\{",
]


@dataclass
class QualityScore:
    """Multi-dimensional quality assessment of a generated code sample."""
    syntax_ok: bool = False
    api_ok: bool = False
    exec_ok: bool = False
    outcome_ok: bool = False

    syntax_score: float = 0.0
    api_score: float = 0.0
    exec_score: float = 0.0
    outcome_score: float = 0.0

    composite_score: float = 0.0

    # Metadata
    api_calls_found: List[str] = field(default_factory=list)
    outcome_signals: List[str] = field(default_factory=list)
    failure_category: str = ""  # "syntax", "import", "api", "runtime", "timeout", "outcome"

    def compute_composite(self):
        self.syntax_score = 1.0 if self.syntax_ok else 0.0
        self.api_score = 1.0 if self.api_ok else 0.0
        self.exec_score = 1.0 if self.exec_ok else 0.0
        self.outcome_score = 1.0 if self.outcome_ok else 0.0

        self.composite_score = (
            SCORE_WEIGHTS["syntax"] * self.syntax_score
            + SCORE_WEIGHTS["api_compliance"] * self.api_score
            + SCORE_WEIGHTS["exec"] * self.exec_score
            + SCORE_WEIGHTS["outcome"] * self.outcome_score
        )
        return self.composite_score


def score_attempt(code: str, success: bool, stdout: str, stderr: str) -> QualityScore:
    """
    Scores a single code attempt on 4 dimensions.
    Called after sandbox execution.
    """
    qs = QualityScore()

    # 1. Syntax — does the code parse?
    try:
        ast.parse(code)
        qs.syntax_ok = True
    except SyntaxError:
        qs.failure_category = "syntax"
        qs.compute_composite()
        return qs

    # 2. API compliance — does the code reference aligntune APIs?
    for pattern in ALIGNTUNE_API_PATTERNS:
        matches = re.findall(pattern, code)
        if matches:
            qs.api_calls_found.extend(matches)
    qs.api_ok = len(qs.api_calls_found) > 0

    if not qs.api_ok:
        qs.failure_category = "api"
        qs.compute_composite()
        return qs

    # 3. Execution — did it run without error?
    qs.exec_ok = success

    if not qs.exec_ok:
        # Classify failure type from stderr
        qs.failure_category = _classify_failure(stderr)
        qs.compute_composite()
        return qs

    # 4. Outcome — did it produce meaningful output?
    combined_output = stdout + stderr
    for pattern in OUTCOME_PATTERNS:
        matches = re.findall(pattern, combined_output, re.IGNORECASE)
        if matches:
            qs.outcome_signals.extend(matches)
    qs.outcome_ok = len(qs.outcome_signals) > 0

    if not qs.outcome_ok:
        qs.failure_category = "outcome"

    qs.compute_composite()
    return qs


def _classify_failure(stderr: str) -> str:
    """Classify a failed execution into a failure category."""
    stderr_lower = stderr.lower()
    if "syntaxerror" in stderr_lower:
        return "syntax"
    if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower:
        return "import"
    if "typeerror" in stderr_lower or "attributeerror" in stderr_lower:
        return "type"
    if "timed out" in stderr_lower or "timeout" in stderr_lower:
        return "timeout"
    if "assertionerror" in stderr_lower:
        return "assertion"
    return "runtime"