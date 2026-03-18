import re
import hashlib
from typing import Optional, List, Dict
from dataclasses import asdict

from config import client, MODEL_NAME
from data_types import TurnAttempt, TaskTrajectory
from sandbox import CodeSandbox
from quality_scorer import score_attempt, QualityScore
from quality_pool import QualityPool, PoolRecord
from generation_constraints import (
    ModelRotator,
    CodeValidator,
    format_constraints_for_prompt,
    PLANNER_SYSTEM_PROMPT,
    SOLVER_SYSTEM_PROMPT,
)

# ==========================================
# Phase 2: Reflexion + Quality Pool + Scoring
# ==========================================

_retriever = None
_rotator = ModelRotator()
_pool = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from rag_retriever import RAGRetriever
        _retriever = RAGRetriever()
    return _retriever


def get_pool() -> QualityPool:
    global _pool
    if _pool is None:
        _pool = QualityPool()
    return _pool


# ------------------------------------------------------------------
# Reflexion: Accumulated verbal reflections across turns
# ------------------------------------------------------------------

class ReflexionMemory:
    """
    Maintains a list of verbal reflections across turns for a single problem.
    Each reflection summarizes what went wrong and what to try differently.
    Injected into every subsequent Planner prompt so the model never
    repeats the same failing approach.
    """

    def __init__(self):
        self.reflections: List[str] = []

    def add_reflection(self, turn: int, traceback: str, code: str, score: QualityScore):
        """Ask the LLM to reflect on the failure, then store it."""
        reflection = _generate_reflection(turn, traceback, code, score)
        if reflection:
            self.reflections.append(f"[Turn {turn}] {reflection}")

    def format_for_prompt(self) -> str:
        if not self.reflections:
            return ""
        lines = [
            "\n" + "=" * 60,
            "REFLECTIONS FROM PREVIOUS ATTEMPTS (do NOT repeat these mistakes):",
            "=" * 60,
        ]
        for r in self.reflections:
            lines.append(f"  • {r}")
        lines.append("")
        return "\n".join(lines)


def _generate_reflection(turn: int, traceback: str, code: str, score: QualityScore) -> Optional[str]:
    """LLM generates a verbal reflection on what went wrong."""
    prompt = f"""You just wrote code that failed. Reflect on what went wrong in 2-3 sentences.

Failure category: {score.failure_category}
Quality: syntax={'✓' if score.syntax_ok else '✗'}, api={'✓' if score.api_ok else '✗'}, exec={'✓' if score.exec_ok else '✗'}

Traceback (last 500 chars):
{traceback[-500:]}

Code snippet (first 500 chars):
{code[:500]}

Write a SHORT reflection: what specific mistake was made, and what concrete change to make next time.
Do NOT write code. Just 2-3 sentences."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()[:300]
    except Exception:
        return None


# ------------------------------------------------------------------
# Planner and Solver (enhanced with few-shots + reflections)
# ------------------------------------------------------------------

def generate_plan(
    problem: str,
    api_context: str,
    constraints: str,
    few_shot_block: str = "",
    reflexion_block: str = "",
    previous_error: Optional[str] = None,
    error_source_context: Optional[str] = None,
    model_switch_note: Optional[str] = None,
) -> str:
    """Planner: generates a plan with few-shot examples and accumulated reflections."""

    prompt = f"Problem: {problem}\n\nAPI Docs:\n{api_context}\n\n{constraints}\n"

    # Inject few-shot examples from quality pool
    if few_shot_block:
        prompt += few_shot_block + "\n"

    # Inject accumulated reflections from previous turns
    if reflexion_block:
        prompt += reflexion_block + "\n"

    if model_switch_note:
        prompt += f"\n⚠️ MODEL SWITCH: {model_switch_note}\n\n"

    if previous_error:
        prompt += "\nPREVIOUS EXECUTION FAILED.\n\n"
        if error_source_context:
            prompt += f"{error_source_context}\n\n"
        else:
            prompt += f"TRACEBACK:\n{previous_error}\n\n"
        prompt += (
            "Analyze the error. If it's from a library internal (trl/unsloth/peft), "
            "do NOT fix the library — switch model, backend, or simplify params.\n"
            "Write a revised step-by-step plan."
        )
    else:
        prompt += "\nWrite a step-by-step plan. Follow the constraints and examples EXACTLY."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def generate_code(
    problem: str,
    plan: str,
    api_context: str,
    constraints: str,
    few_shot_block: str = "",
    previous_code: Optional[str] = None,
    previous_traceback: Optional[str] = None,
) -> str:
    """Solver: translates plan to code. Optionally sees previous failed code."""

    prompt = f"Problem: {problem}\n\nAPI Docs:\n{api_context}\n\n{constraints}\n"

    if few_shot_block:
        prompt += few_shot_block + "\n"

    prompt += f"\nPlan:\n{plan}\n\n"

    # Show the solver what it wrote last time (so it can make targeted fixes)
    if previous_code and previous_traceback:
        prompt += f"""YOUR PREVIOUS CODE (which failed):
```python
{previous_code[:1500]}
```

ERROR:
{previous_traceback[:500]}

Fix the specific issue above. Do NOT rewrite from scratch unless necessary.
"""

    prompt += """
Output ONLY the python code inside ```python ``` blocks.
MUST end with print("SUCCESS: ...") confirming the script ran."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return CodeSandbox.extract_code_blocks(response.choices[0].message.content)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def run_react_loop(
    problem: str,
    api_docs_fallback: str,
    max_turns: int = 3,
    use_rag: bool = True,
) -> TaskTrajectory:
    """
    Phase 2+3 loop with:
    - RAG context retrieval
    - Quality pool few-shot injection
    - Multi-dimensional scoring
    - Reflexion-style accumulated reflections
    - Model rotation on architecture errors
    - Pre-execution code validation
    - Solver sees previous failed code
    """
    problem_str = str(problem)
    problem_hash = hashlib.md5(problem_str.encode()).hexdigest()
    print(f"\n--- Running Task: {problem_str[:80]}... ---")

    sandbox = CodeSandbox()
    constraints = format_constraints_for_prompt()
    reflexion = ReflexionMemory()
    pool = get_pool()

    # Retrieve API context (RAG or static)
    if use_rag:
        try:
            retriever = get_retriever()
            api_context = retriever.retrieve_api_context(problem_str, top_k=5)
            print(f"  📚 RAG: {len(api_context)} chars of API context")
        except Exception as e:
            print(f"  ⚠️ RAG failed ({e}), using static docs")
            api_context = api_docs_fallback
    else:
        api_context = api_docs_fallback

    # Retrieve few-shot examples from quality pool
    few_shot_block = pool.format_few_shots(problem_str, top_k=2)
    if few_shot_block:
        print(f"  📝 Injected {few_shot_block.count('Example')} few-shot examples from quality pool")

    trajectory = TaskTrajectory(
        problem=problem_str,
        api_context=api_context,
        attempts=[],
        is_successful=False,
    )

    current_traceback = None
    error_source_context = None
    model_switch_note = None
    previous_code = None
    previous_traceback_for_solver = None
    best_score = None
    best_attempt = None

    for turn in range(max_turns):
        print(f"  Turn {turn + 1}/{max_turns}...")

        # --- Generate ---
        plan = generate_plan(
            problem_str, api_context, constraints,
            few_shot_block, reflexion.format_for_prompt(),
            current_traceback, error_source_context, model_switch_note,
        )
        code = generate_code(
            problem_str, plan, api_context, constraints,
            few_shot_block, previous_code, previous_traceback_for_solver,
        )

        # --- Pre-execution validation ---
        is_valid, violations = CodeValidator.validate(code)
        if not is_valid:
            print(f"  ⚠️ Validation: {violations}")

        # --- Execute ---
        success, stdout, stderr = sandbox.execute(code)

        # --- Score ---
        score = score_attempt(code, success, stdout, stderr)
        print(f"  📊 Score: {score.composite_score:.2f} "
              f"(syntax={'✓' if score.syntax_ok else '✗'} "
              f"api={'✓' if score.api_ok else '✗'} "
              f"exec={'✓' if score.exec_ok else '✗'} "
              f"outcome={'✓' if score.outcome_ok else '✗'})")

        # Track best attempt (even partial successes are useful for DPO)
        if best_score is None or score.composite_score > best_score.composite_score:
            best_score = score
            best_attempt = (plan, code, score)

        attempt = TurnAttempt(
            turn_number=turn + 1,
            plan=plan,
            code=code,
            traceback=stderr if not success else None,
            success=success,
        )
        trajectory.attempts.append(attempt)

        if success and score.outcome_ok:
            print(f"  ✅ Full pass! (score: {score.composite_score:.2f})")
            trajectory.is_successful = True

            # Add to quality pool for future few-shot injection
            _add_to_pool(problem_str, plan, code, score, turn + 1)
            break

        if success and not score.outcome_ok:
            print(f"  ⚠️ Executed but no outcome signal — partial pass")
            # Still a useful record, add with lower score
            trajectory.is_successful = True
            _add_to_pool(problem_str, plan, code, score, turn + 1)
            break

        # --- Failure handling ---
        raw_traceback = stderr[-1500:]
        current_traceback = raw_traceback
        previous_code = code
        previous_traceback_for_solver = raw_traceback
        model_switch_note = None

        # Reflexion: generate and accumulate a verbal reflection
        reflexion.add_reflection(turn + 1, raw_traceback, code, score)

        # Log failure for curriculum tracking
        pool.log_failure(problem_str, code, score.failure_category, raw_traceback)

        # Model rotation on architecture errors
        if _rotator.is_model_error(raw_traceback):
            model_match = re.search(r'model_name\s*=\s*"([^"]+)"', code)
            if model_match:
                failed_model = model_match.group(1)
                _rotator.mark_failed(problem_hash, failed_model)

                task_type = "sft"
                if "create_rl_trainer" in code:
                    algo_match = re.search(r'algorithm\s*=\s*"([^"]+)"', code)
                    task_type = algo_match.group(1) if algo_match else "dpo"

                backend = "trl"
                backend_match = re.search(r'backend\s*=\s*"([^"]+)"', code)
                if backend_match:
                    backend = backend_match.group(1)

                next_model = _rotator.get_model(task_type, backend, problem_hash)
                if next_model:
                    model_switch_note = (
                        f'Model "{failed_model}" failed. '
                        f'Switch to "{next_model.model_id}" (arch: {next_model.arch}). '
                        f'Use backend="{next_model.supports_backends[0]}", '
                        f'lora_target_modules="{next_model.lora_target_modules}".'
                    )
                    print(f"  🔄 Rotating: {failed_model} → {next_model.model_id}")
                else:
                    model_switch_note = "All models exhausted. Simplify: remove LoRA, minimal config."
                    print(f"  ❌ All models exhausted")

        # RAG error context
        if use_rag:
            try:
                retriever = get_retriever()
                error_source_context = retriever.format_error_context(raw_traceback, max_chars=3000)
            except Exception:
                error_source_context = None
        else:
            error_source_context = None

        print(f"  ❌ Failed ({score.failure_category}). "
              f"{'Model rotated. ' if model_switch_note else ''}"
              f"Reflections: {len(reflexion.reflections)}")

    # --- Post-loop: log unresolved errors ---
    if not trajectory.is_successful and use_rag:
        _log_unresolved(problem_str, trajectory)

    return trajectory


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _add_to_pool(problem: str, plan: str, code: str, score: QualityScore, turn: int):
    """Add a successful attempt to the quality pool."""
    pool = get_pool()

    # Extract model/backend/algorithm from code
    model_match = re.search(r'model_name\s*=\s*"([^"]+)"', code)
    backend_match = re.search(r'backend\s*=\s*"([^"]+)"', code)
    algo_match = re.search(r'algorithm\s*=\s*"([^"]+)"', code)

    record = PoolRecord(
        problem=problem,
        code=code,
        plan=plan,
        composite_score=score.composite_score,
        api_calls=score.api_calls_found,
        outcome_signals=score.outcome_signals,
        model_used=model_match.group(1) if model_match else "",
        backend_used=backend_match.group(1) if backend_match else "",
        algorithm=algo_match.group(1) if algo_match else "sft",
        turn_number=turn,
    )
    pool.add(record)
    print(f"  📦 Added to quality pool (score: {score.composite_score:.2f})")


def _log_unresolved(problem: str, trajectory: TaskTrajectory):
    """Log failed trajectory to bug report DB."""
    try:
        retriever = get_retriever()
        last_traceback = trajectory.attempts[-1].traceback or ""
        suggested_fix = _generate_fix_suggestion(problem, last_traceback)
        retriever.report_unresolved_error(
            problem=problem,
            traceback_text=last_traceback,
            attempts=[
                {"turn": a.turn_number, "success": a.success, "traceback": a.traceback}
                for a in trajectory.attempts
            ],
            suggested_fix=suggested_fix,
        )
    except Exception as e:
        print(f"  ⚠️ Failed to log unresolved error: {e}")


def _generate_fix_suggestion(problem: str, traceback: str) -> Optional[str]:
    """LLM diagnoses root cause for the bug report."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"""A script failed repeatedly.
Problem: {problem[:500]}
Traceback: {traceback[:1000]}

Is this a USER code bug or a LIBRARY bug (aligntune/trl/unsloth)?
Root cause? Suggested fix? Under 200 words."""}],
        )
        return response.choices[0].message.content
    except Exception:
        return None