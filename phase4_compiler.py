import os
import json
from typing import List
from data_types import TaskTrajectory
from quality_scorer import score_attempt

# ==========================================
# Phase 4: Quality-Ranked Dataset Construction
# ==========================================

def compile_datasets(trajectories: List[TaskTrajectory], output_dir: str = "output"):
    """
    Compiles trajectories into SFT and DPO datasets.
    
    Enhanced over the original:
    - DPO pairs are ranked by quality score, not just pass/fail
    - Failed trajectories with partial progress still produce DPO pairs
    - Multiple DPO pairs per trajectory when quality differences exist
    """
    os.makedirs(output_dir, exist_ok=True)

    sft_data = []
    dpo_data = []

    for traj in trajectories:
        # --- SFT: only from successful trajectories ---
        if traj.is_successful:
            final = traj.attempts[-1]
            sft_prompt = f"Problem: {traj.problem}\n\nAPI Docs:\n{traj.api_context}"
            sft_response = f"<plan>\n{final.plan}\n</plan>\n\n```python\n{final.code}\n```"

            sft_data.append({
                "messages": [
                    {"role": "user", "content": sft_prompt},
                    {"role": "assistant", "content": sft_response},
                ]
            })

        # --- DPO: ranked pairs from ALL trajectories with 2+ attempts ---
        if len(traj.attempts) < 2:
            continue

        sft_prompt = f"Problem: {traj.problem}\n\nAPI Docs:\n{traj.api_context}"

        # Score every attempt for ranking
        scored_attempts = []
        for attempt in traj.attempts:
            # We don't have stdout/stderr stored separately, but we can
            # use success + traceback as proxy
            if attempt.success:
                # Successful: high score
                scored_attempts.append((attempt, 1.0))
            elif attempt.traceback:
                # Failed: score based on how far it got
                score = _estimate_score_from_traceback(attempt.code, attempt.traceback)
                scored_attempts.append((attempt, score))
            else:
                scored_attempts.append((attempt, 0.0))

        # Sort by score descending
        scored_attempts.sort(key=lambda x: x[1], reverse=True)

        # Generate DPO pairs: each adjacent pair where score differs
        for i in range(len(scored_attempts) - 1):
            better_attempt, better_score = scored_attempts[i]
            worse_attempt, worse_score = scored_attempts[i + 1]

            # Only create pair if there's a meaningful quality gap
            if better_score - worse_score < 0.05:
                continue

            chosen = f"<plan>\n{better_attempt.plan}\n</plan>\n\n```python\n{better_attempt.code}\n```"
            rejected = f"<plan>\n{worse_attempt.plan}\n</plan>\n\n```python\n{worse_attempt.code}\n```"

            dpo_data.append({
                "prompt": sft_prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "chosen_score": better_score,
                "rejected_score": worse_score,
            })

    # Export
    with open(os.path.join(output_dir, "sft_data.jsonl"), "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(output_dir, "dpo_data.jsonl"), "w") as f:
        for item in dpo_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n--- Phase 4: Dataset Construction Complete ---")
    print(f"Exported {len(sft_data)} SFT examples to {output_dir}/sft_data.jsonl")
    print(f"Exported {len(dpo_data)} DPO pairs to {output_dir}/dpo_data.jsonl")

    # Print DPO quality breakdown
    if dpo_data:
        avg_gap = sum(d["chosen_score"] - d["rejected_score"] for d in dpo_data) / len(dpo_data)
        print(f"Average DPO quality gap: {avg_gap:.3f}")


def _estimate_score_from_traceback(code: str, traceback: str) -> float:
    """
    Estimate a quality score for a failed attempt based on how far it got.
    Higher = closer to success.
    """
    import ast as _ast

    score = 0.0

    # Syntax valid?
    try:
        _ast.parse(code)
        score += 0.10
    except SyntaxError:
        return score

    # Has aligntune imports?
    if "aligntune" in code:
        score += 0.20

    # Classify failure depth from traceback
    traceback_lower = traceback.lower()
    if "modulenotfounderror" in traceback_lower or "importerror" in traceback_lower:
        return score  # Stopped at import
    score += 0.05  # Got past imports

    if "typeerror" in traceback_lower or "attributeerror" in traceback_lower:
        return score + 0.05  # Got to API call but wrong args
    score += 0.05

    if "runtimeerror" in traceback_lower or "valueerror" in traceback_lower:
        return score + 0.10  # Got deeper into execution

    if "timeout" in traceback_lower or "timed out" in traceback_lower:
        return score + 0.20  # Actually ran training (good sign!)

    return score