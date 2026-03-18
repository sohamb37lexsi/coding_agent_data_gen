"""
Run Analysis Script
Usage: python analyze_run.py --output-dir output --examples 15
Generates:
  - output/full_analysis.txt      (terminal report, if redirected)
  - output/run_report.json        (summary statistics)
  - output/problems_generated.json (all problems with metadata)
"""

import json
import os
import re
import argparse
from collections import Counter

def load_jsonl(path):
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def extract_code_from_assistant(text):
    blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    return blocks[-1].strip() if blocks else ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--examples", type=int, default=15)
    args = parser.parse_args()

    traj_path = os.path.join(args.output_dir, "trajectory_sft.json")
    with open(traj_path) as f:
        trajs = json.load(f)

    sft_data = load_jsonl(os.path.join(args.output_dir, "sft_data.jsonl"))
    dpo_data = load_jsonl(os.path.join(args.output_dir, "dpo_data.jsonl"))
    pool_data = load_jsonl(os.path.join(args.output_dir, "quality_pool.jsonl"))
    failure_data = load_jsonl(os.path.join(args.output_dir, "failure_log.jsonl"))

    total = len(trajs)
    passed = [t for t in trajs if t["is_successful"]]
    failed = [t for t in trajs if not t["is_successful"]]

    turn_counts_passed = [len(t["attempts"]) for t in passed]
    turn_counts_failed = [len(t["attempts"]) for t in failed]

    first_try_trajs = [t for t in passed if len(t["attempts"]) == 1]
    multi_turn_trajs = [t for t in passed if len(t["attempts"]) > 1]

    # ================================================================
    # SAVE: problems_generated.json
    # ================================================================
    problems_export = []
    for i, t in enumerate(trajs):
        problems_export.append({
            "problem_number": i + 1,
            "problem": t["problem"],
            "solved": t["is_successful"],
            "turns_taken": len(t["attempts"]),
            "final_score": t["attempts"][-1].get("success", False),
            "algorithm": _detect_algorithm(t),
            "model_used": _detect_model(t),
            "backend_used": _detect_backend(t),
            "failure_category": _get_failure_category(t) if not t["is_successful"] else None,
            "last_error": _get_last_error(t) if not t["is_successful"] else None,
        })

    problems_path = os.path.join(args.output_dir, "problems_generated.json")
    with open(problems_path, "w") as f:
        json.dump(problems_export, f, indent=2)
    print(f"Saved {len(problems_export)} problems to {problems_path}\n")

    # ================================================================
    # SECTION 1: HIGH-LEVEL SUMMARY
    # ================================================================
    print("=" * 70)
    print("PIPELINE RUN ANALYSIS")
    print("=" * 70)

    print(f"\n--- Overall ---")
    print(f"  Total problems:       {total}")
    print(f"  Solved:               {len(passed)} ({100*len(passed)/total:.0f}%)")
    print(f"  Failed:               {len(failed)} ({100*len(failed)/total:.0f}%)")

    first_try = len(first_try_trajs)
    multi_turn = len(multi_turn_trajs)

    print(f"\n--- Resolution Breakdown ---")
    print(f"  First-try passes:     {first_try} ({100*first_try/max(len(passed),1):.0f}% of solved)")
    print(f"  Multi-turn passes:    {multi_turn} ({100*multi_turn/max(len(passed),1):.0f}% of solved)")
    print(f"  Avg turns (solved):   {sum(turn_counts_passed)/max(len(passed),1):.1f}")
    print(f"  Avg turns (failed):   {sum(turn_counts_failed)/max(len(failed),1):.1f}")
    print(f"  Total LLM calls:      {sum(len(t['attempts']) for t in trajs)}")
    print(f"  Wasted LLM calls:     {sum(turn_counts_failed)} ({100*sum(turn_counts_failed)/sum(len(t['attempts']) for t in trajs):.0f}% of total)")

    # Turn distribution
    turn_dist = Counter(len(t["attempts"]) for t in trajs)
    print(f"\n--- Turns Distribution ---")
    for turns in sorted(turn_dist.keys()):
        count = turn_dist[turns]
        solved_at = sum(1 for t in passed if len(t["attempts"]) == turns)
        failed_at = sum(1 for t in failed if len(t["attempts"]) == turns)
        bar = "█" * count
        print(f"  {turns:2d} turns: {bar} {count} (✅ {solved_at}, ❌ {failed_at})")

    # Algorithm distribution
    algo_counter = Counter()
    for rec in pool_data:
        algo_counter[rec.get("algorithm", "unknown")] += 1
    print(f"\n--- Algorithm Coverage (Quality Pool) ---")
    for algo, count in algo_counter.most_common():
        print(f"  {algo:12s}: {count}")

    # Failure categories
    fail_counter = Counter()
    for rec in failure_data:
        fail_counter[rec.get("failure_category", "unknown")] += 1
    print(f"\n--- Failure Categories ---")
    for cat, count in fail_counter.most_common():
        print(f"  {cat:12s}: {count}")

    # Dataset output
    print(f"\n--- Output Datasets ---")
    print(f"  SFT examples:         {len(sft_data)}")
    print(f"  DPO pairs:            {len(dpo_data)}")
    print(f"  Quality pool entries:  {len(pool_data)}")
    print(f"  Failure log entries:   {len(failure_data)}")

    if dpo_data:
        gaps = [d.get("chosen_score", 1) - d.get("rejected_score", 0) for d in dpo_data]
        print(f"  Avg DPO quality gap:  {sum(gaps)/len(gaps):.3f}")

    # ================================================================
    # SECTION 2: ALL PROBLEMS WITH TURNS
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"ALL PROBLEMS (with turn counts)")
    print(f"{'=' * 70}")

    for i, t in enumerate(trajs):
        status = "✅" if t["is_successful"] else "❌"
        turns = len(t["attempts"])
        algo = _detect_algorithm(t)
        print(f"  {status} #{i+1:3d} [{turns:2d} turns] [{algo:6s}] {t['problem'][:90]}...")

    # ================================================================
    # SECTION 3: FIRST-TRY vs MULTI-TURN COMPARISON
    # ================================================================
    n_examples = min(args.examples, max(len(first_try_trajs), len(multi_turn_trajs)))

    print(f"\n{'=' * 70}")
    print(f"COMPARISON: FIRST-TRY PASSES vs MULTI-TURN PASSES")
    print(f"{'=' * 70}")

    print(f"\n{'─' * 70}")
    print(f"FIRST-TRY PASSES ({len(first_try_trajs)} total) — solved on attempt 1")
    print(f"{'─' * 70}")

    for t in first_try_trajs[:n_examples]:
        a = t["attempts"][0]
        code = a["code"]
        algo = _detect_algorithm(t)
        model = _detect_model(t)
        print(f"\n  Problem: {t['problem'][:120]}")
        print(f"  Algorithm: {algo} | Model: {model} | Turns: 1")
        print(f"  Code ({len(code.splitlines())} lines):")
        for line in code.splitlines()[:15]:
            print(f"    {line}")
        if len(code.splitlines()) > 15:
            print(f"    ... ({len(code.splitlines()) - 15} more lines)")
        print()

    print(f"\n{'─' * 70}")
    print(f"MULTI-TURN PASSES ({len(multi_turn_trajs)} total) — needed self-correction")
    print(f"{'─' * 70}")

    # Sort by most turns needed (most interesting first)
    multi_turn_trajs_sorted = sorted(multi_turn_trajs, key=lambda t: len(t["attempts"]), reverse=True)

    for t in multi_turn_trajs_sorted[:n_examples]:
        algo = _detect_algorithm(t)
        model = _detect_model(t)
        turns = len(t["attempts"])
        print(f"\n  Problem: {t['problem'][:120]}")
        print(f"  Algorithm: {algo} | Model: {model} | Turns: {turns}")

        for a in t["attempts"]:
            status = "✅" if a["success"] else "❌"
            code_lines = len(a["code"].splitlines())

            if a["success"]:
                print(f"    Turn {a['turn_number']:2d} {status} — PASSED ({code_lines} lines)")
                print(f"      Code preview: {a['code'].splitlines()[0] if a['code'] else ''}...")
            else:
                error = _extract_error_line(a.get("traceback", ""))
                print(f"    Turn {a['turn_number']:2d} {status} — {error[:100]}")

        # Show final passing code
        final = t["attempts"][-1]
        print(f"  Final Code ({len(final['code'].splitlines())} lines):")
        for line in final["code"].splitlines()[:15]:
            print(f"    {line}")
        if len(final["code"].splitlines()) > 15:
            print(f"    ... ({len(final['code'].splitlines()) - 15} more lines)")
        print()

    # ================================================================
    # SECTION 4: FAILED PROBLEMS ANALYSIS
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"FAILED PROBLEMS — ROOT CAUSE ANALYSIS ({len(failed)} problems)")
    print(f"{'=' * 70}")

    error_groups = {}
    for t in failed:
        error_line = _get_last_error(t)
        error_class = error_line.split(":")[0].strip() if ":" in error_line else error_line[:50]
        if error_class not in error_groups:
            error_groups[error_class] = []
        error_groups[error_class].append(t)

    for error_class, problems in sorted(error_groups.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{len(problems)}x] {error_class}")
        for t in problems[:3]:
            turns = len(t["attempts"])
            algo = _detect_algorithm(t)
            print(f"    → [{turns} turns] [{algo}] {t['problem'][:90]}...")

    # ================================================================
    # SECTION 5: SFT & DPO SAMPLES
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"SFT DATA SAMPLES ({len(sft_data)} total)")
    print(f"{'=' * 70}")

    for i, rec in enumerate(sft_data[:3]):
        asst_content = rec["messages"][1]["content"]
        code = extract_code_from_assistant(asst_content)
        problem_match = re.search(r'Problem: (.+?)(?:\n|$)', rec["messages"][0]["content"])
        problem = problem_match.group(1)[:120] if problem_match else "?"

        print(f"\n  SFT {i+1}: {problem}")
        print(f"  Code: {len(code)} chars, {len(code.splitlines())} lines")
        for line in code.splitlines()[:10]:
            print(f"    {line}")
        if len(code.splitlines()) > 10:
            print(f"    ...")

    if dpo_data:
        print(f"\n{'=' * 70}")
        print(f"DPO PAIR SAMPLES ({len(dpo_data)} total)")
        print(f"{'=' * 70}")

        for i, rec in enumerate(dpo_data[:3]):
            chosen_code = extract_code_from_assistant(rec["chosen"][0]["content"])
            rejected_code = extract_code_from_assistant(rec["rejected"][0]["content"])
            print(f"\n  DPO {i+1}: chosen={rec.get('chosen_score','?')}, rejected={rec.get('rejected_score','?')}, gap={rec.get('chosen_score',1)-rec.get('rejected_score',0):.3f}")
            print(f"  Chosen:   {len(chosen_code.splitlines())} lines")
            print(f"  Rejected: {len(rejected_code.splitlines())} lines")

    # ================================================================
    # SAVE REPORT
    # ================================================================
    report = {
        "summary": {
            "total_problems": total,
            "solved": len(passed),
            "failed": len(failed),
            "pass_rate": round(len(passed) / total, 3),
            "first_try_passes": first_try,
            "multi_turn_passes": multi_turn,
            "first_try_rate": round(first_try / max(len(passed), 1), 3),
            "avg_turns_solved": round(sum(turn_counts_passed) / max(len(passed), 1), 2),
            "avg_turns_failed": round(sum(turn_counts_failed) / max(len(failed), 1), 2),
            "total_llm_calls": sum(len(t["attempts"]) for t in trajs),
            "wasted_llm_calls": sum(turn_counts_failed),
        },
        "datasets": {
            "sft_examples": len(sft_data),
            "dpo_pairs": len(dpo_data),
            "quality_pool": len(pool_data),
            "failure_log": len(failure_data),
        },
        "algorithm_coverage": dict(algo_counter),
        "failure_categories": dict(fail_counter),
        "error_groups": {k: len(v) for k, v in error_groups.items()},
        "turn_distribution": {str(k): v for k, v in sorted(turn_dist.items())},
    }

    report_path = os.path.join(args.output_dir, "run_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n\nSaved: {report_path}")
    print(f"Saved: {problems_path}")


# ================================================================
# Helper functions
# ================================================================

def _detect_algorithm(t):
    """Detect algorithm from the last attempt's code."""
    code = t["attempts"][-1].get("code", "")
    if "create_rl_trainer" in code:
        match = re.search(r'algorithm\s*=\s*["\']([^"\']+)', code)
        return match.group(1) if match else "rl"
    if "create_sft_trainer" in code:
        return "sft"
    if "EvalConfig" in code and "create_" not in code:
        return "eval"
    return "unknown"

def _detect_model(t):
    """Detect model from the last attempt's code."""
    code = t["attempts"][-1].get("code", "")
    match = re.search(r'model_name\s*=\s*["\']([^"\']+)', code)
    return match.group(1) if match else "unknown"

def _detect_backend(t):
    """Detect backend from the last attempt's code."""
    code = t["attempts"][-1].get("code", "")
    match = re.search(r'backend\s*=\s*["\']([^"\']+)', code)
    return match.group(1) if match else "unknown"

def _get_failure_category(t):
    """Get failure category from the last failed attempt."""
    last = t["attempts"][-1]
    tb = (last.get("traceback") or "").lower()
    if "syntaxerror" in tb:
        return "syntax"
    if "modulenotfounderror" in tb or "importerror" in tb:
        return "import"
    if "typeerror" in tb or "attributeerror" in tb:
        return "type"
    if "timeout" in tb or "timed out" in tb:
        return "timeout"
    return "runtime"

def _get_last_error(t):
    """Extract the last error line from a failed trajectory."""
    last = t["attempts"][-1]
    tb = (last.get("traceback") or "").strip()
    lines = tb.splitlines()
    return lines[-1] if lines else "Unknown error"

def _extract_error_line(tb):
    """Extract the final error line from a traceback string."""
    if not tb:
        return "No traceback"
    lines = tb.strip().splitlines()
    return lines[-1] if lines else "Unknown"


if __name__ == "__main__":
    main()