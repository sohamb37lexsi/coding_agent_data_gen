import os
import json
import argparse
from dataclasses import asdict

from api_docs import ALIGNTUNE_API_DOCS
from phase1_generator import generate_problems
from phase2_agent import run_react_loop, get_pool
from phase4_compiler import compile_datasets

# ==========================================
# Main Execution Flow
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="AlignTune Data Synthesis Pipeline")
    parser.add_argument("--num-problems", type=int, default=5, help="Problems to generate")
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns per problem")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--use-rag", action="store_true", default=False, help="Enable RAG retrieval")
    parser.add_argument("--show-bugs", action="store_true", help="Show unresolved errors and exit")
    parser.add_argument("--export-bugs", type=str, default=None, help="Export bugs to JSON and exit")
    parser.add_argument("--pool-stats", action="store_true", help="Show quality pool stats and exit")
    args = parser.parse_args()

    if args.show_bugs:
        _show_bug_reports()
        return
    if args.export_bugs:
        _export_bug_reports(args.export_bugs)
        return
    if args.pool_stats:
        _show_pool_stats()
        return

    print("=" * 60)
    print("AlignTune Data Synthesis Pipeline")
    print("=" * 60)
    print(f"  Problems:   {args.num_problems}")
    print(f"  Max turns:  {args.max_turns}")
    print(f"  RAG:        {'enabled' if args.use_rag else 'static docs'}")
    print(f"  Output:     {args.output_dir}")
    print()

    # Retrieve RAG retriever if enabled (for source_code synthesis strategy)
    rag_retriever = None
    if args.use_rag:
        try:
            from rag_retriever import RAGRetriever
            rag_retriever = RAGRetriever()
            print("  📚 RAG retriever initialized")
        except Exception as e:
            print(f"  ⚠️ RAG init failed ({e}), source_code strategy will fallback")

    # Quality pool for few-shot injection and mutation strategy
    pool = get_pool()
    pool_size = pool.collection.count() if hasattr(pool, 'collection') else 0
    print(f"  📦 Quality pool: {pool_size} existing records")
    print()

    # Phase 1: Multi-strategy problem generation
    generated_problems = generate_problems(
        ALIGNTUNE_API_DOCS,
        num_problems=args.num_problems,
        quality_pool=pool,
        rag_retriever=rag_retriever,
    )
    print(f"\n  Generated {len(generated_problems)} problems total\n")

    # Phase 2 & 3: Planner-Solver loop with Reflexion + scoring
    all_trajectories = []
    for i, problem in enumerate(generated_problems, 1):
        print(f"\n{'='*60}")
        print(f"Problem {i}/{len(generated_problems)}")
        print(f"{'='*60}")
        traj = run_react_loop(
            problem,
            api_docs_fallback=ALIGNTUNE_API_DOCS,
            max_turns=args.max_turns,
            use_rag=args.use_rag,
        )
        all_trajectories.append(traj)

    # Phase 4: Compile quality-ranked datasets
    compile_datasets(all_trajectories, output_dir=args.output_dir)

    # Export full trajectories for GRPO
    os.makedirs(args.output_dir, exist_ok=True)
    traj_path = os.path.join(args.output_dir, "trajectory_sft.json")
    with open(traj_path, "w") as f:
        json.dump([asdict(t) for t in all_trajectories], f, indent=2)
    print(f"Exported multi-turn trajectories to {traj_path}")

    # Summary
    total = len(all_trajectories)
    passed = sum(1 for t in all_trajectories if t.is_successful)
    pool_stats = pool.get_stats()
    failure_dist = pool.get_failure_distribution()

    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Problems solved:    {passed}/{total} ({100*passed/max(total,1):.0f}%)")
    print(f"  Quality pool size:  {pool_stats['total_records']}")
    if pool_stats.get('by_algorithm'):
        print(f"  By algorithm:       {pool_stats['by_algorithm']}")
    if failure_dist:
        print(f"  Failure categories: {failure_dist}")
    if args.use_rag and passed < total:
        print(f"\n  {total - passed} unresolved errors logged to bug report DB")
        print(f"  Run: python main.py --show-bugs")
    print(f"{'='*60}")


# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------

def _show_bug_reports():
    from rag_retriever import RAGRetriever
    retriever = RAGRetriever()
    for severity in ["high", "medium", "low"]:
        errors = retriever.get_unresolved_errors(severity=severity)
        if not errors:
            continue
        print(f"\n{'='*60}")
        print(f"  [{severity.upper()}] — {len(errors)} unresolved errors")
        print(f"{'='*60}")
        for err in errors:
            print(f"\n  #{err['id']} | {err['created_at']} | {err['source_library']}")
            print(f"  Error: {err['error_type']}")
            print(f"  Problem: {err['problem'][:100]}...")
            if err['suggested_fix']:
                print(f"  Fix: {err['suggested_fix'][:300]}...")
            print(f"  {'-'*40}")
    retriever.close()


def _export_bug_reports(filepath: str):
    from rag_retriever import RAGRetriever
    retriever = RAGRetriever()
    all_errors = []
    for severity in ["high", "medium", "low", "unknown"]:
        all_errors.extend(retriever.get_unresolved_errors(severity=severity))
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(all_errors, f, indent=2)
    print(f"Exported {len(all_errors)} bug reports to {filepath}")
    retriever.close()


def _show_pool_stats():
    pool = get_pool()
    stats = pool.get_stats()
    failures = pool.get_failure_distribution()
    print(f"\n--- Quality Pool Statistics ---")
    print(f"  Total passing records: {stats['total_records']}")
    if stats.get('by_algorithm'):
        for algo, count in stats['by_algorithm'].items():
            print(f"    {algo}: {count}")
    if stats.get('by_backend'):
        for backend, count in stats['by_backend'].items():
            print(f"    {backend}: {count}")
    if failures:
        print(f"\n--- Failure Distribution ---")
        for cat, count in sorted(failures.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")


if __name__ == "__main__":
    main()