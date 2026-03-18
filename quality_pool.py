import os
import json
import hashlib
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict

import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# Quality Pool: Stores passing solutions,
# retrieves them as few-shot examples for
# future synthesis and repair rounds.
# ==========================================

CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
POOL_COLLECTION = "quality_pool"
POOL_JSONL = os.getenv("QUALITY_POOL_PATH", "./output/quality_pool.jsonl")


@dataclass
class PoolRecord:
    """A single passing solution stored in the quality pool."""
    problem: str
    code: str
    plan: str
    composite_score: float
    api_calls: List[str]
    outcome_signals: List[str]
    model_used: str = ""
    backend_used: str = ""
    algorithm: str = ""  # "sft", "dpo", "grpo", etc.
    turn_number: int = 1
    content_hash: str = ""

    def __post_init__(self):
        self.content_hash = hashlib.md5(self.code.encode()).hexdigest()


class QualityPool:
    """
    Persistent store of passing solutions with ChromaDB-backed retrieval.
    
    Two uses:
    1. Few-shot injection: retrieve similar passing solutions to inject into
       synthesis and solver prompts.
    2. Round-over-round feedback: pool grows across pipeline runs, each run
       benefits from all prior successful generations.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL, device="cpu",
        )
        self.collection = self.client.get_or_create_collection(
            name=POOL_COLLECTION,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        # Also persist to JSONL for portability
        os.makedirs(os.path.dirname(POOL_JSONL) or ".", exist_ok=True)

    def add(self, record: PoolRecord):
        """Add a passing solution to the pool. Deduplicates by content hash."""
        embed_text = f"Problem: {record.problem[:500]}\nCode: {record.code[:1000]}"

        self.collection.upsert(
            ids=[record.content_hash],
            documents=[embed_text],
            metadatas=[{
                "problem": record.problem[:2000],
                "code": record.code[:5000],
                "plan": record.plan[:2000],
                "composite_score": record.composite_score,
                "api_calls": json.dumps(record.api_calls),
                "outcome_signals": json.dumps(record.outcome_signals),
                "model_used": record.model_used,
                "backend_used": record.backend_used,
                "algorithm": record.algorithm,
                "turn_number": record.turn_number,
            }],
        )

        # Append to JSONL
        with open(POOL_JSONL, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def retrieve_similar(self, query: str, top_k: int = 3, min_score: float = 0.3) -> List[Dict]:
        """
        Retrieve the most similar passing solutions for few-shot injection.
        Returns list of dicts with 'problem', 'code', 'plan', 'similarity'.
        """
        if self.collection.count() == 0:
            return []

        # Don't request more than what exists
        actual_k = min(top_k, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["metadatas", "distances"],
        )

        shots = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                similarity = 1 - results["distances"][0][i]

                if similarity < min_score:
                    continue

                shots.append({
                    "problem": meta.get("problem", ""),
                    "code": meta.get("code", ""),
                    "plan": meta.get("plan", ""),
                    "composite_score": meta.get("composite_score", 0),
                    "similarity": round(similarity, 3),
                })

        return shots

    def format_few_shots(self, query: str, top_k: int = 2) -> str:
        """
        Retrieves similar solutions and formats them as a prompt block.
        Injected into Planner/Solver prompts for few-shot guidance.
        """
        shots = self.retrieve_similar(query, top_k=top_k)
        if not shots:
            return ""

        parts = [
            "\n" + "=" * 60,
            "EXAMPLES OF PASSING SOLUTIONS (from previous runs):",
            "=" * 60,
        ]
        for i, shot in enumerate(shots, 1):
            parts.append(f"\n### Example {i} (score: {shot['composite_score']:.2f})")
            parts.append(f"Problem: {shot['problem'][:300]}")
            parts.append(f"```python\n{shot['code'][:1500]}\n```")

        return "\n".join(parts)

    def get_stats(self) -> Dict:
        """Pool statistics."""
        total = self.collection.count()
        stats = {"total_records": total}

        if total > 0:
            all_records = self.collection.get(include=["metadatas"])
            algorithms = {}
            backends = {}
            for meta in all_records["metadatas"]:
                algo = meta.get("algorithm", "unknown")
                backend = meta.get("backend_used", "unknown")
                algorithms[algo] = algorithms.get(algo, 0) + 1
                backends[backend] = backends.get(backend, 0) + 1
            stats["by_algorithm"] = algorithms
            stats["by_backend"] = backends

        return stats

    def get_failure_distribution(self) -> Dict[str, int]:
        """
        Returns failure category counts from the JSONL log.
        Used by the curriculum tracker to bias future generation.
        """
        # This reads from a separate failure log, not the pool itself
        # (pool only stores successes)
        failure_log = POOL_JSONL.replace("quality_pool", "failure_log")
        counts: Dict[str, int] = {}
        if os.path.exists(failure_log):
            with open(failure_log) as f:
                for line in f:
                    rec = json.loads(line)
                    cat = rec.get("failure_category", "unknown")
                    counts[cat] = counts.get(cat, 0) + 1
        return counts

    def log_failure(self, problem: str, code: str, failure_category: str, traceback: str):
        """Log a failed attempt for curriculum tracking."""
        failure_log = POOL_JSONL.replace("quality_pool", "failure_log")
        with open(failure_log, "a") as f:
            f.write(json.dumps({
                "problem": problem[:2000],
                "code": code[:3000],
                "failure_category": failure_category,
                "traceback": traceback[:1000],
            }) + "\n")