import os
import re
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# RAG Retriever: Error-Aware Code Search (ChromaDB)
# ==========================================

CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

CODE_COLLECTION = "code_chunks"
BUGS_COLLECTION = "unresolved_errors"


@dataclass
class RetrievedChunk:
    source: str
    filepath: str
    chunk_type: str
    name: str
    parent_class: Optional[str]
    content: str
    similarity: float


class RAGRetriever:
    """Retrieves relevant code chunks and API docs via ChromaDB."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device="cpu",  # CPU is fine for single-query retrieval
        )
        self.collection = self.client.get_or_create_collection(
            name=CODE_COLLECTION,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.bugs_collection = self.client.get_or_create_collection(
            name=BUGS_COLLECTION,
            embedding_function=self.embed_fn,
        )

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Semantic search over indexed code chunks."""
        where_filter = None
        conditions = []
        if source_filter:
            conditions.append({"source": source_filter})
        if chunk_type_filter:
            conditions.append({"chunk_type": chunk_type_filter})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1 - distance  # cosine distance -> similarity

                chunks.append(RetrievedChunk(
                    source=meta.get("source", ""),
                    filepath=meta.get("filepath", ""),
                    chunk_type=meta.get("chunk_type", ""),
                    name=meta.get("name", ""),
                    parent_class=meta.get("parent_class", "") or None,
                    content=meta.get("content", ""),
                    similarity=similarity,
                ))

        return chunks

    # ------------------------------------------------------------------
    # Error-specific retrieval
    # ------------------------------------------------------------------

    def retrieve_for_error(self, traceback_text: str, top_k: int = 5) -> Dict[str, List[RetrievedChunk]]:
        """
        Given a traceback, retrieves relevant code from all three libraries.
        """
        error_info = self._parse_traceback(traceback_text)
        results = {}

        error_query = f"{error_info['error_type']}: {error_info['error_message']}"
        results["by_error"] = self.search(error_query, top_k=top_k)

        if error_info["mentioned_names"]:
            name_query = " ".join(error_info["mentioned_names"][:5])
            results["by_names"] = self.search(name_query, top_k=3)

        for lib in ["aligntune", "trl", "unsloth"]:
            if lib in traceback_text.lower():
                results[f"from_{lib}"] = self.search(
                    error_query, top_k=3, source_filter=lib
                )

        results["api_docs"] = self.search(
            error_query, top_k=2, source_filter="api_docs"
        )

        return results

    def retrieve_api_context(self, problem: str, top_k: int = 5) -> str:
        """
        Replaces the static API docs string with RAG-retrieved context.
        """
        chunks = self.search(problem, top_k=top_k, source_filter="api_docs")
        code_chunks = self.search(problem, top_k=3, source_filter="aligntune")

        context_parts = ["# Relevant API Documentation\n"]
        for c in chunks:
            context_parts.append(f"## {c.name}\n{c.content}\n")

        if code_chunks:
            context_parts.append("\n# Relevant Source Code (aligntune internals)\n")
            for c in code_chunks:
                label = f"{c.filepath}:{c.name}"
                if c.parent_class:
                    label = f"{c.filepath}:{c.parent_class}.{c.name}"
                context_parts.append(f"### {label}\n```python\n{c.content}\n```\n")

        return "\n".join(context_parts)

    def format_error_context(self, traceback_text: str, max_chars: int = 4000) -> str:
        """
        Builds rich error context for the LLM: traceback + relevant source code.
        """
        retrieved = self.retrieve_for_error(traceback_text)

        parts = [f"TRACEBACK:\n{traceback_text}\n"]
        parts.append("=" * 60)
        parts.append("RELEVANT SOURCE CODE FROM LIBRARIES:\n")

        seen_hashes = set()
        char_count = len(parts[0])

        for category, chunks in retrieved.items():
            for chunk in chunks:
                chunk_id = f"{chunk.filepath}:{chunk.name}"
                if chunk_id in seen_hashes:
                    continue
                seen_hashes.add(chunk_id)

                entry = (
                    f"\n--- [{chunk.source}] {chunk.filepath} :: {chunk.name} "
                    f"(similarity: {chunk.similarity:.3f}) ---\n"
                    f"{chunk.content[:800]}\n"
                )

                if char_count + len(entry) > max_chars:
                    break
                parts.append(entry)
                char_count += len(entry)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Traceback parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_traceback(traceback_text: str) -> Dict:
        info = {
            "error_type": "Unknown",
            "error_message": "",
            "mentioned_names": [],
            "mentioned_files": [],
        }

        lines = traceback_text.strip().splitlines()

        if lines:
            last = lines[-1]
            if ":" in last:
                parts = last.split(":", 1)
                info["error_type"] = parts[0].strip()
                info["error_message"] = parts[1].strip()
            else:
                info["error_message"] = last.strip()

        for line in lines:
            match = re.search(r'in (\w+)', line)
            if match:
                info["mentioned_names"].append(match.group(1))
            match = re.search(r'File "(.+?)"', line)
            if match:
                info["mentioned_files"].append(match.group(1))

        id_matches = re.findall(r"'(\w+)'", info["error_message"])
        info["mentioned_names"].extend(id_matches)

        return info

    # ------------------------------------------------------------------
    # Bug report DB
    # ------------------------------------------------------------------

    def report_unresolved_error(
        self,
        problem: str,
        traceback_text: str,
        attempts: List[Dict],
        suggested_fix: Optional[str] = None,
    ) -> str:
        """Logs an unresolved error to the bug report collection."""
        error_info = self._parse_traceback(traceback_text)

        source_lib = "unknown"
        for lib in ["aligntune", "trl", "unsloth"]:
            if lib in traceback_text.lower():
                source_lib = lib
                break

        # Retrieve related chunks for context
        related = self.retrieve_for_error(traceback_text, top_k=3)
        related_json = []
        for cat, chunks in related.items():
            for c in chunks:
                related_json.append({
                    "source": c.source, "filepath": c.filepath,
                    "name": c.name, "similarity": round(c.similarity, 3),
                })

        severity = "low"
        if error_info["error_type"] in ("TypeError", "AttributeError", "ImportError"):
            severity = "medium"
        if "segfault" in traceback_text.lower() or "CUDA" in traceback_text:
            severity = "high"
        if error_info["error_type"] == "NotImplementedError":
            severity = "high"

        error_id = f"bug_{int(time.time())}_{hash(traceback_text) % 10000}"

        self.bugs_collection.upsert(
            ids=[error_id],
            documents=[f"{error_info['error_type']}: {error_info['error_message']}"],
            metadatas=[{
                "problem": problem[:1000],
                "error_traceback": traceback_text[:3000],
                "error_type": error_info["error_type"],
                "source_library": source_lib,
                "related_chunks": json.dumps(related_json)[:2000],
                "attempts": json.dumps(attempts)[:3000],
                "suggested_fix": (suggested_fix or "")[:2000],
                "severity": severity,
                "status": "open",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }],
        )

        print(f"  📋 Logged unresolved error {error_id} (severity: {severity}, source: {source_lib})")
        return error_id

    def get_unresolved_errors(
        self,
        source_library: Optional[str] = None,
        severity: Optional[str] = None,
        status: str = "open",
    ) -> List[Dict]:
        """Query the bug report collection for developer review."""
        conditions = [{"status": status}]
        if source_library:
            conditions.append({"source_library": source_library})
        if severity:
            conditions.append({"severity": severity})

        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        results = self.bugs_collection.get(
            where=where_filter,
            include=["metadatas"],
        )

        errors = []
        if results["ids"]:
            for i, bug_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                errors.append({
                    "id": bug_id,
                    "created_at": meta.get("created_at", ""),
                    "problem": meta.get("problem", ""),
                    "error_type": meta.get("error_type", ""),
                    "source_library": meta.get("source_library", ""),
                    "severity": meta.get("severity", ""),
                    "suggested_fix": meta.get("suggested_fix", ""),
                    "traceback_preview": meta.get("error_traceback", "")[:300],
                })

        # Sort: high > medium > low
        severity_order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
        errors.sort(key=lambda e: severity_order.get(e["severity"], 3))

        return errors

    def close(self):
        """No-op for ChromaDB (no persistent connection to close)."""
        pass