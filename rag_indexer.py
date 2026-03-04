import os
import ast
import hashlib
import json
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# RAG Indexer: Parse & Embed Code + Docs (ChromaDB)
# ==========================================

CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# Collection names
CODE_COLLECTION = "code_chunks"
BUGS_COLLECTION = "unresolved_errors"


@dataclass
class CodeChunk:
    """A single indexable unit of code or documentation."""
    source: str            # "aligntune", "trl", "unsloth", "api_docs"
    filepath: str
    chunk_type: str        # "function", "class", "method", "module_doc", "api_doc"
    name: str
    content: str
    start_line: int = 0
    end_line: int = 0
    parent_class: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self):
        self.content_hash = hashlib.md5(self.content.encode()).hexdigest()


class PythonCodeParser:
    """Extracts functions, classes, and methods from Python source files."""

    @staticmethod
    def parse_file(filepath: str, source_label: str) -> List[CodeChunk]:
        chunks = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                source_code = f.read()
            tree = ast.parse(source_code)
        except (SyntaxError, UnicodeDecodeError):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if content.strip():
                chunks.append(CodeChunk(
                    source=source_label, filepath=filepath,
                    chunk_type="raw_module", name=Path(filepath).stem,
                    content=content[:4000],
                ))
            return chunks

        lines = source_code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent_is_class = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        for child in ast.iter_child_nodes(parent):
                            if child is node:
                                parent_is_class = True
                if parent_is_class:
                    continue
                chunk = PythonCodeParser._extract_function(node, lines, filepath, source_label)
                chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):
                cls_chunk = PythonCodeParser._extract_class(node, lines, filepath, source_label)
                chunks.append(cls_chunk)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_chunk = PythonCodeParser._extract_function(
                            item, lines, filepath, source_label, parent_class=node.name
                        )
                        chunks.append(method_chunk)

        return chunks

    @staticmethod
    def _extract_function(node, lines, filepath, source_label, parent_class=None) -> CodeChunk:
        start = node.lineno - 1
        end = node.end_lineno or (start + 1)
        content = "\n".join(lines[start:end])
        docstring = ast.get_docstring(node)
        return CodeChunk(
            source=source_label, filepath=filepath,
            chunk_type="method" if parent_class else "function",
            name=node.name, content=content,
            start_line=start + 1, end_line=end,
            parent_class=parent_class,
            signature=f"def {node.name}(...)",
            docstring=docstring,
        )

    @staticmethod
    def _extract_class(node, lines, filepath, source_label) -> CodeChunk:
        start = node.lineno - 1
        end = node.end_lineno or (start + 1)
        class_header_lines = []
        for i in range(start, min(end, start + 30)):
            class_header_lines.append(lines[i])
        content = "\n".join(class_header_lines)
        docstring = ast.get_docstring(node)
        return CodeChunk(
            source=source_label, filepath=filepath,
            chunk_type="class", name=node.name, content=content,
            start_line=start + 1, end_line=end,
            docstring=docstring,
        )


class DocsParser:
    """Splits API documentation into section-level chunks."""

    @staticmethod
    def parse_api_docs(docs_text: str, source_label: str = "api_docs") -> List[CodeChunk]:
        chunks = []
        sections = docs_text.split("\n## ")
        for section in sections:
            if not section.strip():
                continue
            lines = section.strip().splitlines()
            title = lines[0].strip("# ").strip()
            content = "\n".join(lines)
            chunks.append(CodeChunk(
                source=source_label, filepath="api_docs",
                chunk_type="api_doc", name=title, content=content,
            ))

        code_blocks = docs_text.split("```python")
        for i, block in enumerate(code_blocks[1:], 1):
            code = block.split("```")[0].strip()
            if len(code) > 50:
                chunks.append(CodeChunk(
                    source=source_label, filepath="api_docs",
                    chunk_type="api_example", name=f"example_{i}",
                    content=code,
                ))
        return chunks


class RAGIndexer:
    """Indexes code chunks into ChromaDB for semantic retrieval."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device=EMBED_DEVICE,
        )
        self.collection = self.client.get_or_create_collection(
            name=CODE_COLLECTION,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        # Bug report collection (stores metadata, not used for similarity search)
        self.bugs_collection = self.client.get_or_create_collection(
            name=BUGS_COLLECTION,
            embedding_function=self.embed_fn,
        )

    def _build_embed_text(self, chunk: CodeChunk) -> str:
        parts = [f"[{chunk.source}] {chunk.chunk_type}: {chunk.name}"]
        if chunk.parent_class:
            parts.append(f"class: {chunk.parent_class}")
        if chunk.docstring:
            parts.append(chunk.docstring[:300])
        parts.append(chunk.content[:1500])
        return "\n".join(parts)

    def index_chunks(self, chunks: List[CodeChunk], batch_size: int = 64):
        """Upserts code chunks into ChromaDB."""
        # Deduplicate by content_hash before indexing
        seen = {}
        for c in chunks:
            if c.content_hash not in seen:
                seen[c.content_hash] = c
        unique_chunks = list(seen.values())
        print(f"  Indexing {len(unique_chunks)} chunks ({len(chunks) - len(unique_chunks)} duplicates skipped)...")

        for i in range(0, len(unique_chunks), batch_size):
            batch = unique_chunks[i:i + batch_size]

            ids = [c.content_hash for c in batch]
            documents = [self._build_embed_text(c) for c in batch]
            metadatas = [
                {
                    "source": c.source,
                    "filepath": c.filepath,
                    "chunk_type": c.chunk_type,
                    "name": c.name,
                    "parent_class": c.parent_class or "",
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "content": c.content[:5000],
                }
                for c in batch
            ]

            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        print(f"  Done. Indexed {len(unique_chunks)} chunks.")

    def index_repo(self, repo_path: str, source_label: str):
        """Walks a Python repo and indexes all .py files."""
        print(f"\nIndexing repo: {source_label} ({repo_path})")
        py_files = glob.glob(os.path.join(repo_path, "**/*.py"), recursive=True)
        print(f"  Found {len(py_files)} Python files")

        all_chunks = []
        for fpath in py_files:
            rel = os.path.relpath(fpath, repo_path)
            if any(skip in rel for skip in ["__pycache__", "test_", "setup.py", ".egg"]):
                continue
            chunks = PythonCodeParser.parse_file(fpath, source_label)
            for c in chunks:
                c.filepath = rel
            all_chunks.extend(chunks)

        print(f"  Parsed {len(all_chunks)} code chunks")
        self.index_chunks(all_chunks)
        return len(all_chunks)

    def index_api_docs(self, docs_text: str, source_label: str = "api_docs"):
        """Indexes the API documentation string."""
        print(f"\nIndexing API docs ({source_label})...")
        chunks = DocsParser.parse_api_docs(docs_text, source_label)
        print(f"  Parsed {len(chunks)} doc chunks")
        self.index_chunks(chunks)
        return len(chunks)

    def get_stats(self):
        """Print index statistics."""
        total = self.collection.count()
        print(f"\n--- Index Statistics ---")
        print(f"  Total chunks: {total}")

        # Get breakdown by source
        for source in ["aligntune", "trl", "unsloth", "api_docs"]:
            result = self.collection.get(where={"source": source}, limit=1, include=[])
            # ChromaDB doesn't have a count-with-filter, so we use a workaround
            result = self.collection.get(where={"source": source}, include=[])
            count = len(result["ids"])
            if count > 0:
                print(f"  {source:15s} | {count} chunks")

        bug_count = self.bugs_collection.count()
        print(f"  {'unresolved_bugs':15s} | {bug_count} entries")

    def clear(self, source: Optional[str] = None):
        """Clear indexed data."""
        if source:
            # Get all IDs for this source and delete them
            result = self.collection.get(where={"source": source}, include=[])
            if result["ids"]:
                self.collection.delete(ids=result["ids"])
            print(f"Cleared all chunks for source: {source}")
        else:
            self.client.delete_collection(CODE_COLLECTION)
            self.collection = self.client.get_or_create_collection(
                name=CODE_COLLECTION,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            print("Cleared all indexed chunks.")


# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index codebases into ChromaDB for RAG")
    parser.add_argument("--aligntune-path", type=str, help="Path to aligntune repo")
    parser.add_argument("--trl-path", type=str, help="Path to trl repo")
    parser.add_argument("--unsloth-path", type=str, help="Path to unsloth repo")
    parser.add_argument("--api-docs", action="store_true", help="Index api_docs.py")
    parser.add_argument("--clear", action="store_true", help="Clear all indexed data first")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    args = parser.parse_args()

    indexer = RAGIndexer()

    if args.clear:
        indexer.clear()

    if args.aligntune_path:
        indexer.index_repo(args.aligntune_path, "aligntune")

    if args.trl_path:
        indexer.index_repo(args.trl_path, "trl")

    if args.unsloth_path:
        indexer.index_repo(args.unsloth_path, "unsloth")

    if args.api_docs:
        from api_docs import ALIGNTUNE_API_DOCS
        indexer.index_api_docs(ALIGNTUNE_API_DOCS, "api_docs")

    if args.stats:
        indexer.get_stats()