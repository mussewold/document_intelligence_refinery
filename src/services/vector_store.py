from __future__ import annotations

"""
Simple in-memory semantic store for LDUs.

This is intentionally lightweight: it stores text and uses a simple
bag-of-words cosine similarity as a stand-in for true vector search.
You can later plug in real embeddings by providing pre-computed
vectors instead.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import math
import os
from pathlib import Path

from ..models.ldu import LDU

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


@dataclass
class VectorHit:
    ldu_id: str
    score: float
    metadata: Dict[str, Any]


class ChromaSemanticStore:
    """A persistent semantic store using ChromaDB."""

    def __init__(self, persist_dir: str = ".refinery/chroma_db") -> None:
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        if chromadb is None:
            raise ImportError("chromadb is not installed. Run `uv add chromadb`.")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="refinery_ldus",
            metadata={"hnsw:space": "cosine"}
        )

    def add_ldus(self, ldus: Sequence[LDU]) -> None:
        if not ldus:
            return

        ids = []
        documents = []
        metadatas = []

        for l in ldus:
            ids.append(l.id)
            documents.append(l.content or "")
            # ChromaDB metadatas cannot contain None or complex objects easily
            meta = {
                "page_no": l.page_no,
                "chunk_type": l.chunk_type,
            }
            if l.content_hash:
                meta["content_hash"] = l.content_hash
            metadatas.append(meta)

        # Batch upsert to handle updates gracefully
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def search(self, query: str, k: int = 5) -> List[VectorHit]:
        if not query.strip():
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["metadatas", "distances"]
            )
        except Exception:
            return []

        out: List[VectorHit] = []
        if not results["ids"] or not results["ids"][0]:
            return out

        # results is a dict of lists of lists. First element is for the single query.
        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(ids)

        for ldu_id, dist, meta in zip(ids, distances, metas):
            # Convert cosine distance back to similarity score (1.0 is exact match, distance 0.0)
            score = max(0.0, 1.0 - dist)
            # Only return actual hits reasonably similar
            if score > 0.05:
                out.append(VectorHit(ldu_id=ldu_id, score=score, metadata=meta))

        # Sort by best score descending
        out.sort(key=lambda x: x.score, reverse=True)
        return out

