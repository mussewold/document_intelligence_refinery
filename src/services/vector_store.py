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

from ..models.ldu import LDU


@dataclass
class VectorHit:
    ldu_id: str
    score: float
    metadata: Dict[str, Any]


class InMemorySemanticStore:
    """A tiny TF-style store suitable for local testing.

    - add_ldus(ldus): indexes their content
    - search(query, k): returns best-matching LDUs
    """

    def __init__(self) -> None:
        self._docs: Dict[str, str] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._vocab: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in text.split() if t]

    def _build_vocab(self) -> None:
        vocab: Dict[str, int] = {}
        for text in self._docs.values():
            for tok in set(self._tokenize(text)):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._vocab = vocab

    def add_ldus(self, ldus: Sequence[LDU]) -> None:
        for l in ldus:
            self._docs[l.id] = l.content or ""
            self._meta[l.id] = {
                "page_no": l.page_no,
                "chunk_type": l.chunk_type,
                "content_hash": l.content_hash,
            }
        self._build_vocab()

    def _vectorize(self, text: str) -> Dict[int, float]:
        tokens = self._tokenize(text)
        counts: Dict[int, float] = {}
        for tok in tokens:
            if tok not in self._vocab:
                continue
            idx = self._vocab[tok]
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return counts

    def _cosine(self, a: Dict[int, float], b: Dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a.get(i, 0.0) * b.get(i, 0.0) for i in set(a) | set(b))
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def search(self, query: str, k: int = 5) -> List[VectorHit]:
        q_vec = self._vectorize(query)
        hits: List[Tuple[float, str]] = []
        for ldu_id, text in self._docs.items():
            d_vec = self._vectorize(text)
            score = self._cosine(q_vec, d_vec)
            if score > 0.0:
                hits.append((score, ldu_id))
        hits.sort(key=lambda x: x[0], reverse=True)
        out: List[VectorHit] = []
        for score, ldu_id in hits[:k]:
            out.append(VectorHit(ldu_id=ldu_id, score=score, metadata=self._meta.get(ldu_id, {})))
        return out

