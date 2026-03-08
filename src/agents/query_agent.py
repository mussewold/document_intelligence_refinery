from __future__ import annotations

"""
Query Interface Agent (front-end of the refinery).

Implements three conceptual tools:
  - pageindex_navigate (tree traversal over PageIndexNode)
  - semantic_search (vector/semantic retrieval over LDUs)
  - structured_query (SQL over extracted fact tables)

Every answer includes provenance via ProvenanceChain entries.
Audit mode: verify or refute claims with citations.

This module is written so that each public method can be wrapped as a
LangGraph tool, even though we don't depend on LangGraph directly here.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.extracted_document import ExtractedDocument
from ..models.ldu import LDU
from ..models.page_index import PageIndexNode
from ..models.provenance import BoundingBox
from ..models.provenance_chain import ProvenanceChain, ProvenanceStep
from ..services.fact_table_extractor import FactTableExtractor
from ..services.page_index_builder import query_page_index
from ..services.vector_store import InMemorySemanticStore, VectorHit


class ProvenanceCitation(BaseModel):
    document_name: str
    page_number: int
    bbox: Optional[BoundingBox] = None
    content_hash: Optional[str] = None
    snippet: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    provenance_chain: List[ProvenanceCitation]


@dataclass
class QueryInterfaceConfig:
    semantic_k: int = 5
    audit_threshold: float = 0.2  # minimal semantic score to consider as support


class QueryInterfaceAgent:
    def __init__(
        self,
        *,
        config: Optional[QueryInterfaceConfig] = None,
        fact_extractor: Optional[FactTableExtractor] = None,
    ) -> None:
        self.config = config or QueryInterfaceConfig()
        self.fact_extractor = fact_extractor or FactTableExtractor()

    # ---- Tool 1: pageindex_navigate -------------------------------------------------

    def pageindex_navigate(
        self,
        root: PageIndexNode,
        topic: str,
        top_k: int = 3,
    ) -> List[PageIndexNode]:
        """
        Navigate the PageIndex tree for a topic and return the top-k sections.
        This is a thin wrapper over query_page_index.
        """
        return query_page_index(root, topic, top_k=top_k)

    # ---- Tool 2: semantic_search ----------------------------------------------------

    def semantic_search(
        self,
        query: str,
        ldus: List[LDU],
        restrict_to_ldu_ids: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> List[VectorHit]:
        """
        Semantic search over LDUs using a simple in-memory semantic store.
        Optionally restrict to a subset of LDU ids.
        """
        store = InMemorySemanticStore()
        if restrict_to_ldu_ids:
            subset = [l for l in ldus if l.id in set(restrict_to_ldu_ids)]
        else:
            subset = ldus
        store.add_ldus(subset)
        return store.search(query, k or self.config.semantic_k)

    # ---- Tool 3: structured_query ---------------------------------------------------

    def structured_query(self, sql: str) -> List[tuple]:
        """
        Execute SQL over the facts table (read-only SELECT).
        """
        return self.fact_extractor.structured_query(sql)

    # ---- Answer & Audit helpers -----------------------------------------------------

    def _citations_from_hits(
        self,
        hits: List[VectorHit],
        ldus_by_id: Dict[str, LDU],
        doc: ExtractedDocument,
    ) -> List[ProvenanceCitation]:
        out: List[ProvenanceCitation] = []
        for h in hits:
            l = ldus_by_id.get(h.ldu_id)
            if not l:
                continue
            out.append(
                ProvenanceCitation(
                    document_name=doc.doc_id,
                    page_number=l.page_no,
                    bbox=None,  # could be added by extending LDU.bbox
                    content_hash=l.content_hash,
                    snippet=(l.content or "")[:300],
                )
            )
        return out

    def answer_question(
        self,
        question: str,
        *,
        doc: ExtractedDocument,
        ldus: List[LDU],
        page_index_root: Optional[PageIndexNode] = None,
    ) -> QueryResponse:
        """
        High-level query:
          1) Optionally use PageIndex to pick best sections.
          2) Semantic search over LDUs (restricted if we have sections).
          3) Return answer + provenance citations.
        """
        ldus_by_id = {l.id: l for l in ldus}

        restrict_ids: Optional[List[str]] = None
        if page_index_root is not None:
            sections = self.pageindex_navigate(page_index_root, question, top_k=3)
            restrict_ids = []
            for s in sections:
                restrict_ids.extend(s.ldu_ids)

        hits = self.semantic_search(question, ldus, restrict_to_ldu_ids=restrict_ids)
        if not hits:
            return QueryResponse(
                answer="I could not find any relevant content in the document.",
                provenance_chain=[],
            )

        # For now, answer is just the best LDU's content.
        best = hits[0]
        best_ldu = ldus_by_id[best.ldu_id]
        answer_text = best_ldu.content or ""
        citations = self._citations_from_hits([best], ldus_by_id, doc)
        return QueryResponse(answer=answer_text, provenance_chain=citations)

    def audit_claim(
        self,
        claim: str,
        *,
        doc: ExtractedDocument,
        ldus: List[LDU],
        page_index_root: Optional[PageIndexNode] = None,
    ) -> QueryResponse:
        """
        Audit Mode: given a claim, either return a supporting citation
        or flag as not found / unverifiable.
        """
        ldus_by_id = {l.id: l for l in ldus}

        restrict_ids: Optional[List[str]] = None
        if page_index_root is not None:
            sections = self.pageindex_navigate(page_index_root, claim, top_k=3)
            restrict_ids = []
            for s in sections:
                restrict_ids.extend(s.ldu_ids)

        hits = self.semantic_search(claim, ldus, restrict_to_ldu_ids=restrict_ids)
        if not hits or hits[0].score < self.config.audit_threshold:
            return QueryResponse(
                answer="Claim not found / unverifiable in the current document.",
                provenance_chain=[],
            )

        best = hits[0]
        best_ldu = ldus_by_id[best.ldu_id]
        answer_text = f"The claim is supported by the document: {best_ldu.content}"
        citations = self._citations_from_hits([best], ldus_by_id, doc)
        return QueryResponse(answer=answer_text, provenance_chain=citations)

