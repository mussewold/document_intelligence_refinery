"""
LangGraph-based Query Agent for Document Intelligence Refinery.

Three retrieval tools:
  1. pageindex_navigate  – hierarchical tree traversal
  2. semantic_search     – TF-IDF cosine similarity over LDUs
  3. structured_query    – SQL SELECT over the SQLite facts table

Every tool call emits ProvenanceCitation entries so the caller always
receives a full ProvenanceChain alongside the answer.

Audit mode: given a claim, verify against the document or flag as
"not found / unverifiable".
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from ..models.extracted_document import ExtractedDocument
from ..models.ldu import LDU
from ..models.page_index import PageIndexNode
from ..models.provenance import BoundingBox
from ..services.fact_table_extractor import FactTableExtractor
from ..agents.indexer import query_page_index
from ..services.vector_store import ChromaSemanticStore, VectorHit

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────

class ProvenanceCitation(BaseModel):
    """Single source citation attached to an answer or audit result."""
    document_name: str
    page_number: int
    bbox: Optional[BoundingBox] = None
    content_hash: Optional[str] = None
    snippet: Optional[str] = None
    tool_used: str = "unknown"


class QueryResponse(BaseModel):
    """Final response from the agent."""
    answer: str
    provenance_chain: List[ProvenanceCitation] = Field(default_factory=list)
    tool_trace: List[str] = Field(default_factory=list)
    verified: Optional[bool] = None  # set in audit mode
    audit_score: Optional[float] = None


# ──────────────────────────────────────────────────────────────────
# LangGraph State
# ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    ldus: List[LDU]
    doc: Optional[ExtractedDocument]
    page_index_root: Optional[PageIndexNode]
    audit_mode: bool
    audit_threshold: float
    # Accumulated results
    hits: List[VectorHit]
    sql_results: List[tuple]
    provenance_chain: List[ProvenanceCitation]
    tool_trace: List[str]
    answer: str
    verified: Optional[bool]
    audit_score: Optional[float]


# ──────────────────────────────────────────────────────────────────
# Core Agent class
# ──────────────────────────────────────────────────────────────────

class RefineryAgent:
    """
    LangGraph agent exposing three retrieval tools and an optional audit mode.

    Usage
    -----
    agent = RefineryAgent()
    result: QueryResponse = agent.run(
        question="What was the revenue in Q3?",
        ldus=ldus,
        doc=doc,
        page_index_root=root,
    )
    """

    def __init__(
        self,
        fact_extractor: Optional[FactTableExtractor] = None,
        semantic_k: int = 5,
        audit_threshold: float = 0.20,
    ) -> None:
        self.fact_extractor = fact_extractor or FactTableExtractor()
        self.semantic_k = semantic_k
        self.audit_threshold = audit_threshold
        self._graph = self._build_graph()

    # ── Tool 1: pageindex_navigate ────────────────────────────────

    def _pageindex_navigate(
        self,
        state: AgentState,
        topic: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        root = state.get("page_index_root")
        if root is None:
            return {"tool_trace": state["tool_trace"] + ["pageindex_navigate: no index"]}

        sections = query_page_index(root, topic, top_k=top_k)
        restrict_ids: List[str] = []
        for s in sections:
            restrict_ids.extend(s.ldu_ids)

        citations = [
            ProvenanceCitation(
                document_name=state["doc"].doc_id if state.get("doc") else "unknown",
                page_number=s.page_start,
                snippet=s.summary or s.title,
                tool_used="pageindex_navigate",
            )
            for s in sections
        ]
        return {
            "hits": state["hits"] + [],  # no semantic hits from this step
            "provenance_chain": state["provenance_chain"] + citations,
            "tool_trace": state["tool_trace"] + [
                f"pageindex_navigate: found {len(sections)} sections for '{topic}'"
            ],
            "_restrict_ldu_ids": restrict_ids,
        }

    # ── Tool 2: semantic_search ───────────────────────────────────

    def _semantic_search(
        self,
        state: AgentState,
        query: str,
        restrict_ldu_ids: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        ldus = state["ldus"]
        doc = state.get("doc")
        doc_id = doc.doc_id if doc else "unknown"

        store = ChromaSemanticStore()
        if restrict_ldu_ids:
            subset = [l for l in ldus if l.id in set(restrict_ldu_ids)]
        else:
            subset = ldus
        store.add_ldus(subset)

        hits = store.search(query, k or self.semantic_k)
        ldus_by_id = {l.id: l for l in ldus}

        citations: List[ProvenanceCitation] = []
        for h in hits:
            ldu = ldus_by_id.get(h.ldu_id)
            if ldu:
                citations.append(
                    ProvenanceCitation(
                        document_name=doc_id,
                        page_number=ldu.page_no,
                        bbox=ldu.bbox,
                        content_hash=ldu.content_hash,
                        snippet=(ldu.content or "")[:300],
                        tool_used="semantic_search",
                    )
                )
        return {
            "hits": state["hits"] + hits,
            "provenance_chain": state["provenance_chain"] + citations,
            "tool_trace": state["tool_trace"] + [
                f"semantic_search: {len(hits)} hits for '{query}'"
            ],
        }

    # ── Tool 3: structured_query ──────────────────────────────────

    def _structured_query(
        self,
        state: AgentState,
        sql: str,
    ) -> Dict[str, Any]:
        try:
            rows = self.fact_extractor.structured_query(sql)
        except Exception as exc:
            return {
                "sql_results": state["sql_results"],
                "tool_trace": state["tool_trace"] + [f"structured_query ERROR: {exc}"],
            }

        citations: List[ProvenanceCitation] = []
        for row in rows:
            # row: (id, doc_id, ldu_id, page_no, key, value, unit, period, content_hash)
            try:
                _, doc_id, _, page_no, key, value, unit, period, content_hash = row
                snippet = f"{key}: {value} {unit or ''} {period or ''}".strip()
                citations.append(
                    ProvenanceCitation(
                        document_name=str(doc_id),
                        page_number=int(page_no),
                        content_hash=str(content_hash) if content_hash else None,
                        snippet=snippet,
                        tool_used="structured_query",
                    )
                )
            except (TypeError, ValueError):
                continue

        return {
            "sql_results": state["sql_results"] + list(rows),
            "provenance_chain": state["provenance_chain"] + citations,
            "tool_trace": state["tool_trace"] + [
                f"structured_query: {len(rows)} rows for SQL"
            ],
        }

    # ── Graph nodes ───────────────────────────────────────────────

    def _node_navigate(self, state: AgentState) -> AgentState:
        question = state["question"]
        update = self._pageindex_navigate(state, topic=question)
        return {**state, **update}

    def _node_semantic(self, state: AgentState) -> AgentState:
        question = state["question"]
        restrict = getattr(state, "_restrict_ldu_ids", None)  # type: ignore[attr-defined]
        # Extract from previous navigate step stored in state dict
        restrict = state.get("_restrict_ldu_ids")  # type: ignore[call-overload]
        update = self._semantic_search(state, query=question, restrict_ldu_ids=restrict)
        return {**state, **update}

    def _node_structured(self, state: AgentState) -> AgentState:
        question = state["question"]
        # Build a simple SQL from the question keywords
        keywords = [w.lower() for w in question.split() if len(w) > 3]
        if keywords:
            like_clause = " OR ".join(f"key LIKE '%{kw}%'" for kw in keywords[:3])
            sql = f"SELECT * FROM facts WHERE {like_clause} LIMIT 20"
        else:
            sql = "SELECT * FROM facts LIMIT 20"
        update = self._structured_query(state, sql=sql)
        return {**state, **update}

    def _node_synthesise(self, state: AgentState) -> AgentState:
        hits = state.get("hits", [])
        ldus_by_id = {l.id: l for l in state.get("ldus", [])}
        sql_results = state.get("sql_results", [])
        audit_mode = state.get("audit_mode", False)
        threshold = state.get("audit_threshold", self.audit_threshold)

        answer = ""
        verified: Optional[bool] = None
        audit_score: Optional[float] = None

        if hits:
            best = hits[0]
            best_ldu = ldus_by_id.get(best.ldu_id)
            audit_score = best.score

            if audit_mode:
                if best.score >= threshold:
                    verified = True
                    content = best_ldu.content if best_ldu else ""
                    answer = f"✓ VERIFIED — supported by document content:\n\n{content}"
                else:
                    verified = False
                    answer = "⚠ UNVERIFIABLE — no sufficiently similar passage found in the document."
            else:
                content = best_ldu.content if best_ldu else ""
                answer = content or "Relevant content found but no text available."

        elif sql_results:
            # Compose answer from fact rows
            parts = []
            for row in sql_results[:5]:
                try:
                    _, doc_id, _, page_no, key, value, unit, period, _ = row
                    parts.append(f"• {key}: {value} {unit or ''} ({period or 'N/A'})")
                except (TypeError, ValueError):
                    parts.append(str(row))
            answer = "Found the following facts from structured data:\n" + "\n".join(parts)
            if audit_mode:
                verified = True
                audit_score = 1.0

        else:
            if audit_mode:
                verified = False
                answer = "⚠ UNVERIFIABLE — claim not found in the document."
            else:
                answer = "I could not find any relevant content in the document."

        msg = AIMessage(content=answer)
        return {
            **state,
            "messages": [msg],
            "answer": answer,
            "verified": verified,
            "audit_score": audit_score,
        }

    # ── Graph assembly ────────────────────────────────────────────

    def _build_graph(self) -> Any:
        builder: StateGraph = StateGraph(AgentState)

        builder.add_node("navigate", self._node_navigate)
        builder.add_node("semantic", self._node_semantic)
        builder.add_node("structured", self._node_structured)
        builder.add_node("synthesise", self._node_synthesise)

        builder.add_edge(START, "navigate")
        builder.add_edge("navigate", "semantic")
        builder.add_edge("semantic", "structured")
        builder.add_edge("structured", "synthesise")
        builder.add_edge("synthesise", END)

        return builder.compile()

    # ── Public API ────────────────────────────────────────────────

    def run(
        self,
        question: str,
        *,
        ldus: List[LDU],
        doc: Optional[ExtractedDocument] = None,
        page_index_root: Optional[PageIndexNode] = None,
        audit_mode: bool = False,
    ) -> QueryResponse:
        """Synchronous entry point for the LangGraph agent."""
        initial: AgentState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "ldus": ldus,
            "doc": doc,
            "page_index_root": page_index_root,
            "audit_mode": audit_mode,
            "audit_threshold": self.audit_threshold,
            "hits": [],
            "sql_results": [],
            "provenance_chain": [],
            "tool_trace": [],
            "answer": "",
            "verified": None,
            "audit_score": None,
        }
        final = self._graph.invoke(initial)
        return QueryResponse(
            answer=final["answer"],
            provenance_chain=final["provenance_chain"],
            tool_trace=final["tool_trace"],
            verified=final.get("verified"),
            audit_score=final.get("audit_score"),
        )

    def audit(
        self,
        claim: str,
        *,
        ldus: List[LDU],
        doc: Optional[ExtractedDocument] = None,
        page_index_root: Optional[PageIndexNode] = None,
    ) -> QueryResponse:
        """Audit mode shortcut."""
        return self.run(
            claim,
            ldus=ldus,
            doc=doc,
            page_index_root=page_index_root,
            audit_mode=True,
        )
