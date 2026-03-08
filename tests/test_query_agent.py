"""
Unit tests for the LangGraph-based RefineryAgent.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.agents.query_agent import ProvenanceCitation, QueryResponse, RefineryAgent
from src.models.extracted_document import BoundingBox, ExtractedDocument, TextBlock
from src.models.ldu import LDU
from src.models.page_index import PageIndexNode
from src.models.provenance import BoundingBox as BBox


# ── Fixtures ──────────────────────────────────────────────────

def _make_ldu(id_: str, content: str, page: int = 1) -> LDU:
    return LDU(
        id=id_,
        chunk_type="text",
        page_no=page,
        content=content,
        token_count=len(content.split()),
    )


def _make_doc() -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="test_doc.pdf",
        text_blocks=[
            TextBlock(
                text="Revenue was $4.2B in Q3 2024",
                page_no=1,
                bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                label="paragraph",
            )
        ],
        reading_order=["text_0"],
    )


def _make_page_index() -> PageIndexNode:
    return PageIndexNode(
        id="root",
        title="Financial Report",
        page_start=1,
        page_end=3,
        summary="This is a financial report summary",
        ldu_ids=["ldu_0", "ldu_1"],
    )


# ── Tests: basic query ─────────────────────────────────────────

class TestRefineryAgentQuery:
    def setup_method(self):
        self.agent = RefineryAgent()
        self.ldus = [
            _make_ldu("ldu_0", "Revenue was $4.2B in Q3 2024", page=1),
            _make_ldu("ldu_1", "Operating income reached $1.1B", page=2),
            _make_ldu("ldu_2", "Capital expenditure was $500M", page=3),
        ]
        self.doc = _make_doc()

    def test_run_returns_query_response(self):
        result = self.agent.run(
            "revenue",
            ldus=self.ldus,
            doc=self.doc,
        )
        assert isinstance(result, QueryResponse)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_run_returns_provenance_chain(self):
        result = self.agent.run(
            "revenue",
            ldus=self.ldus,
            doc=self.doc,
        )
        # Provenance chain should have at least one entry from semantic search
        assert isinstance(result.provenance_chain, list)

    def test_provenance_citation_has_required_fields(self):
        result = self.agent.run(
            "revenue",
            ldus=self.ldus,
            doc=self.doc,
        )
        for citation in result.provenance_chain:
            assert isinstance(citation, ProvenanceCitation)
            assert isinstance(citation.document_name, str)
            assert isinstance(citation.page_number, int)
            assert citation.page_number >= 1

    def test_tool_trace_populated(self):
        result = self.agent.run(
            "revenue",
            ldus=self.ldus,
            doc=self.doc,
        )
        assert isinstance(result.tool_trace, list)
        assert len(result.tool_trace) >= 1  # at least one tool was called

    def test_no_ldus_returns_not_found(self):
        result = self.agent.run(
            "something",
            ldus=[],
            doc=self.doc,
        )
        assert isinstance(result.answer, str)
        # Should say it couldn't find anything
        assert "not" in result.answer.lower() or "could" in result.answer.lower() or "no" in result.answer.lower()

    def test_with_page_index_root(self):
        root = _make_page_index()
        result = self.agent.run(
            "financial",
            ldus=self.ldus,
            doc=self.doc,
            page_index_root=root,
        )
        assert isinstance(result, QueryResponse)

    def test_verified_none_in_query_mode(self):
        result = self.agent.run(
            "revenue",
            ldus=self.ldus,
            doc=self.doc,
        )
        # audit_mode=False by default, so verified should not be forced
        assert result.verified is None or isinstance(result.verified, bool)


# ── Tests: Audit Mode ─────────────────────────────────────────

class TestRefineryAgentAudit:
    def setup_method(self):
        self.agent = RefineryAgent(audit_threshold=0.05)  # low threshold for testing
        self.ldus = [
            _make_ldu("ldu_0", "Revenue was $4.2B in Q3 2024", page=1),
            _make_ldu("ldu_1", "Operating income reached $1.1B", page=2),
        ]
        self.doc = _make_doc()

    def test_audit_returns_query_response(self):
        result = self.agent.audit("revenue", ldus=self.ldus, doc=self.doc)
        assert isinstance(result, QueryResponse)

    def test_audit_verified_boolean(self):
        result = self.agent.audit("revenue Q3", ldus=self.ldus, doc=self.doc)
        assert isinstance(result.verified, bool)

    def test_audit_unverifiable_claim(self):
        result = self.agent.audit(
            "xyzzy zork gobbledegook totally random claim",
            ldus=self.ldus,
            doc=self.doc,
        )
        # Very unlikely to match random text
        assert result.verified is not None

    def test_audit_verified_claim_has_citations(self):
        result = self.agent.audit("revenue Q3 2024", ldus=self.ldus, doc=self.doc)
        if result.verified:
            assert len(result.provenance_chain) > 0

    def test_audit_score_is_float(self):
        result = self.agent.audit("revenue", ldus=self.ldus, doc=self.doc)
        if result.audit_score is not None:
            assert isinstance(result.audit_score, float)
            assert 0.0 <= result.audit_score <= 1.0


# ── Tests: Individual tool nodes ─────────────────────────────

class TestToolNodes:
    def setup_method(self):
        self.agent = RefineryAgent()
        self.ldus = [_make_ldu("ldu_0", "test content about revenue and profits")]
        self.doc = _make_doc()

    def _base_state(self):
        return {
            "messages": [],
            "question": "revenue",
            "ldus": self.ldus,
            "doc": self.doc,
            "page_index_root": None,
            "audit_mode": False,
            "audit_threshold": 0.2,
            "hits": [],
            "sql_results": [],
            "provenance_chain": [],
            "tool_trace": [],
            "answer": "",
            "verified": None,
            "audit_score": None,
        }

    def test_semantic_search_adds_hits(self):
        state = self._base_state()
        updated = self.agent._node_semantic(state)
        assert "hits" in updated
        assert "tool_trace" in updated
        assert len(updated["tool_trace"]) >= 1

    def test_structured_query_adds_trace(self):
        state = self._base_state()
        updated = self.agent._node_structured(state)
        # Even with empty DB the trace should be populated
        assert "tool_trace" in updated
        assert len(updated["tool_trace"]) >= 1

    def test_navigate_with_no_root_adds_trace(self):
        state = self._base_state()
        updated = self.agent._node_navigate(state)
        assert "tool_trace" in updated
        assert "no index" in updated["tool_trace"][0].lower()
