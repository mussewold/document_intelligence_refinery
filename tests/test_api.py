"""
Integration tests for the FastAPI server endpoints.
"""
from __future__ import annotations

import json
import pytest

from fastapi.testclient import TestClient

from src.api.server import app, _doc_store
from src.models.ldu import LDU
from src.models.extracted_document import ExtractedDocument, TextBlock
from src.models.provenance import BoundingBox


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_doc_store():
    """Reset the in-memory doc store between tests."""
    _doc_store.clear()
    yield
    _doc_store.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _seed_doc():
    """Insert a synthetic document into the in-memory store."""
    ldus = [
        LDU(
            id="ldu_0",
            chunk_type="text",
            page_no=1,
            content="Revenue was $4.2B in Q3 2024",
            token_count=7,
        ),
        LDU(
            id="ldu_1",
            chunk_type="text",
            page_no=2,
            content="Operating income reached $1.1B in Q3 2024",
            token_count=8,
        ),
    ]
    doc = ExtractedDocument(
        doc_id="test_doc.pdf",
        text_blocks=[
            TextBlock(
                text="Revenue was $4.2B in Q3 2024",
                page_no=1,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20),
                label="paragraph",
            )
        ],
        reading_order=["text_0"],
    )
    _doc_store["test_doc.pdf"] = {"ldus": ldus, "doc": doc, "page_index_root": None}


# ── Tests: Health ─────────────────────────────────────────────

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Tests: Documents ──────────────────────────────────────────

def test_list_documents_empty(client):
    r = client.get("/api/documents")
    assert r.status_code == 200
    # May include profiles from disk but should be a list
    assert isinstance(r.json(), list)


def test_list_documents_with_seeded(client):
    _seed_doc()
    r = client.get("/api/documents")
    assert r.status_code == 200
    ids = [d["doc_id"] for d in r.json()]
    assert "test_doc.pdf" in ids


# ── Tests: Query ─────────────────────────────────────────────

def test_query_no_documents(client):
    r = client.post("/api/query", json={"question": "revenue"})
    assert r.status_code == 404


def test_query_returns_answer(client):
    _seed_doc()
    r = client.post("/api/query", json={"question": "revenue", "doc_id": "test_doc.pdf"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)


def test_query_returns_provenance_chain(client):
    _seed_doc()
    r = client.post("/api/query", json={"question": "revenue"})
    assert r.status_code == 200
    data = r.json()
    assert "provenance_chain" in data
    assert isinstance(data["provenance_chain"], list)


def test_query_provenance_citation_schema(client):
    _seed_doc()
    r = client.post("/api/query", json={"question": "revenue Q3 2024"})
    assert r.status_code == 200
    data = r.json()
    for c in data["provenance_chain"]:
        assert "document_name" in c
        assert "page_number" in c
        assert isinstance(c["page_number"], int)


def test_query_returns_tool_trace(client):
    _seed_doc()
    r = client.post("/api/query", json={"question": "revenue"})
    assert r.status_code == 200
    data = r.json()
    assert "tool_trace" in data
    assert isinstance(data["tool_trace"], list)


# ── Tests: Audit ─────────────────────────────────────────────

def test_audit_no_documents(client):
    r = client.post("/api/audit", json={"claim": "revenue was $4.2B"})
    assert r.status_code == 404


def test_audit_returns_verified_field(client):
    _seed_doc()
    r = client.post("/api/audit", json={"claim": "revenue was $4.2B"})
    assert r.status_code == 200
    data = r.json()
    assert "verified" in data
    assert data["verified"] is not None  # must be True or False, not None after audit


def test_audit_returns_answer(client):
    _seed_doc()
    r = client.post("/api/audit", json={"claim": "revenue Q3 2024"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_audit_unverifiable_returns_false(client):
    _seed_doc()
    r = client.post("/api/audit", json={"claim": "xyzzy zork gobbledegook qqqqqq"})
    assert r.status_code == 200
    data = r.json()
    # Should return False for a nonsense claim
    assert data["verified"] == False


# ── Tests: Facts ─────────────────────────────────────────────

def test_facts_empty(client):
    r = client.get("/api/facts")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_facts_filter_by_key(client):
    r = client.get("/api/facts?key=revenue")
    assert r.status_code == 200
    for row in r.json():
        assert "revenue" in row["key"].lower()
