"""
FastAPI server for the Document Intelligence Refinery.

Endpoints
---------
GET  /                   Serve the frontend SPA
POST /api/upload         Upload a PDF and run the full extraction pipeline
GET  /api/documents      List processed documents (profiles from .refinery/profiles/)
POST /api/query          Answer a question using the LangGraph agent
POST /api/audit          Verify a claim using Audit Mode
GET  /api/facts          Return rows from the facts SQLite table
GET  /api/health         Health check
"""
from __future__ import annotations

import json
import logging
import sqlite3
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agents.query_agent import ProvenanceCitation, QueryResponse, RefineryAgent
from ..agents.triage import TriageAgent
from ..models.ldu import LDU
from ..agents.chunker import ChunkingEngine
from ..services.fact_table_extractor import DEFAULT_DB_PATH, FactTableExtractor
from ..agents.indexer import PageIndexBuilder
from ..services.triage_services.artifact_loader import DocumentArtifacts

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
FRONTEND_DIR = BASE_DIR / "frontend"
PROFILES_DIR = BASE_DIR / ".refinery" / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="Document Intelligence Refinery",
    description="LangGraph-powered document query and audit API",
    version="1.0.0",
)

# ── Shared singletons ─────────────────────────────────────────────
_triage_agent = TriageAgent()
_fact_extractor = FactTableExtractor()
_refinery_agent = RefineryAgent(fact_extractor=_fact_extractor)
_chunking_engine = ChunkingEngine()


# ── In-memory document store (per-session) ────────────────────────
# Maps doc_id -> {"ldus": [], "doc": ExtractedDocument, "page_index_root": PageIndexNode}
_doc_store: Dict[str, Dict[str, Any]] = {}


# ── Request / Response models ─────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None  # restrict to a specific document context


class AuditRequest(BaseModel):
    claim: str
    doc_id: Optional[str] = None


class FactRecord(BaseModel):
    id: int
    doc_id: str
    ldu_id: str
    page_no: int
    key: str
    value: str
    unit: Optional[str]
    period: Optional[str]
    content_hash: Optional[str]


# ── Helper: gather LDUs for a doc (or all docs) ───────────────────

def _get_ldus(doc_id: Optional[str]) -> tuple:
    """Return (ldus, doc, page_index_root) for the given doc_id (or combined)."""
    if doc_id and doc_id in _doc_store:
        entry = _doc_store[doc_id]
        return entry["ldus"], entry.get("doc"), entry.get("page_index_root")
    # If no specific doc, pool all LDUs
    all_ldus: List[LDU] = []
    for entry in _doc_store.values():
        all_ldus.extend(entry["ldus"])
    return all_ldus, None, None


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a PDF, run triage + extraction + chunking + fact extraction."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to a temp location
    upload_dir = BASE_DIR / ".refinery" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / file.filename

    contents = await file.read()
    pdf_path.write_bytes(contents)

    try:
        # Step 1: Triage
        profile = await _triage_agent.analyze(str(pdf_path))
        doc_id = pdf_path.name

        # Step 2: Load artifacts
        artifacts = DocumentArtifacts(str(pdf_path))

        # Step 3: Extract document using the ExtractionRouter
        from ..agents.extractor import ExtractionRouter
        router = ExtractionRouter()
        extraction_result = await router.extract(profile, artifacts, doc_id=doc_id)
        extracted_doc = extraction_result.document

        # Step 4: Chunk into LDUs
        ldus = _chunking_engine.build_ldus(extracted_doc)

        # Step 5: Build PageIndex
        pib = PageIndexBuilder()
        page_index_root = await pib.build(extracted_doc, ldus)

        # Step 6: Extract facts
        n_facts = _fact_extractor.extract_and_store(doc_id, ldus)

        # Store in memory
        _doc_store[doc_id] = {
            "ldus": ldus,
            "doc": extracted_doc,
            "page_index_root": page_index_root,
            "profile": profile,
        }

        return {
            "doc_id": doc_id,
            "strategy": extraction_result.strategy_used,
            "escalated": extraction_result.escalated,
            "ldu_count": len(ldus),
            "fact_count": n_facts,
            "profile": profile.model_dump(),
        }

    except Exception as exc:
        logger.error("Upload failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")


@app.get("/api/documents")
async def list_documents() -> List[Dict[str, Any]]:
    """List processed document profiles."""
    docs = []
    # First from in-memory store
    for doc_id, entry in _doc_store.items():
        profile = entry.get("profile")
        docs.append({
            "doc_id": doc_id,
            "ldu_count": len(entry.get("ldus", [])),
            "profile": profile.model_dump() if profile else {},
        })
    # Also from profiles on disk (from previous sessions)
    if PROFILES_DIR.exists():
        for p in sorted(PROFILES_DIR.glob("*.json")):
            name = p.stem  # filename without .json
            if name not in _doc_store:
                try:
                    data = json.loads(p.read_text())
                    docs.append({"doc_id": name, "ldu_count": 0, "profile": data})
                except Exception:
                    continue
    return docs


@app.post("/api/query")
async def query_document(req: QueryRequest) -> Dict[str, Any]:
    """Answer a question using the LangGraph refinery agent."""
    ldus, doc, page_index_root = _get_ldus(req.doc_id)
    if not ldus:
        raise HTTPException(
            status_code=404,
            detail="No documents loaded. Please upload a PDF first.",
        )
    result: QueryResponse = _refinery_agent.run(
        req.question,
        ldus=ldus,
        doc=doc,
        page_index_root=page_index_root,
        audit_mode=False,
    )
    return result.model_dump()


@app.post("/api/audit")
async def audit_claim(req: AuditRequest) -> Dict[str, Any]:
    """Verify or refute a claim using Audit Mode."""
    ldus, doc, page_index_root = _get_ldus(req.doc_id)
    if not ldus:
        raise HTTPException(
            status_code=404,
            detail="No documents loaded. Please upload a PDF first.",
        )
    result: QueryResponse = _refinery_agent.audit(
        req.claim,
        ldus=ldus,
        doc=doc,
        page_index_root=page_index_root,
    )
    return result.model_dump()


@app.get("/api/facts")
async def get_facts(
    doc_id: Optional[str] = Query(default=None),
    key: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=1000),
) -> List[Dict[str, Any]]:
    """Return rows from the facts SQLite table."""
    if not DEFAULT_DB_PATH.exists():
        return []

    conn = sqlite3.connect(DEFAULT_DB_PATH)
    try:
        clauses = []
        params: List[Any] = []
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if key:
            clauses.append("key LIKE ?")
            params.append(f"%{key}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT id, doc_id, ldu_id, page_no, key, value, unit, period, content_hash FROM facts {where} LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    cols = ["id", "doc_id", "ldu_id", "page_no", "key", "value", "unit", "period", "content_hash"]
    return [dict(zip(cols, row)) for row in rows]


# ── Static files (must come AFTER API routes) ─────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{path:path}")
    async def serve_static(path: str) -> FileResponse:
        file_path = FRONTEND_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Dev entry-point ───────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
