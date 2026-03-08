"""
Run full extraction pipeline: triage → ExtractionRouter → extractors.

Usage:
  uv run python run_extraction.py [path/to/doc.pdf]

Uses default PDF under ./data/ if no path is given. Results are logged to
.refinery/extraction_ledger.jsonl.
"""
import asyncio
import json
import sys
from pathlib import Path

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent
from src.services.triage_services.artifact_loader import DocumentArtifacts
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder, query_page_index


async def main(pdf_path: str) -> None:
    path = Path(pdf_path)
    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Triage: {path}")
    triage = TriageAgent()
    profile = await triage.analyze(str(path))
    print(f"  origin_type={getattr(profile, 'origin_type', '?')} layout_complexity={profile.layout_complexity} cost={profile.estimated_extraction_cost}")

    artifacts = DocumentArtifacts(str(path))
    router = ExtractionRouter(confidence_threshold=0.7)

    print("Extracting...")
    result = await router.extract(profile, artifacts)
    doc = result.document
    print(f"  strategy_used={result.strategy_used} escalated={result.escalated}")
    print(f"  text_blocks={len(doc.text_blocks)} tables={len(doc.tables)} figures={len(doc.figures)}")

    # ---- Stage 3: ChunkingEngine -> LDUs ----
    engine = ChunkingEngine()
    ldus = engine.build_ldus(doc)
    print(f"ChunkingEngine: produced {len(ldus)} LDUs")
    if ldus:
        first = ldus[0]
        print("  First LDU:", first.id, first.chunk_type, first.page_no, f"tokens={first.token_count}")

    # ---- PageIndex: build tree and test a query ----
    builder = PageIndexBuilder()
    root = await builder.build(doc, ldus)
    print(f"PageIndex root: {root.title} pages {root.page_start}-{root.page_end}")
    print(f"  child_sections={len(root.child_sections)}")

    topic = "capital expenditure projections for Q3"
    sections = query_page_index(root, topic, top_k=3)
    print(f"Top sections for topic '{topic}':")
    for s in sections:
        print(f"  - {s.title} (pages {s.page_start}-{s.page_end})")

    # ---- Ledger: show last entry ----
    ledger = Path(".refinery/extraction_ledger.jsonl")
    if ledger.is_file():
        with open(ledger, encoding="utf-8") as f:
            lines = f.readlines()
        if lines:
            last = json.loads(lines[-1])
            print("Ledger (last entry):", json.dumps(last, indent=2))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/RepoInvestigator_Architecture_Report.pdf"
    asyncio.run(main(target))
