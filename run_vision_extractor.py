"""
Run Strategy C — Vision-Augmented extraction on a PDF.

Requires OPENROUTER_API_KEY in the environment. Uses budget guard to cap
token spend per document (default 500k tokens).
"""
import asyncio
import os
from pathlib import Path

from src.services.triage_services.artifact_loader import DocumentArtifacts
from src.strategies import VisionExtractor


async def main(pdf_path: str) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY to run the vision extractor.")
        return

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    artifacts = DocumentArtifacts(str(path))
    extractor = VisionExtractor(
        model_id="google/gemini-2.0-flash-001",
        budget_cap_tokens=500_000,
    )

    doc = await extractor.extract(
        artifacts=artifacts,
        doc_id=path.name,
        trigger_reason="scanned_image",
    )

    meta = doc.metadata or {}
    print(f"PDF: {path}")
    print(f"Text blocks: {len(doc.text_blocks)}")
    print(f"Tables: {len(doc.tables)}")
    print(f"Figures: {len(doc.figures)}")
    print(f"Reading order entries: {len(doc.reading_order)}")
    print(f"Usage: input_tokens={meta.get('input_tokens')} output_tokens={meta.get('output_tokens')} estimated_cost_usd={meta.get('estimated_cost_usd')}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/RepoInvestigator_Architecture_Report.pdf"
    asyncio.run(main(target))
