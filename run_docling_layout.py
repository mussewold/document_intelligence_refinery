import asyncio
from pathlib import Path

from src.models.document_profile import LayoutComplexity
from src.services.triage_services.artifact_loader import DocumentArtifacts
from src.strategies import DoclingLayoutExtractor


async def main(pdf_path: str) -> None:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    artifacts = DocumentArtifacts(str(path))
    extractor = DoclingLayoutExtractor()

    # For testing we can pass fixed triage signals; in the full pipeline these
    # would come from TriageAgent.
    layout = await extractor.extract(
        artifacts=artifacts,
        doc_id=path.name,
        origin_type="mixed",
        layout_complexity=LayoutComplexity.multi_column,
    )

    doc = layout.document

    print(f"PDF: {path}")
    print(f"Engine: {layout.engine}")
    print(f"Origin type: {layout.origin_type}")
    print(f"Layout complexity: {layout.layout_complexity}")
    print(f"Text blocks: {len(doc.text_blocks)}")
    print(f"Tables: {len(doc.tables)}")
    print(f"Figures: {len(doc.figures)}")
    print(f"Reading order entries: {len(doc.reading_order)}")

    # Show a small sample of the extracted structure
    if doc.text_blocks:
        tb = doc.text_blocks[0]
        print("\nFirst text block:")
        print(f"  page={tb.page_no}, label={tb.label}, bbox={tb.bbox}")
        print(f"  text={tb.text[:200]!r}")

    if doc.tables:
        t = doc.tables[0]
        print("\nFirst table:")
        print(f"  page={t.page_no}, caption={t.caption!r}")
        print(f"  header rows={len(t.headers)}, body rows={len(t.rows)}")

    if doc.figures:
        f = doc.figures[0]
        print("\nFirst figure:")
        print(f"  page={f.page_no}, caption={f.caption!r}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "./data/RepoInvestigator_Architecture_Report.pdf"

    asyncio.run(main(target))

