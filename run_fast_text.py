import asyncio
from pathlib import Path

from src.services.triage_services.artifact_loader import DocumentArtifacts
from src.strategies import FastTextExtractor


async def main(pdf_path: str) -> None:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    artifacts = DocumentArtifacts(str(path))
    extractor = FastTextExtractor()

    doc = await extractor.extract(artifacts, doc_id=path.name)
    print(f"Type of output: {type(doc)}")

    meta = doc.metadata or {}
    confidence = meta.get("confidence")
    aggregates = meta.get("aggregates", {})
    page_signals = meta.get("page_signals", [])

    if confidence is not None:
        print(f"Overall confidence: {confidence:.3f}")
    print(f"Text blocks: {len(doc.text_blocks)}")
    print(f"Reading order entries: {len(doc.reading_order)}")
    if aggregates:
        print("Aggregates:")
        for k, v in aggregates.items():
            print(f"  {k}: {v}")
    if page_signals:
        print("\nFirst 3 page signals:")
        for s in page_signals[:3]:
            print(
                f"  Page {s['page_index']}: "
                f"chars={s['char_count']}, "
                f"density={s['char_density']:.2e}, "
                f"image_ratio={s['image_area_ratio']:.2f}, "
                f"font_meta={s['has_font_metadata']}, "
                f"page_conf={s['page_confidence']:.3f}"
            )


if __name__ == "__main__":
    # Default to the RepoInvestigator sample if no argument is given.
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "./data/RepoInvestigator_Architecture_Report.pdf"

    asyncio.run(main(target))

