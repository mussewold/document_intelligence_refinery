from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, TypedDict

from ..services.triage_services.artifact_loader import DocumentArtifacts
from ..models.extracted_document import ExtractedDocument, TextBlock
from ..models.provenance import BoundingBox


class PageSignalDict(TypedDict):
    page_index: int
    char_count: int
    char_density: float
    image_area_ratio: float
    has_font_metadata: bool
    page_confidence: float


@dataclass
class PageSignal:
    page_index: int
    char_count: int
    char_density: float
    image_area_ratio: float
    has_font_metadata: bool
    page_confidence: float

    def to_dict(self) -> PageSignalDict:
        # asdict keeps the structure simple and explicit
        return asdict(self)  # type: ignore[return-value]


class FastTextExtractor:
    """
    Strategy A — Fast Text (low-cost engine).

    This extractor is intended to be used when:
      - origin_type == "native_digital"
      - layout_complexity == "single_column"

    It performs lightweight text extraction with pdfplumber (via DocumentArtifacts)
    and computes a multi-signal confidence score per page and for the document:
      - character count per page
      - character density (characters vs. page area)
      - image-to-page area ratio
      - presence of font metadata

    Thresholds are documented in `extraction_rules.yaml` and mirrored here as
    defaults. They can be overridden at construction time or by a higher-level
    strategy orchestrator.
    """

    def __init__(
        self,
        *,
        min_chars_per_page: int = 100,
        min_avg_char_density: float = 1e-5,
        max_image_area_ratio: float = 0.5,
        min_confidence: float = 0.7,
    ) -> None:
        self.min_chars_per_page = min_chars_per_page
        self.min_avg_char_density = min_avg_char_density
        self.max_image_area_ratio = max_image_area_ratio
        self.min_confidence = min_confidence

    @staticmethod
    def is_applicable(
        origin_type: str,
        layout_complexity: Optional[
            Literal[
                "single_column",
                "multi_column",
                "table_heavy",
                "figure_heavy",
                "mixed",
            ]
        ],
    ) -> bool:
        """
        Check if Strategy A — Fast Text should be considered for a document.
        """
        return origin_type == "native_digital" and layout_complexity in {
            "single_column",
        }

    async def extract(
        self,
        artifacts: DocumentArtifacts,
        doc_id: str,
    ) -> ExtractedDocument:
        """
        Perform text extraction using pdfplumber and compute confidence signals.

        This method assumes the caller already validated that
        `FastTextExtractor.is_applicable(...)` returned True.
        """
        pdf = await artifacts.load_pdf()

        page_texts: List[str] = []
        text_blocks: List[TextBlock] = []
        page_signals: List[PageSignal] = []

        total_chars = 0
        total_char_density = 0.0
        total_pages = len(pdf.pages) if getattr(pdf, "pages", None) else 0

        for index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            page_texts.append(text)

            page_no = index + 1

            char_count = len(text)
            width = float(getattr(page, "width", 0) or 0)
            height = float(getattr(page, "height", 0) or 0)
            page_area = (width * height) or 1.0

            char_density = char_count / page_area

            image_area = 0.0
            for img in getattr(page, "images", []):
                try:
                    image_area += float(img.get("width", 0)) * float(
                        img.get("height", 0)
                    )
                except Exception:
                    # If image metadata is malformed, just skip that image.
                    continue

            image_ratio = image_area / page_area

            has_font_metadata = False
            try:
                for ch in getattr(page, "chars", []):
                    # pdfplumber char objects generally expose fontname/size
                    if ch.get("fontname") or ch.get("size"):
                        has_font_metadata = True
                        break
            except Exception:
                has_font_metadata = False

            page_confidence = self._compute_page_confidence(
                char_count=char_count,
                char_density=char_density,
                image_ratio=image_ratio,
                has_font_metadata=has_font_metadata,
            )

            page_signals.append(
                PageSignal(
                    page_index=index,
                    char_count=char_count,
                    char_density=char_density,
                    image_area_ratio=image_ratio,
                    has_font_metadata=has_font_metadata,
                    page_confidence=page_confidence,
                )
            )

            total_chars += char_count
            total_char_density += char_density

            # For the fast path we treat each page as a single text block
            # spanning the full page.
            bbox = BoundingBox(x0=0.0, y0=0.0, x1=width, y1=height)
            text_blocks.append(
                TextBlock(
                    text=text,
                    page_no=page_no,
                    bbox=bbox,
                    label="paragraph",
                )
            )

        full_text = "\n\n".join(page_texts)

        # Aggregate document-level metrics
        avg_char_per_page = total_chars / total_pages if total_pages else 0.0
        avg_char_density = (
            total_char_density / total_pages if total_pages else 0.0
        )
        avg_image_ratio = (
            sum(p.image_area_ratio for p in page_signals) / total_pages
            if total_pages
            else 0.0
        )

        # Base confidence: mean of per-page confidences
        if page_signals:
            base_confidence = sum(
                p.page_confidence for p in page_signals
            ) / len(page_signals)
        else:
            base_confidence = 0.0

        # Apply document-level gates from Strategy A description
        confidence = base_confidence

        if avg_char_per_page < self.min_chars_per_page:
            # Not enough text overall → heavily penalize
            confidence *= 0.3

        if avg_char_density < self.min_avg_char_density:
            confidence *= 0.5

        if avg_image_ratio > self.max_image_area_ratio:
            # Images dominate the page → low confidence
            confidence *= 0.2

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # Reading order is simply sequential by page for this fast strategy.
        reading_order = [f"page_{i}" for i in range(1, total_pages + 1)]

        metadata = {
            "engine": "fast_text",
            "strategy": "fast_text",
            "confidence": confidence,
            "aggregates": {
                "avg_char_per_page": avg_char_per_page,
                "avg_char_density": avg_char_density,
                "avg_image_ratio": avg_image_ratio,
                "total_chars": total_chars,
                "total_pages": total_pages,
            },
            "page_signals": [p.to_dict() for p in page_signals],
        }

        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=text_blocks,
            tables=[],
            figures=[],
            reading_order=reading_order,
            metadata=metadata,
        )

    def _compute_page_confidence(
        self,
        *,
        char_count: int,
        char_density: float,
        image_ratio: float,
        has_font_metadata: bool,
    ) -> float:
        """
        Multi-signal confidence for a single page.

        The components are intentionally simple and monotonic so they are easy
        to reason about and tune during Phase 0 exploration.
        """

        # Text signal — increases with character count up to 2x the threshold
        text_signal = min(1.0, char_count / (self.min_chars_per_page * 2.0))

        # Density signal — reward densities around/above the configured minimum
        if char_density <= 0:
            density_signal = 0.0
        else:
            density_signal = min(1.0, char_density / (self.min_avg_char_density * 2.0))

        # Image signal — full score when well below the max ratio, decays above it
        if image_ratio <= self.max_image_area_ratio:
            image_signal = 1.0
        else:
            # Linearly decay; a page that is almost entirely image gets near-zero
            over = min(1.0, (image_ratio - self.max_image_area_ratio) / 0.5)
            image_signal = max(0.0, 1.0 - over)

        # Font metadata — modest boost if we see font info on this page
        font_signal = 1.0 if has_font_metadata else 0.7

        # Weighted combination of the signals
        page_confidence = (
            0.4 * text_signal
            + 0.3 * density_signal
            + 0.2 * image_signal
            + 0.1 * font_signal
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, page_confidence))

