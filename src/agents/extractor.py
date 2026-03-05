"""
ExtractionRouter: strategy-pattern router that reads DocumentProfile and
delegates to the correct extractor (A/B/C), with confidence-gated escalation.
Logs every extraction to .refinery/extraction_ledger.jsonl.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..models.document_profile import (
    DocumentProfile,
    ExtractionCost,
    LayoutComplexity,
)
from ..models.extracted_document import ExtractedDocument
from ..services.triage_services.artifact_loader import DocumentArtifacts
from ..strategies import (
    DoclingLayoutExtractor,
    FastTextExtractor,
    VisionExtractor,
)
from ..services.extraction_ledger import append_extraction

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of routing: extracted document plus which strategy was used and any escalation."""

    document: ExtractedDocument
    strategy_used: str  # "fast_text" | "docling" | "vision"
    escalated: bool = False
    escalation_path: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.document.doc_id


class ExtractionRouter:
    """
    Reads DocumentProfile and delegates to the appropriate extractor (A, B, or C).
    Applies confidence-gated escalation: if Strategy A runs but confidence is below
    threshold, re-runs with Strategy B (and optionally B → C if needed).
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.7,
        fast_text: Optional[FastTextExtractor] = None,
        docling: Optional[DoclingLayoutExtractor] = None,
        vision: Optional[VisionExtractor] = None,
        enable_escalation_a_to_b: bool = True,
        enable_escalation_b_to_c: bool = True,
        ledger_path: Optional[Path] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._fast_text = fast_text or FastTextExtractor()
        self._docling = docling or DoclingLayoutExtractor()
        self._vision = vision or VisionExtractor()
        self._enable_escalation_a_to_b = enable_escalation_a_to_b
        self._enable_escalation_b_to_c = enable_escalation_b_to_c
        self._ledger_path: Optional[Path] = ledger_path if ledger_path is not None else Path(".refinery/extraction_ledger.jsonl")

    def _log_to_ledger(
        self,
        result: ExtractionResult,
        processing_time_seconds: float,
    ) -> None:
        """Write one line to .refinery/extraction_ledger.jsonl."""
        if self._ledger_path is None:
            return
        meta = result.document.metadata or {}
        confidence_score = meta.get("confidence")
        cost_estimate = meta.get("estimated_cost_usd")
        append_extraction(
            doc_id=result.doc_id,
            strategy_used=result.strategy_used,
            confidence_score=confidence_score,
            cost_estimate=cost_estimate,
            processing_time_seconds=processing_time_seconds,
            escalated=result.escalated,
            escalation_path=result.escalation_path,
            ledger_path=self._ledger_path,
        )

    def _select_initial_strategy(self, profile: DocumentProfile) -> str:
        """
        Choose which extractor to try first from profile only.
        Returns "vision" | "docling" | "fast_text".
        """
        origin = getattr(profile, "origin_type", "native_digital") or "native_digital"
        cost = profile.estimated_extraction_cost
        layout = profile.layout_complexity

        if isinstance(cost, ExtractionCost):
            cost = cost.value if hasattr(cost, "value") else str(cost)
        if isinstance(layout, LayoutComplexity):
            layout = layout.value if hasattr(layout, "value") else str(layout)

        # C first when document clearly needs vision
        if origin == "scanned_image" or cost == ExtractionCost.needs_vision_model.value:
            return "vision"
        if cost == "needs_vision_model":
            return "vision"

        # B when layout is complex or origin mixed
        if origin == "mixed" or layout in {
            LayoutComplexity.multi_column.value,
            LayoutComplexity.table_heavy.value,
        }:
            return "docling"
        if layout in ("multi_column", "table_heavy"):
            return "docling"

        # A when native_digital + single_column
        if origin == "native_digital" and layout == "single_column":
            return "fast_text"

        # Default: layout-aware for safety
        return "docling"

    def _confidence_from_document(self, doc: ExtractedDocument) -> Optional[float]:
        """Extract strategy confidence from ExtractedDocument.metadata if present."""
        meta = doc.metadata or {}
        return meta.get("confidence")

    async def extract(
        self,
        profile: DocumentProfile,
        artifacts: DocumentArtifacts,
        *,
        doc_id: Optional[str] = None,
        handwriting_detected: bool = False,
    ) -> ExtractionResult:
        """
        Run the correct extractor from profile, with confidence-gated escalation.
        Uses profile.origin_type, profile.layout_complexity, profile.estimated_extraction_cost.
        Logs each run to .refinery/extraction_ledger.jsonl.
        """
        t0 = time.perf_counter()
        doc_id = doc_id or profile.file_name
        origin = getattr(profile, "origin_type", "native_digital") or "native_digital"
        layout = profile.layout_complexity
        if isinstance(layout, LayoutComplexity):
            layout_value = layout.value
        else:
            layout_value = str(layout)

        cost_val = getattr(profile.estimated_extraction_cost, "value", profile.estimated_extraction_cost) or str(profile.estimated_extraction_cost)
        layout_enum = layout if isinstance(layout, LayoutComplexity) else LayoutComplexity(layout_value)

        # 1) Force vision when scanned or vision is required
        if origin == "scanned_image" or cost_val == "needs_vision_model":
            logger.info("ExtractionRouter: using vision (scanned or needs_vision_model)")
            doc = await self._vision.extract(
                artifacts=artifacts,
                doc_id=doc_id,
                trigger_reason="scanned_image" if origin == "scanned_image" else "needs_vision_model",
            )
            result = ExtractionResult(
                document=doc,
                strategy_used="vision",
                escalated=False,
                escalation_path=[],
            )
            self._log_to_ledger(result, time.perf_counter() - t0)
            return result

        if handwriting_detected and VisionExtractor.is_applicable(
            origin, None, self.confidence_threshold, handwriting_detected=True
        ):
            logger.info("ExtractionRouter: using vision (handwriting_detected)")
            doc = await self._vision.extract(
                artifacts=artifacts,
                doc_id=doc_id,
                trigger_reason="handwriting_detected",
            )
            result = ExtractionResult(
                document=doc,
                strategy_used="vision",
                escalated=False,
                escalation_path=[],
            )
            self._log_to_ledger(result, time.perf_counter() - t0)
            return result

        # 2) Try Strategy A when applicable
        if FastTextExtractor.is_applicable(origin, layout_value):
            doc = await self._fast_text.extract(artifacts=artifacts, doc_id=doc_id)
            confidence = self._confidence_from_document(doc)
            if confidence is not None and confidence >= self.confidence_threshold:
                logger.info("ExtractionRouter: using fast_text (confidence %.3f >= %.3f)", confidence, self.confidence_threshold)
                result = ExtractionResult(
                    document=doc,
                    strategy_used="fast_text",
                    escalated=False,
                    escalation_path=[],
                )
                self._log_to_ledger(result, time.perf_counter() - t0)
                return result
            if self._enable_escalation_a_to_b:
                logger.info(
                    "ExtractionRouter: fast_text confidence %.3f < %.3f, escalating to docling",
                    confidence or 0.0,
                    self.confidence_threshold,
                )
                layout_result = await self._docling.extract(
                    artifacts=artifacts,
                    doc_id=doc_id,
                    origin_type=origin,
                    layout_complexity=layout_enum,
                )
                doc = layout_result.document
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["escalated_from"] = "fast_text"
                doc.metadata["strategy_used"] = "docling"
                result = ExtractionResult(
                    document=doc,
                    strategy_used="docling",
                    escalated=True,
                    escalation_path=["fast_text", "docling"],
                )
                self._log_to_ledger(result, time.perf_counter() - t0)
                return result
            result = ExtractionResult(
                document=doc,
                strategy_used="fast_text",
                escalated=False,
                escalation_path=[],
            )
            self._log_to_ledger(result, time.perf_counter() - t0)
            return result

        # 3) Strategy B when applicable (mixed, multi_column, table_heavy)
        if DoclingLayoutExtractor.is_applicable(origin, layout_enum):
            logger.info("ExtractionRouter: using docling")
            layout_result = await self._docling.extract(
                artifacts=artifacts,
                doc_id=doc_id,
                origin_type=origin,
                layout_complexity=layout_enum,
            )
            doc = layout_result.document
            # Optional B → C escalation: e.g. very few text blocks
            if self._enable_escalation_b_to_c and doc.metadata:
                n_blocks = len(doc.text_blocks)
                n_pages = doc.metadata.get("num_pages") or profile.total_pages or 1
                if n_pages and n_blocks < 2 and n_pages > 0:
                    logger.info("ExtractionRouter: docling yielded very few blocks, escalating to vision")
                    doc = await self._vision.extract(
                        artifacts=artifacts,
                        doc_id=doc_id,
                        trigger_reason="low_yield_from_docling",
                    )
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["escalated_from"] = "docling"
                    doc.metadata["strategy_used"] = "vision"
                    result = ExtractionResult(
                        document=doc,
                        strategy_used="vision",
                        escalated=True,
                        escalation_path=["docling", "vision"],
                    )
                    self._log_to_ledger(result, time.perf_counter() - t0)
                    return result
            result = ExtractionResult(
                document=doc,
                strategy_used="docling",
                escalated=False,
                escalation_path=[],
            )
            self._log_to_ledger(result, time.perf_counter() - t0)
            return result

        # 4) Fallback: try Docling then Vision if needed
        logger.info("ExtractionRouter: fallback to docling")
        layout_result = await self._docling.extract(
            artifacts=artifacts,
            doc_id=doc_id,
            origin_type=origin,
            layout_complexity=layout_enum,
        )
        result = ExtractionResult(
            document=layout_result.document,
            strategy_used="docling",
            escalated=False,
            escalation_path=[],
        )
        self._log_to_ledger(result, time.perf_counter() - t0)
        return result
