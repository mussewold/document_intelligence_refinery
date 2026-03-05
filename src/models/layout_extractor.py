from typing import Literal, Optional

from pydantic import BaseModel, Field

from .extracted_document import ExtractedDocument
from .document_profile import LayoutComplexity


class LayoutExtractor(BaseModel):
    """
    Pydantic wrapper describing a layout-aware extraction run.

    Strategy B — Layout-Aware (Docling) is represented by this model so that
    downstream components can inspect which engine was used, under which
    triage triggers, and access the normalized `ExtractedDocument`.
    """

    engine: Literal["docling"] = "docling"

    # Triage signals that led to selecting this strategy.
    origin_type: str = Field(..., description="Detected PDF origin type, e.g. native_digital, mixed.")
    layout_complexity: LayoutComplexity = Field(
        ..., description="Layout complexity from the triage agent."
    )

    # Normalized internal schema produced by the adapter.
    document: ExtractedDocument = Field(
        ..., description="Doc content normalized to the Refinery schema."
    )

    # Optional strategy-specific metadata (e.g., timings, debug info).
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional debug/trace metadata for this extraction run.",
    )

