from typing import Literal, Optional

from pydantic import BaseModel, Field

from .provenance import BoundingBox
from .provenance_chain import ProvenanceChain


class LDU(BaseModel):
    """
    Logical Document Unit: a referenceable unit of document content (e.g. text block,
    table, or figure) with a stable id and optional provenance chain.
    """

    id: str = Field(..., description="Stable identifier for this unit (e.g. text_0, table_1)")
    type: Literal["text_block", "table", "figure"] = "text_block"
    page_no: int = Field(..., ge=1, description="1-based page number")
    bbox: Optional[BoundingBox] = Field(default=None, description="Spatial extent on the page")
    content_ref: Optional[str] = Field(default=None, description="Reference to content or inline text snippet")
    provenance: Optional[ProvenanceChain] = Field(default=None, description="Chain tracing origin of this unit")
    meta: Optional[dict] = Field(default=None, description="Optional strategy- or engine-specific metadata")
