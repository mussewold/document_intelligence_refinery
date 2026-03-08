from typing import Literal, Optional, List

from pydantic import BaseModel, Field

from .provenance import BoundingBox
from .provenance_chain import ProvenanceChain


class LDU(BaseModel):
    """
    Logical Document Unit: a referenceable unit of document content (e.g. text block,
    table, or figure) with a stable id and optional provenance chain.

    Stage 3 requirement: each LDU carries content, chunk_type, page_refs,
    bounding_box, parent_section, token_count, and a content_hash.
    """

    id: str = Field(..., description="Stable identifier for this unit (e.g. text_0, table_1)")
    type: Literal["text_block", "table", "figure", "section_header", "list"] = "text_block"
    chunk_type: Literal["text", "table", "figure", "section_header", "list"] = "text"

    # Primary page for this unit plus optional additional references (for multi-page tables, etc.).
    page_no: int = Field(..., ge=1, description="1-based primary page number")
    page_refs: List[int] = Field(default_factory=list, description="All pages this unit touches")

    bbox: Optional[BoundingBox] = Field(default=None, description="Spatial extent on the primary page")

    # Resolved content text for this unit.
    content: str = Field("", description="Resolved textual content for this LDU")

    # Section information and tokenization.
    parent_section: Optional[str] = Field(default=None, description="Nearest section header/title governing this unit")
    token_count: int = Field(0, ge=0, description="Approximate token count for this unit")

    # Stable content hash for provenance verification.
    content_hash: Optional[str] = Field(default=None, description="Stable hash over content + geometry + doc_id")

    # Back-references and additional provenance.
    content_ref: Optional[str] = Field(default=None, description="Reference to original content id or snippet")
    provenance: Optional[ProvenanceChain] = Field(default=None, description="Chain tracing origin of this unit")
    meta: Optional[dict] = Field(default=None, description="Optional strategy- or engine-specific metadata")
