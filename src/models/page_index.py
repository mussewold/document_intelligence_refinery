from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PageIndex(BaseModel):
    """
    Index of a single page: page number plus the list of LDU ids that appear on it.
    Used to resolve "which units are on page N" and to build document-level indexes.
    """

    page_no: int = Field(..., ge=1, description="1-based page number")
    ldu_ids: list[str] = Field(default_factory=list, description="Ordered list of LDU ids on this page")


class PageIndexNode(BaseModel):
    """
    Hierarchical PageIndex node (section) inspired by VectifyAI's PageIndex.

    Represents a logical section with:
      - title
      - page_start / page_end
      - child_sections
      - key_entities
      - summary (2–3 sentences)
      - data_types_present (tables, figures, equations, etc.)
    """

    id: str = Field(..., description="Stable identifier for the section node")
    title: str = Field(..., description="Section title")
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)

    child_sections: List["PageIndexNode"] = Field(default_factory=list)

    key_entities: List[str] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None, description="Short LLM-generated summary (2–3 sentences)")
    data_types_present: List[str] = Field(
        default_factory=list,
        description="Data types present under this section: tables, figures, equations, etc.",
    )

    # Optional: LDUs that belong to this section (ids only)
    ldu_ids: List[str] = Field(default_factory=list, description="IDs of LDUs contained in this section")


PageIndexNode.model_rebuild()

