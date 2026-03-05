from pydantic import BaseModel, Field


class PageIndex(BaseModel):
    """
    Index of a single page: page number plus the list of LDU ids that appear on it.
    Used to resolve "which units are on page N" and to build document-level indexes.
    """

    page_no: int = Field(..., ge=1, description="1-based page number")
    ldu_ids: list[str] = Field(default_factory=list, description="Ordered list of LDU ids on this page")
