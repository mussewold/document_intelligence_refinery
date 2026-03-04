from pydantic import BaseModel, Field
from typing import List, Optional, Any
from .provenance import BoundingBox  # Assuming a shared bbox schema

class TableCell(BaseModel):
    text: str
    is_header: bool = False
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1

class TableObject(BaseModel):
    """Structured table representation including headers and rows [1]"""
    headers: List[List[TableCell]]
    rows: List[List[TableCell]]
    caption: Optional[str] = None
    page_no: int
    bbox: BoundingBox

class TextBlock(BaseModel):
    """Text segment with mandatory spatial provenance [1]"""
    text: str
    page_no: int
    bbox: BoundingBox
    label: str  # e.g., paragraph, section_header, footnote [4]

class FigureObject(BaseModel):
    """Figure representation with caption and spatial coordinates [1]"""
    caption: Optional[str]
    page_no: int
    bbox: BoundingBox

class ExtractedDocument(BaseModel):
    """
    The universal internal schema for the Refinery.
    All adapters (Docling, MinerU, VLM) must normalize into this model [1].
    """
    doc_id: str
    text_blocks: List[TextBlock] = Field(
        ..., description="All text elements in logical reading order"
    )
    tables: List[TableObject] = Field(
        default_factory=list, description="Extracted tables as structured objects"
    )
    figures: List[FigureObject] = Field(
        default_factory=list, description="Extracted figures and their captions"
    )
    reading_order: List[str] = Field(
        ..., description="Reconstructed human-readable sequence of element IDs [1]"
    )
    metadata: Optional[dict] = None