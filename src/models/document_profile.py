from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


# ----------------------------
# Enums (Strongly Typed Fields)
# ----------------------------

class LayoutComplexity(str, Enum):
    single_column = "single_column"
    multi_column = "multi_column"
    table_heavy = "table_heavy"
    figure_heavy = "figure_heavy"
    mixed = "mixed"


class ExtractionCost(str, Enum):
    fast_text_sufficient = "fast_text_sufficient"
    needs_layout_model = "needs_layout_model"
    needs_vision_model = "needs_vision_model"


class DomainHint(str, Enum):
    financial = "financial"
    legal = "legal"
    technical = "technical"
    medical = "medical"
    general = "general"


# ----------------------------
# Supporting Models
# ----------------------------

class PageProfile(BaseModel):
    page_number: int
    language: str
    language_confidence: float
    layout_complexity: LayoutComplexity
    domain_hint: DomainHint


# ----------------------------
# Main Document Profile
# ----------------------------

class DocumentProfile(BaseModel):
    file_name: str

    # Document-level signals
    origin_type: str = "native_digital"  # native_digital | scanned_image | mixed | form_fillable
    primary_language: str
    language_confidence: float

    layout_complexity: LayoutComplexity
    domain_hint: DomainHint
    estimated_extraction_cost: ExtractionCost

    # Statistics
    total_pages: int
    total_text_length: int

    # Optional per-page breakdown
    pages: Optional[List[PageProfile]] = None

    # Optional debug metadata
    element_counts: Optional[dict] = None

    class Config:
        use_enum_values = True