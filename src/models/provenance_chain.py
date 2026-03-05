from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ProvenanceStep(BaseModel):
    """Single step in a provenance chain (e.g. source file, page, engine, element)."""

    step_type: Literal["source", "page", "engine", "element", "segment"] = "element"
    value: str | int
    label: Optional[str] = None
    meta: Optional[dict[str, Any]] = None


class ProvenanceChain(BaseModel):
    """
    Ordered chain of provenance steps tracing where a piece of content came from.
    E.g. [source=doc.pdf, page=1, engine=docling, element=text_0].
    """

    steps: list[ProvenanceStep] = Field(default_factory=list, description="Ordered provenance steps")
