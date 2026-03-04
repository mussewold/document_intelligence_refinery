from pydantic import BaseModel


class BoundingBox(BaseModel):
    """
    Simple bounding-box schema shared across models.

    Coordinates are in PDF page space (usually points), with origin at
    the bottom-left, matching the conventions used by pdfplumber/pypdf.
    """

    x0: float
    y0: float
    x1: float
    y1: float

