from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from ..models.provenance import BoundingBox


def compute_content_hash(
    *,
    doc_id: str,
    page_refs: Sequence[int],
    content: str,
    bbox: Optional[BoundingBox] = None,
    hash_len: int = 16,
) -> str:
    """
    Compute a stable content hash for an LDU.

    Follows the same spirit as Week 1's spatial hashing:
    - doc_id
    - sorted page_refs
    - rounded bounding box coordinates (if present)
    - normalized content
    """
    norm_text = " ".join(content.split())[:10_000]
    pages = ",".join(str(p) for p in sorted(set(page_refs))) if page_refs else ""

    bbox_str = ""
    if bbox is not None:
        # Round to 2 decimal places to be robust to tiny jitter.
        bbox_str = f"{bbox.x0:.2f},{bbox.y0:.2f},{bbox.x1:.2f},{bbox.y1:.2f}"

    payload = f"{doc_id}|{pages}|{bbox_str}|{norm_text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:hash_len]

