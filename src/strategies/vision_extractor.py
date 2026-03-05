"""
Strategy C — Vision-Augmented (high cost).

Uses a VLM via OpenRouter (Gemini Flash / GPT-4o-mini) for multimodal extraction.
Triggers: scanned_image, or Strategy A/B confidence below threshold, or handwriting.
Budget guard enforces a per-document token cap and logs estimated cost.
"""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Any

import httpx
import numpy as np

from ..models.extracted_document import (
    ExtractedDocument,
    FigureObject,
    TableCell,
    TableObject,
    TextBlock,
)
from ..models.provenance import BoundingBox as OurBoundingBox
from ..services.triage_services.artifact_loader import DocumentArtifacts

from .vision_config import (
    BudgetGuard,
    DEFAULT_VISION_MODEL,
    get_model_costs,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Placeholder bbox when VLM does not provide coordinates
PLACEHOLDER_BBOX = OurBoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

EXTRACTION_PROMPT = """You are a document understanding assistant. For the given page image, extract structure as JSON only (no markdown, no explanation).

Return a single JSON object with this exact shape:
{
  "text_blocks": [ {"text": "<content>", "page_no": 1, "label": "paragraph|section_header|title|list_item|footnote"} ],
  "tables": [ {"headers": [[{"text":"","is_header":true,"row_index":0,"col_index":0,"row_span":1,"col_span":1}]], "rows": [[{"text":"","is_header":false,"row_index":0,"col_index":0,"row_span":1,"col_span":1}]], "caption": null, "page_no": 1} ],
  "figures": [ {"caption": null, "page_no": 1} ],
  "reading_order": ["text_0", "table_0", "figure_0"]
}

Rules:
- page_no for this image is {page_no}.
- For text_blocks: preserve reading order; label can be paragraph, section_header, title, list_item, footnote.
- For tables: use headers and rows as lists of list of cells; each cell has text, is_header, row_index, col_index, row_span, col_span.
- For figures: include if the page has diagrams/photos; caption if visible.
- reading_order: list of IDs like text_0, text_1, table_0, figure_0 in reading order.
Return only the JSON object."""


def _image_to_base64_url(image: np.ndarray) -> str:
    """Encode a numpy image (RGB) to a data URL for OpenRouter."""
    from PIL import Image
    if image.ndim == 2:
        pil = Image.fromarray(image).convert("RGB")
    else:
        pil = Image.fromarray(image)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _parse_json_from_content(content: str) -> dict[str, Any]:
    """Extract JSON from model output, allowing ```json ... ``` wrapper."""
    content = content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        content = match.group(1).strip()
    return json.loads(content)


def _cell_from_dict(d: Any, r: int, c: int) -> TableCell:
    if isinstance(d, str):
        return TableCell(text=d, is_header=False, row_index=r, col_index=c)
    return TableCell(
        text=str(d.get("text", "")),
        is_header=bool(d.get("is_header", False)),
        row_index=r,
        col_index=c,
        row_span=int(d.get("row_span", 1)),
        col_span=int(d.get("col_span", 1)),
    )


def _table_from_dict(t: dict, page_no: int) -> TableObject:
    headers: list[list[TableCell]] = []
    for ri, row in enumerate(t.get("headers") or []):
        headers.append([_cell_from_dict(cell, ri, ci) for ci, cell in enumerate(row)])
    rows: list[list[TableCell]] = []
    for ri, row in enumerate(t.get("rows") or []):
        rows.append([_cell_from_dict(cell, ri, ci) for ci, cell in enumerate(row)])
    cap = t.get("caption")
    caption = str(cap).strip() or None if cap is not None else None
    return TableObject(
        headers=headers,
        rows=rows,
        caption=caption,
        page_no=int(t.get("page_no", page_no)),
        bbox=PLACEHOLDER_BBOX,
    )


def _build_document_from_page_json(
    page_json: dict[str, Any],
    page_no: int,
    base_tb: int,
    base_ta: int,
    base_fig: int,
) -> tuple[list[TextBlock], list[TableObject], list[FigureObject], list[str]]:
    """Convert one page's JSON into text_blocks, tables, figures, and reading_order segment."""
    text_blocks: list[TextBlock] = []
    tables: list[TableObject] = []
    figures: list[FigureObject] = []
    order: list[str] = []

    for i, tb in enumerate(page_json.get("text_blocks") or []):
        text_blocks.append(
            TextBlock(
                text=str(tb.get("text", "")),
                page_no=int(tb.get("page_no", page_no)),
                bbox=PLACEHOLDER_BBOX,
                label=str(tb.get("label", "paragraph")),
            )
        )
        order.append(f"text_{base_tb + i}")

    for i, t in enumerate(page_json.get("tables") or []):
        tables.append(_table_from_dict(t, page_no))
        order.append(f"table_{base_ta + i}")

    for i, f in enumerate(page_json.get("figures") or []):
        cap = f.get("caption")
        figures.append(
            FigureObject(
                caption=str(cap).strip() or None if cap else None,
                page_no=int(f.get("page_no", page_no)),
                bbox=PLACEHOLDER_BBOX,
            )
        )
        order.append(f"figure_{base_fig + i}")

    # If model returned reading_order for this page, use it to reorder; else keep default order
    ro = page_json.get("reading_order") or []
    if ro:
        idx_by_id: dict[str, int] = {}
        for idx, sid in enumerate(order):
            idx_by_id[sid] = idx
        reordered = []
        for s in ro:
            if isinstance(s, str) and s in idx_by_id:
                reordered.append(s)
        if reordered:
            order = reordered

    return text_blocks, tables, figures, order


class VisionExtractor:
    """
    Strategy C — Vision-Augmented. Uses OpenRouter VLM for page images.
    Enforces a per-document budget cap and logs estimated cost.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = DEFAULT_VISION_MODEL,
        budget_cap_tokens: int = 500_000,
        base_url: str = OPENROUTER_URL,
    ) -> None:
        self.api_key = api_key or ""
        self.model_id = model_id
        self.base_url = base_url
        self.budget_cap_tokens = budget_cap_tokens
        c_in, c_out = get_model_costs(model_id)
        self._cost_per_1k_input = c_in
        self._cost_per_1k_output = c_out

    def _budget_guard(self) -> BudgetGuard:
        return BudgetGuard(
            max_tokens=self.budget_cap_tokens,
            cost_per_1k_input=self._cost_per_1k_input,
            cost_per_1k_output=self._cost_per_1k_output,
        )

    @staticmethod
    def is_applicable(
        origin_type: str,
        strategy_a_b_confidence: float | None,
        confidence_threshold: float,
        handwriting_detected: bool = False,
    ) -> bool:
        """True if we should use Strategy C: scanned, low confidence, or handwriting."""
        if origin_type == "scanned_image":
            return True
        if handwriting_detected:
            return True
        if strategy_a_b_confidence is not None and strategy_a_b_confidence < confidence_threshold:
            return True
        return False

    async def extract(
        self,
        artifacts: DocumentArtifacts,
        doc_id: str,
        trigger_reason: str = "scanned_image",
    ) -> ExtractedDocument:
        """
        Load page images, send each to OpenRouter with extraction prompt,
        merge into one ExtractedDocument. Budget guard caps total tokens per document.
        """
        import os
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for VisionExtractor")

        images: list[np.ndarray] = await artifacts.load_images()
        guard = self._budget_guard()

        all_text_blocks: list[TextBlock] = []
        all_tables: list[TableObject] = []
        all_figures: list[FigureObject] = []
        reading_order: list[str] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for page_no, image in enumerate(images, start=1):
                prompt = EXTRACTION_PROMPT.format(page_no=page_no)
                image_url = _image_to_base64_url(image)
                payload = {
                    "model": self.model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    "max_tokens": 4096,
                }

                resp = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                usage = data.get("usage") or {}
                inp = int(usage.get("prompt_tokens", 0))
                out = int(usage.get("completion_tokens", 0))
                guard.add_usage(inp, out)

                content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or "{}"
                try:
                    page_json = _parse_json_from_content(content)
                except (json.JSONDecodeError, TypeError) as e:
                    page_json = {"text_blocks": [{"text": content[:5000], "page_no": page_no, "label": "paragraph"}], "tables": [], "figures": [], "reading_order": ["text_0"]}

                tb, ta, fig, order = _build_document_from_page_json(
                    page_json,
                    page_no,
                    base_tb=len(all_text_blocks),
                    base_ta=len(all_tables),
                    base_fig=len(all_figures),
                )
                all_text_blocks.extend(tb)
                all_tables.extend(ta)
                all_figures.extend(fig)
                reading_order.extend(order)

        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=all_text_blocks,
            tables=all_tables,
            figures=all_figures,
            reading_order=reading_order,
            metadata={
                "engine": "vision_openrouter",
                "strategy": "vision_augmented",
                "model_id": self.model_id,
                "trigger_reason": trigger_reason,
                "total_pages": len(images),
                "input_tokens": guard.input_tokens,
                "output_tokens": guard.output_tokens,
                "estimated_cost_usd": guard.estimated_cost_usd,
            },
        )
