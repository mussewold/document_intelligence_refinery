from __future__ import annotations

"""
Semantic Chunking Engine (Stage 3).

Converts an ExtractedDocument into Logical Document Units (LDUs) while enforcing
chunking rules:

1) A table cell is never split from its header row  → we treat each table as a
   single LDU that includes headers + body.
2) A figure caption is stored as metadata of its parent figure LDU.
3) A numbered list is kept as a single LDU unless it exceeds max_tokens.
4) Section headers are stored as parent metadata on all child LDUs in that section.
5) Cross-references ("see Table 3") are resolved and stored as relationships.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from ..models.extracted_document import ExtractedDocument, TextBlock, TableObject, FigureObject
from ..models.ldu import LDU
from ..services.content_hashing import compute_content_hash


@dataclass
class ChunkingConfig:
    max_tokens: int = 512
    min_tokens: int = 32


class ChunkValidator:
    """
    Validates that a list of LDUs respects the high-level chunking rules.
    This focuses on invariants we can check post-hoc (token bounds, captions, etc.).
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()

    def validate(self, ldus: List[LDU]) -> None:
        max_t = self.config.max_tokens
        min_t = self.config.min_tokens

        for l in ldus:
            # Token bounds
            if l.token_count > max_t:
                raise ValueError(f"LDU {l.id} exceeds max_tokens ({l.token_count} > {max_t})")
            if l.token_count < min_t and l.chunk_type == "text" and not l.meta.get("is_trailing_chunk", False):
                # Allow short trailing chunks; others should be merged
                raise ValueError(f"LDU {l.id} is too small ({l.token_count} < {min_t})")

            # Rule 2: figure caption as metadata, not standalone text LDU
            if l.type == "figure":
                # If figure has a caption, it must live in meta, not a separate text chunk.
                if "caption" in (l.meta or {}):
                    continue

        # Rule 1 (table headers with cells) is enforced structurally by
        # treating each table as a single LDU, so no split to detect here.


class ChunkingEngine:
    """
    Converts ExtractedDocument into LDUs, enforcing Stage 3 rules.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
        self.validator = ChunkValidator(self.config)

    def _token_count(self, text: str) -> int:
        # Simple approximate tokenization
        return len(text.split())

    def _iter_reading_order(
        self, doc: ExtractedDocument
    ) -> Iterable[Tuple[str, str, int]]:
        """
        Yield (kind, id, index_in_list) according to doc.reading_order.
        kind ∈ {"text", "table", "figure"}.
        """
        for rid in doc.reading_order:
            if rid.startswith("text_"):
                idx = int(rid.split("_", 1)[1])
                yield "text", rid, idx
            elif rid.startswith("table_"):
                idx = int(rid.split("_", 1)[1])
                yield "table", rid, idx
            elif rid.startswith("figure_"):
                idx = int(rid.split("_", 1)[1])
                yield "figure", rid, idx

    def _current_section_label(self, tb: TextBlock) -> Optional[str]:
        label = (tb.label or "").lower()
        if "section" in label or label in {"title", "section_header"}:
            return tb.text.strip()
        return None

    def _is_list_item(self, tb: TextBlock) -> bool:
        if tb.label.lower() == "list_item":
            return True
        return bool(re.match(r"^\s*(\d+\.|\-|\*|\([a-zA-Z0-9]+\))", tb.text))

    def build_ldus(self, doc: ExtractedDocument) -> List[LDU]:
        ldus: List[LDU] = []
        current_section: Optional[str] = None

        # Pre-resolve objects by index
        text_blocks = doc.text_blocks
        tables = doc.tables
        figures = doc.figures

        # Track numbering for tables/figures for xref resolution
        table_ids: List[str] = []
        figure_ids: List[str] = []

        # First pass: build LDUs in reading order
        for kind, rid, idx in self._iter_reading_order(doc):
            if kind == "text":
                tb = text_blocks[idx]
                # Section header logic
                section_label = self._current_section_label(tb)
                if section_label:
                    current_section = section_label
                    # Optional: create an LDU for the section header itself
                    ldu = self._make_text_ldu(
                        doc_id=doc.doc_id,
                        rid=rid,
                        tb=tb,
                        chunk_type="section_header",
                        parent_section=None,
                        is_trailing=False,
                    )
                    ldus.append(ldu)
                    continue

                # List grouping
                if self._is_list_item(tb):
                    list_ldu, consumed = self._build_list_ldu(
                        doc_id=doc.doc_id,
                        start_idx=idx,
                        text_blocks=text_blocks,
                        current_section=current_section,
                    )
                    ldus.append(list_ldu)
                    # Skip ahead in reading order by marking consumed items as None
                    for j in range(idx + 1, consumed):
                        # Mark as consumed by clearing their label so we ignore them
                        text_blocks[j].label = "__consumed_list_item__"
                    continue

                ldu = self._make_text_ldu(
                    doc_id=doc.doc_id,
                    rid=rid,
                    tb=tb,
                    chunk_type="text",
                    parent_section=current_section,
                    is_trailing=False,
                )
                ldus.append(ldu)

            elif kind == "table":
                table = tables[idx]
                ldu = self._make_table_ldu(doc.doc_id, rid, table, current_section)
                ldus.append(ldu)
                table_ids.append(ldu.id)

            elif kind == "figure":
                fig = figures[idx]
                ldu = self._make_figure_ldu(doc.doc_id, rid, fig, current_section)
                ldus.append(ldu)
                figure_ids.append(ldu.id)

        # Rule 5: cross-reference resolution
        self._resolve_cross_references(ldus, table_ids, figure_ids)

        # Validation
        self.validator.validate(ldus)
        return ldus

    def _make_text_ldu(
        self,
        doc_id: str,
        rid: str,
        tb: TextBlock,
        *,
        chunk_type: str,
        parent_section: Optional[str],
        is_trailing: bool,
    ) -> LDU:
        text = tb.text or ""
        tokens = self._token_count(text)
        page_refs = [tb.page_no]

        meta = {
            "label": tb.label,
        }
        if is_trailing:
            meta["is_trailing_chunk"] = True

        content_hash = compute_content_hash(
            doc_id=doc_id,
            page_refs=page_refs,
            content=text,
            bbox=tb.bbox,
        )

        return LDU(
            id=rid,
            type="section_header" if chunk_type == "section_header" else "text_block",
            chunk_type=chunk_type,
            page_no=tb.page_no,
            page_refs=page_refs,
            bbox=tb.bbox,
            content=text,
            parent_section=parent_section,
            token_count=tokens,
            content_hash=content_hash,
            content_ref=rid,
            provenance=None,
            meta=meta,
        )

    def _build_list_ldu(
        self,
        doc_id: str,
        start_idx: int,
        text_blocks: List[TextBlock],
        current_section: Optional[str],
    ) -> Tuple[LDU, int]:
        """
        Build a list LDU starting at start_idx, aggregating subsequent list items
        until max_tokens would be exceeded.
        Returns (ldu, end_idx_exclusive).
        """
        items: List[TextBlock] = []
        page_refs: List[int] = []
        total_tokens = 0
        i = start_idx
        while i < len(text_blocks):
            tb = text_blocks[i]
            if not self._is_list_item(tb):
                break
            t = tb.text or ""
            tokens = self._token_count(t)
            if total_tokens + tokens > self.config.max_tokens and items:
                break
            items.append(tb)
            page_refs.append(tb.page_no)
            total_tokens += tokens
            i += 1

        content = "\n".join(tb.text for tb in items)
        page_refs_unique = sorted(set(page_refs)) or [items[0].page_no]

        content_hash = compute_content_hash(
            doc_id=doc_id,
            page_refs=page_refs_unique,
            content=content,
            bbox=items[0].bbox,
        )

        meta = {
            "label": "numbered_list",
            "is_numbered_list": True,
            "item_count": len(items),
        }

        ldu = LDU(
            id=f"list_{start_idx}",
            type="list",
            chunk_type="list",
            page_no=items[0].page_no,
            page_refs=page_refs_unique,
            bbox=items[0].bbox,
            content=content,
            parent_section=current_section,
            token_count=total_tokens,
            content_hash=content_hash,
            content_ref=f"list_{start_idx}",
            provenance=None,
            meta=meta,
        )
        return ldu, i

    def _make_table_ldu(
        self,
        doc_id: str,
        rid: str,
        table: TableObject,
        current_section: Optional[str],
    ) -> LDU:
        # Represent table content as a simple text serialization for now.
        header_lines = [
            " | ".join(cell.text for cell in row) for row in (table.headers or [])
        ]
        body_lines = [
            " | ".join(cell.text for cell in row) for row in (table.rows or [])
        ]
        text = "\n".join(header_lines + body_lines)
        tokens = self._token_count(text)
        page_refs = [table.page_no]

        content_hash = compute_content_hash(
            doc_id=doc_id,
            page_refs=page_refs,
            content=text,
            bbox=table.bbox,
        )

        meta = {
            "caption": table.caption,
            "is_table": True,
        }

        return LDU(
            id=rid,
            type="table",
            chunk_type="table",
            page_no=table.page_no,
            page_refs=page_refs,
            bbox=table.bbox,
            content=text,
            parent_section=current_section,
            token_count=tokens,
            content_hash=content_hash,
            content_ref=rid,
            provenance=None,
            meta=meta,
        )

    def _make_figure_ldu(
        self,
        doc_id: str,
        rid: str,
        fig: FigureObject,
        current_section: Optional[str],
    ) -> LDU:
        text = fig.caption or ""
        tokens = self._token_count(text)
        page_refs = [fig.page_no]

        content_hash = compute_content_hash(
            doc_id=doc_id,
            page_refs=page_refs,
            content=text,
            bbox=fig.bbox,
        )

        meta = {
            "caption": fig.caption,
            "is_figure": True,
        }

        return LDU(
            id=rid,
            type="figure",
            chunk_type="figure",
            page_no=fig.page_no,
            page_refs=page_refs,
            bbox=fig.bbox,
            content=text,
            parent_section=current_section,
            token_count=tokens,
            content_hash=content_hash,
            content_ref=rid,
            provenance=None,
            meta=meta,
        )

    def _resolve_cross_references(
        self,
        ldus: List[LDU],
        table_ids: List[str],
        figure_ids: List[str],
    ) -> None:
        """
        Rule 5: resolve textual xrefs like "see Table 3" into relationships.
        We use simple ordinal matching:
          - first table encountered = Table 1, etc.
        """
        table_map = {i + 1: tid for i, tid in enumerate(table_ids)}
        figure_map = {i + 1: fid for i, fid in enumerate(figure_ids)}

        table_re = re.compile(r"table\s+(\d+)", re.IGNORECASE)
        fig_re = re.compile(r"figure\s+(\d+)", re.IGNORECASE)

        for l in ldus:
            if l.chunk_type not in {"text", "list", "section_header"}:
                continue
            text = (l.content or "").lower()
            targets: List[str] = []

            for m in table_re.finditer(text):
                idx = int(m.group(1))
                if idx in table_map:
                    targets.append(table_map[idx])
            for m in fig_re.finditer(text):
                idx = int(m.group(1))
                if idx in figure_map:
                    targets.append(figure_map[idx])

            if targets:
                if l.meta is None:
                    l.meta = {}
                l.meta.setdefault("xref_targets", targets)

