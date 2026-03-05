from __future__ import annotations

import asyncio
from typing import Any, List

from docling.document_converter import DocumentConverter
from docling_core.types.doc import BoundingBox as DoclingBoundingBox
from docling_core.types.doc import DocItem, DoclingDocument, ProvenanceItem, TableItem

from ..models.document_profile import LayoutComplexity
from ..models.extracted_document import (
    ExtractedDocument,
    FigureObject,
    TableCell,
    TableObject,
    TextBlock,
)
from ..models.layout_extractor import LayoutExtractor as LayoutExtractorModel
from ..models.provenance import BoundingBox
from ..services.triage_services.artifact_loader import DocumentArtifacts


class DoclingDocumentAdapter:
    """
    Adapter that converts a DoclingDocument into the internal ExtractedDocument
    schema used by the Refinery.
    """

    def to_extracted_document(self, doc: DoclingDocument, doc_id: str) -> ExtractedDocument:
        text_blocks: List[TextBlock] = []
        tables: List[TableObject] = []
        figures: List[FigureObject] = []
        reading_order: List[str] = []

        # Text items in document reading order
        for idx, text_item in enumerate(doc.texts):
            text = getattr(text_item, "text", "") or ""
            prov = self._first_prov(text_item)
            page_no, bbox = self._page_and_bbox_from_prov(prov)

            label = getattr(text_item, "label", None)
            label_str = getattr(label, "value", None) or str(label) or "paragraph"

            text_blocks.append(
                TextBlock(
                    text=text,
                    page_no=page_no,
                    bbox=bbox,
                    label=label_str,
                )
            )
            reading_order.append(f"text_{idx}")

        # Tables as structured JSON
        for t_idx, table_item in enumerate(doc.tables):
            prov = self._first_prov(table_item)
            page_no, table_bbox = self._page_and_bbox_from_prov(prov)

            caption_text = None
            caption = getattr(table_item, "caption", None)
            if caption is not None:
                caption_text = getattr(caption, "text", None) or str(caption)

            headers: List[List[TableCell]] = []
            rows: List[List[TableCell]] = []

            data = getattr(table_item, "data", None)
            if data is not None:
                # data is a TableData instance; its `grid` property returns a 2D list
                # of docling_core.types.doc.TableCell objects.
                grid = data.grid
                for r_idx, row_cells in enumerate(grid):
                    current_row: List[TableCell] = []
                    for c_idx, cell in enumerate(row_cells):
                        is_header = bool(
                            getattr(cell, "column_header", False)
                            or getattr(cell, "row_header", False)
                            or getattr(cell, "row_section", False)
                        )
                        current_row.append(
                            TableCell(
                                text=getattr(cell, "text", "") or "",
                                is_header=is_header,
                                row_index=r_idx,
                                col_index=c_idx,
                                row_span=getattr(cell, "row_span", 1),
                                col_span=getattr(cell, "col_span", 1),
                            )
                        )
                    rows.append(current_row)

                # Split header vs body rows: any leading rows where at least one
                # cell is marked as header.
                header_rows: List[List[TableCell]] = []
                body_rows: List[List[TableCell]] = []
                header_phase = True
                for row in rows:
                    if header_phase and any(c.is_header for c in row):
                        header_rows.append(row)
                    else:
                        header_phase = False
                        body_rows.append(row)

                headers = header_rows
                rows = body_rows

            tables.append(
                TableObject(
                    headers=headers,
                    rows=rows,
                    caption=caption_text,
                    page_no=page_no,
                    bbox=table_bbox,
                )
            )
            reading_order.append(f"table_{t_idx}")

        # Figures/pictures with captions
        for f_idx, pic in enumerate(doc.pictures):
            prov = self._first_prov(pic)
            page_no, fig_bbox = self._page_and_bbox_from_prov(prov)

            caption_text = None
            caption = getattr(pic, "caption", None)
            if caption is not None:
                caption_text = getattr(caption, "text", None) or str(caption)

            figures.append(
                FigureObject(
                    caption=caption_text,
                    page_no=page_no,
                    bbox=fig_bbox,
                )
            )
            reading_order.append(f"figure_{f_idx}")

        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
            metadata={
                "engine": "docling",
                "strategy": "layout_aware",
                "docling_version": getattr(doc, "version", None),
                "num_pages": len(getattr(doc, "pages", {}) or {}),
            },
        )

    def _first_prov(self, item: DocItem | TableItem | Any) -> ProvenanceItem | None:
        prov_list = getattr(item, "prov", None) or []
        if prov_list:
            return prov_list[0]
        return None

    def _page_and_bbox_from_prov(
        self, prov: ProvenanceItem | None
    ) -> tuple[int, BoundingBox]:
        if prov is None or getattr(prov, "bbox", None) is None:
            # Fallback: unknown page/bbox
            return 1, BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

        dl_bbox: DoclingBoundingBox = prov.bbox

        # Docling uses (l, t, r, b); map to our (x0, y0, x1, y1).
        x0 = float(getattr(dl_bbox, "l", 0.0))
        y0 = float(getattr(dl_bbox, "b", 0.0))
        x1 = float(getattr(dl_bbox, "r", 0.0))
        y1 = float(getattr(dl_bbox, "t", 0.0))

        page_no = int(getattr(prov, "page_no", 1) or 1)

        return page_no, BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


class DoclingLayoutExtractor:
    """
    Strategy B — Layout-Aware Tool (Docling).

    Triggers when:
      - layout_complexity in {multi_column, table_heavy}
      - OR origin_type == "mixed"

    It uses Docling's DocumentConverter to obtain a DoclingDocument and
    normalizes it via DoclingDocumentAdapter into the internal schema.
    The result is wrapped in a pydantic LayoutExtractor model.
    """

    def __init__(self) -> None:
        self._converter = DocumentConverter()
        self._adapter = DoclingDocumentAdapter()

    @staticmethod
    def is_applicable(
        origin_type: str,
        layout_complexity: LayoutComplexity,
    ) -> bool:
        return origin_type == "mixed" or layout_complexity in {
            LayoutComplexity.multi_column,
            LayoutComplexity.table_heavy,
        }

    async def extract(
        self,
        artifacts: DocumentArtifacts,
        doc_id: str,
        origin_type: str,
        layout_complexity: LayoutComplexity,
    ) -> LayoutExtractorModel:
        """
        Run Docling on the source PDF and return a LayoutExtractor model
        containing the normalized ExtractedDocument.
        """
        loop = asyncio.get_running_loop()

        # DocumentConverter.convert is synchronous and can be relatively heavy,
        # so we run it in a thread pool.
        def _convert() -> DoclingDocument:
            result = self._converter.convert(str(artifacts.pdf_path))
            return result.document

        dl_doc: DoclingDocument = await loop.run_in_executor(None, _convert)

        extracted_doc = self._adapter.to_extracted_document(dl_doc, doc_id=doc_id)

        return LayoutExtractorModel(
            engine="docling",
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            document=extracted_doc,
            metadata={
                "source_path": str(artifacts.pdf_path),
            },
        )

