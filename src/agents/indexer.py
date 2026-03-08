from __future__ import annotations

"""
PageIndex Builder and Query.

Builds a hierarchical PageIndex tree (PageIndexNode) over an ExtractedDocument + LDUs
and supports querying the tree for the top-k most relevant sections for a topic.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_core.prompts import PromptTemplate
except ImportError:
    HuggingFaceEndpoint = None
    PromptTemplate = None

logger = logging.getLogger(__name__)

from ..models.extracted_document import ExtractedDocument, TextBlock
from ..models.ldu import LDU
from ..models.page_index import PageIndexNode


@dataclass
class PageIndexConfig:
    max_summary_chars: int = 2000  # limit text passed into summarizer
    summary_sentences: int = 3


class SimpleSummarizer:
    """
    Very cheap summarizer fallback: takes the first N characters and trims at a sentence
    boundary. This avoids any external dependency if no LLM is configured.
    """

    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    async def summarize(self, title: str, text: str, sentences: int = 3) -> str:
        if not text:
            return title
        snippet = text[: self.max_chars]
        parts = [p.strip() for p in snippet.split(".") if p.strip()]
        if not parts:
            return snippet.strip()
        return ". ".join(parts[:sentences]).strip() + "."


class LLMSummarizer:
    """
    LLM-based summarizer using HuggingFace serverless inference.
    Falls back to SimpleSummarizer if initialization or API call fails.
    """

    def __init__(self, max_chars: int = 3000) -> None:
        self.max_chars = max_chars
        self.fallback = SimpleSummarizer(max_chars)
        self.llm = None
        
        hf_token = os.environ.get("HUGGINGFACE_ACCESS_KEY") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if hf_token and HuggingFaceEndpoint:
            try:
                # Using a fast instruction model
                self.llm = HuggingFaceEndpoint(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    task="text-generation",
                    max_new_tokens=150,
                    do_sample=False,
                    huggingfacehub_api_token=hf_token,
                )
                self.prompt = PromptTemplate.from_template(
                    "<|system|>\nYou are a concise document analyst. Summarize the text in 1-2 short sentences.\n"
                    "<|user|>\nSection Title: {title}\nText: {text}\n<|assistant|>\n"
                )
            except Exception as e:
                logger.warning(f"Could not init HuggingFace Endpoint for summarizer: {e}")
                self.llm = None

    async def summarize(self, title: str, text: str, sentences: int = 3) -> str:
        if not text:
            return title
        
        if not self.llm:
            return await self.fallback.summarize(title, text, sentences)
            
        snippet = text[: self.max_chars]
        try:
            # We use an executor because running LLM chain might be blocking
            import asyncio
            loop = asyncio.get_running_loop()
            chain = self.prompt | self.llm
            result = await loop.run_in_executor(None, chain.invoke, {"title": title, "text": snippet})
            # Clean up potential leading/trailing whitespace
            return result.strip()
        except Exception as e:
            logger.warning(f"LLM Summarization failed, falling back to basic extraction. Error: {e}")
            return await self.fallback.summarize(title, text, sentences)


class PageIndexBuilder:
    """
    Build a PageIndexNode tree from an ExtractedDocument + LDUs.

    For now we build a two-level tree:
      root (document) -> section nodes derived from section headers.
    """

    def __init__(
        self,
        config: Optional[PageIndexConfig] = None,
        summarizer: Optional[LLMSummarizer] = None,
    ) -> None:
        self.config = config or PageIndexConfig()
        self.summarizer = summarizer or LLMSummarizer(self.config.max_summary_chars)

    def _section_headers(self, doc: ExtractedDocument) -> List[Tuple[int, TextBlock]]:
        """
        Return list of (index, TextBlock) that look like section headers.
        """
        headers: List[Tuple[int, TextBlock]] = []
        for idx, tb in enumerate(doc.text_blocks):
            label = (tb.label or "").lower()
            if label in {"title", "section_header"} or "section" in label:
                headers.append((idx, tb))
        return headers

    def _assign_ldus_to_sections(
        self,
        doc: ExtractedDocument,
        ldus: List[LDU],
        headers: List[Tuple[int, TextBlock]],
    ) -> List[PageIndexNode]:
        """
        Build flat list of PageIndexNode (sections) and assign LDU ids based on parent_section.
        """
        sections: List[PageIndexNode] = []
        if not headers:
            # Single root section covering full doc
            pages = {l.page_no for l in ldus}
            node = PageIndexNode(
                id="section_0",
                title=doc.doc_id,
                page_start=min(pages) if pages else 1,
                page_end=max(pages) if pages else 1,
                ldu_ids=[l.id for l in ldus],
            )
            sections.append(node)
            return sections

        # Map section titles to node ids
        for i, (_, tb) in enumerate(headers):
            title = tb.text.strip() or f"Section {i+1}"
            sections.append(
                PageIndexNode(
                    id=f"section_{i}",
                    title=title,
                    page_start=tb.page_no,
                    page_end=tb.page_no,
                )
            )

        # Estimate page_end per section: use next header's page_start - 1
        for i in range(len(sections) - 1):
            sections[i].page_end = max(sections[i].page_start, sections[i + 1].page_start)
        # Last section extends to last page with any LDU
        last_pages = [l.page_no for l in ldus] or [sections[-1].page_start]
        sections[-1].page_end = max(last_pages)

        # Assign LDUs based on parent_section (string match to header title)
        title_to_node = {s.title: s for s in sections}
        for l in ldus:
            if l.parent_section and l.parent_section in title_to_node:
                title_to_node[l.parent_section].ldu_ids.append(l.id)
            else:
                # Fallback: put under the earliest section whose page range covers this LDU
                for s in sections:
                    if s.page_start <= l.page_no <= s.page_end:
                        s.ldu_ids.append(l.id)
                        break

        return sections

    def _collect_text_for_section(self, section: PageIndexNode, ldus_by_id: dict[str, LDU]) -> str:
        texts: List[str] = []
        for lid in section.ldu_ids:
            l = ldus_by_id.get(lid)
            if l and l.chunk_type in {"text", "list", "section_header"}:
                texts.append(l.content or "")
        return "\n".join(texts)

    def _data_types_for_section(self, section: PageIndexNode, ldus_by_id: dict[str, LDU]) -> List[str]:
        types: set[str] = set()
        for lid in section.ldu_ids:
            l = ldus_by_id.get(lid)
            if not l:
                continue
            if l.chunk_type == "table":
                types.add("tables")
            if l.chunk_type == "figure":
                types.add("figures")
            if "equation" in (l.meta or {}).get("label", "").lower():
                types.add("equations")
        return sorted(types)

    def _key_entities(self, text: str, max_entities: int = 10) -> List[str]:
        """
        Very lightweight key-entity extractor: collect capitalized noun-ish phrases.
        This is intentionally simple to avoid adding heavy NLP dependencies.
        """
        candidates = set()
        # Split on non-word boundaries and find tokens starting with capital letter
        tokens = re.findall(r"[A-Z][a-zA-Z0-9_]+(?:\s+[A-Z][a-zA-Z0-9_]+)*", text)
        for t in tokens:
            cleaned = t.strip()
            if len(cleaned.split()) <= 6:
                candidates.add(cleaned)
            if len(candidates) >= max_entities:
                break
        return sorted(candidates)

    async def build(self, doc: ExtractedDocument, ldus: List[LDU]) -> PageIndexNode:
        """
        Build a PageIndex tree for the document.
        """
        headers = self._section_headers(doc)
        sections = self._assign_ldus_to_sections(doc, ldus, headers)

        ldus_by_id = {l.id: l for l in ldus}

        # Summaries, entities, and data types
        for s in sections:
            text = self._collect_text_for_section(s, ldus_by_id)
            s.data_types_present = self._data_types_for_section(s, ldus_by_id)
            s.key_entities = self._key_entities(text)
            s.summary = await self.summarizer.summarize(s.title, text, sentences=self.config.summary_sentences)

        # Root node represents the whole doc
        all_pages = [l.page_no for l in ldus] or [1]
        root = PageIndexNode(
            id="root",
            title=doc.doc_id,
            page_start=min(all_pages),
            page_end=max(all_pages),
            child_sections=sections,
            ldu_ids=[l.id for l in ldus],
            data_types_present=self._data_types_for_section(
                PageIndexNode(id="_tmp", title="", page_start=1, page_end=1, ldu_ids=[l.id for l in ldus]),
                ldus_by_id,
            ),
        )
        # Root-level entities & summary
        full_text = "\n".join(l.content or "" for l in ldus)
        root.key_entities = self._key_entities(full_text)
        root.summary = await self.summarizer.summarize(root.title, full_text, sentences=self.config.summary_sentences)
        
        # Save to disk as JSON
        output_dir = Path(".refinery/pageindex")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Handle doc_id ending with .pdf to name files .json
        base_name = doc.doc_id.replace(".pdf", "")
        file_path = output_dir / f"{base_name}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(root.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save PageIndex to {file_path}: {e}")
            
        return root


def _section_text_for_scoring(node: PageIndexNode) -> str:
    base = node.title or ""
    if node.summary:
        base += " " + node.summary
    return base.lower()


def query_page_index(root: PageIndexNode, topic: str, top_k: int = 3) -> List[PageIndexNode]:
    """
    Given a topic string, traverse the PageIndex tree and return the top-k most
    relevant sections based on simple keyword overlap scoring.
    """
    topic_tokens = {t for t in re.findall(r"\\w+", topic.lower()) if len(t) > 2}
    scored: List[Tuple[float, PageIndexNode]] = []

    def visit(node: PageIndexNode) -> None:
        text = _section_text_for_scoring(node)
        tokens = {t for t in re.findall(r"\\w+", text) if len(t) > 2}
        if not tokens:
            score = 0.0
        else:
            overlap = topic_tokens & tokens
            score = len(overlap) / max(len(topic_tokens), 1)
        scored.append((score, node))
        for c in node.child_sections:
            visit(c)

    visit(root)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for s, n in scored[:top_k] if s > 0.0]

