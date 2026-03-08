from __future__ import annotations

"""
FactTable extractor: extract simple key-value financial facts into SQLite.

This is a heuristic extractor aimed at statements like:
  - "Revenue was $4.2B in Q3 2024"
  - "Operating income: 3.1 billion USD"
"""

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ..models.ldu import LDU

DEFAULT_DB_PATH = Path(".refinery/facts.db")


@dataclass
class FactRecord:
    doc_id: str
    ldu_id: str
    page_no: int
    key: str
    value: str
    unit: Optional[str]
    period: Optional[str]
    content_hash: Optional[str]


class FactTableExtractor:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    ldu_id TEXT NOT NULL,
                    page_no INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    unit TEXT,
                    period TEXT,
                    content_hash TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _parse_facts_from_ldu(self, l: LDU, doc_id: str) -> List[FactRecord]:
        text = l.content or ""
        page_no = l.page_no
        facts: List[FactRecord] = []

        # Simple patterns for financial metrics
        metric_keywords = ["revenue", "income", "profit", "eps", "earnings", "expenses", "capex", "capital expenditure"]
        money_re = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(billion|million|thousand|bn|m|k|B|M|K)?", re.IGNORECASE)
        period_re = re.compile(r"\b(Q[1-4]\s*\d{4}|\d{4}\s*Q[1-4]|FY\s*\d{4}|fiscal\s+year\s+\d{4})\b", re.IGNORECASE)

        lowered = text.lower()
        for kw in metric_keywords:
            if kw not in lowered:
                continue
            # Look for the first monetary value after the keyword
            kw_idx = lowered.find(kw)
            snippet = text[kw_idx : kw_idx + 200]
            m_val = money_re.search(snippet)
            if not m_val:
                continue
            amount = m_val.group(1)
            unit = m_val.group(2)
            unit_norm = None
            if unit:
                unit_norm = unit.lower()

            m_period = period_re.search(snippet)
            period = m_period.group(0).strip() if m_period else None

            facts.append(
                FactRecord(
                    doc_id=doc_id,
                    ldu_id=l.id,
                    page_no=page_no,
                    key=kw,
                    value=amount,
                    unit=unit_norm,
                    period=period,
                    content_hash=l.content_hash,
                )
            )
        return facts

    def extract_and_store(self, doc_id: str, ldus: Iterable[LDU]) -> int:
        """Extract fact records from LDUs and insert into SQLite. Returns number inserted."""
        records: List[FactRecord] = []
        for l in ldus:
            # Only consider text/list chunks for numeric facts
            if l.chunk_type not in {"text", "list"}:
                continue
            records.extend(self._parse_facts_from_ldu(l, doc_id))

        if not records:
            return 0

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                """
                INSERT INTO facts (doc_id, ldu_id, page_no, key, value, unit, period, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.doc_id,
                        r.ldu_id,
                        r.page_no,
                        r.key,
                        r.value,
                        r.unit,
                        r.period,
                        r.content_hash,
                    )
                    for r in records
                ],
            )
            conn.commit()
        finally:
            conn.close()
        return len(records)

    def structured_query(self, sql: str) -> List[tuple]:
        """Execute a read-only SQL query against the facts table."""
        if not sql.strip().lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed for structured_query")
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(sql)
            return cur.fetchall()
        finally:
            conn.close()

