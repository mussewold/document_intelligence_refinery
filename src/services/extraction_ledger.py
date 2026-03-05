"""
Extraction ledger: append-only JSONL log of every extraction run.

Each line is a JSON object with strategy_used, confidence_score, cost_estimate,
processing_time, and related fields for auditing and cost tracking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path(".refinery/extraction_ledger.jsonl")


def append_extraction(
    doc_id: str,
    strategy_used: str,
    *,
    confidence_score: Optional[float] = None,
    cost_estimate: Optional[float] = None,
    processing_time_seconds: float = 0.0,
    escalated: bool = False,
    escalation_path: Optional[list[str]] = None,
    ledger_path: Optional[Path] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Append one extraction record to the ledger file (one JSON object per line).
    Creates .refinery if it does not exist.
    """
    path = ledger_path or DEFAULT_LEDGER_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "doc_id": doc_id,
        "strategy_used": strategy_used,
        "confidence_score": confidence_score,
        "cost_estimate": cost_estimate,
        "processing_time_seconds": round(processing_time_seconds, 3),
        "escalated": escalated,
        "escalation_path": escalation_path or [],
    }
    if extra:
        record["extra"] = extra

    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        logger.warning("Failed to write extraction ledger: %s", e)
