"""
Strategy C — Vision-Augmented: budget guard and model config.

Tracks token spend per document and enforces a configurable cap.
Estimated cost is logged; no single document may exceed the budget cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Rough cost per 1K tokens (USD) for budget-aware selection and logging.
# Update from https://openrouter.ai/docs/models when needed.
OPENROUTER_COST_PER_1K: dict[str, tuple[float, float]] = {
    "google/gemini-2.0-flash-exp:free": (0.0, 0.0),
    "google/gemini-2.0-flash-001": (0.000_075, 0.000_30),   # input, output per 1K
    "openai/gpt-4o-mini": (0.000_15, 0.000_60),              # approximate per 1K
}

DEFAULT_VISION_MODEL = "google/gemini-2.0-flash-001"
BUDGET_CAP_TOKENS_DEFAULT = 500_000  # per document


@dataclass
class BudgetGuard:
    """
    Tracks token usage and estimated cost for a single document.
    Enforces a configurable cap; add_usage() raises if cap would be exceeded.
    """

    max_tokens: int = BUDGET_CAP_TOKENS_DEFAULT
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    _input_tokens: int = field(default=0, repr=False)
    _output_tokens: int = field(default=0, repr=False)

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        in_cost = (self._input_tokens / 1000.0) * self.cost_per_1k_input
        out_cost = (self._output_tokens / 1000.0) * self.cost_per_1k_output
        return round(in_cost + out_cost, 6)

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record usage and raise if over cap. Call after each OpenRouter request."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative")
        new_total = self._input_tokens + self._output_tokens + input_tokens + output_tokens
        if new_total > self.max_tokens:
            raise RuntimeError(
                f"Document budget cap exceeded: {new_total} tokens (cap {self.max_tokens}). "
                "Aborting to avoid unbounded cost."
            )
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        logger.info(
            "Vision extraction usage: input=%s output=%s total=%s estimated_cost_usd=%s",
            self._input_tokens,
            self._output_tokens,
            self.total_tokens,
            self.estimated_cost_usd,
        )

    def would_exceed(self, input_tokens: int, output_tokens: int) -> bool:
        """Return True if adding this usage would exceed the cap."""
        return (
            self._input_tokens + self._output_tokens + input_tokens + output_tokens
            > self.max_tokens
        )


def get_model_costs(model_id: str) -> tuple[float, float]:
    """Return (cost_per_1k_input, cost_per_1k_output) for the model. Default 0,0."""
    t = OPENROUTER_COST_PER_1K.get(model_id, (0.0, 0.0))
    return t
