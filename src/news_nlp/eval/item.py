"""One row to judge: the text and the model's stored prediction. Kept from the v1 sampling module (T-220, which sampled the v1 tables and is gone):
``judges`` and the summary-judging scripts still take this shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalItem:
    """One row to judge: the source text plus the model's stored prediction."""

    article_id: int
    bucket: str  # "low_conf" | "representative" | stage-specific "target_<x>"
    stratum_population: int  # N_h: population size of this item's bucket
    title: str
    body_text: str
    prediction: dict[str, Any]
