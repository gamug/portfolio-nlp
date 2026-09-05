"""Structured judge verdicts, one pydantic model per stage.

The judge is asked to reply with a raw JSON object; ``judges._invoke_with_repair``
extracts the first ``{...}`` and validates it against one of these. ``parse_failed``
is set only by the deterministic fallback when the judge could not return usable
JSON after one repair attempt -- the aggregators in ``metrics`` count those
separately and exclude them from the accuracy numbers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from news_nlp.taxonomy import CATEGORY_SLUGS, OTHER_LABEL

_CATEGORY_CHOICES = (*CATEGORY_SLUGS, OTHER_LABEL)

SentimentLabel = Literal["positive", "negative", "neutral"]


class SentimentVerdict(BaseModel):
    """Judge's read on one ``article_sentiment`` row."""

    agrees: bool
    ideal_label: SentimentLabel
    severity: int = Field(default=0, ge=0, le=2)  # 0 = agree, 1 = adjacent, 2 = opposite
    rationale: str = ""
    parse_failed: bool = False


class CategoryVerdict(BaseModel):
    """Judge's read on one ``article_category`` row (10-label taxonomy + 'other')."""

    agrees: bool
    ideal_slug: str
    severity: int = Field(default=0, ge=0, le=2)
    rationale: str = ""
    parse_failed: bool = False

    @field_validator("ideal_slug")
    @classmethod
    def _coerce_unknown_slug(cls, value: str) -> str:
        # Keep the run alive: an out-of-taxonomy slug is treated as 'other'.
        return value if value in _CATEGORY_CHOICES else OTHER_LABEL


class EntityRef(BaseModel):
    """A reference to one entity by surface text + type (offsets not required --
    the judge only names the wrong / missed ones, not every span)."""

    text: str
    entity_type: str = ""


class NerVerdict(BaseModel):
    """Judge's read on one article's extracted entity set.

    Error-only contract: the judge lists just the *incorrect* predicted spans
    (``wrong``) and the *missed* entities, never a verdict per span -- so the
    reply size scales with the error count, not the (often 100s of) entities.
    ``metrics.aggregate_ner`` derives TP/FP/FN from the predicted count.
    """

    wrong: list[EntityRef] = Field(default_factory=list)
    missed: list[EntityRef] = Field(default_factory=list)
    rationale: str = ""
    parse_failed: bool = False


class SummaryVerdict(BaseModel):
    """Judge's read on one ``article_summary`` row (1-5 scales)."""

    faithfulness: int = Field(default=3, ge=1, le=5)
    coverage: int = Field(default=3, ge=1, le=5)
    conciseness: int = Field(default=3, ge=1, le=5)
    hallucinations: list[str] = Field(default_factory=list)
    rationale: str = ""
    parse_failed: bool = False
