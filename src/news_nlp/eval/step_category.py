"""LLM-as-judge for step 7 (``--step category``): an independent label per (text, company), blind.

The v1 ``category`` eval (``judges.judge_category``) judges ``article_category`` rows of the v1 chain and scores a
candidate through the v1 two-level zero-shot stage (``category_stage``), not through ``news_category_stage``: it would not
measure the model step 7 runs (T-209, AI behavior #1). This module is the judge for step 7's own reading: 12 labels, the
text the step reads and the company it is read with. The judge is **blind**: it labels from the text and the company
alone, and never sees a model's answer, so two models can be compared against the same judge labels with one call per text
(a verdict that shows the prediction anchors the judge on it).

Kept free of ``strands`` / ``mlflow`` imports (the agent is the structural ``JudgeAgent`` protocol), so it is hermetic.
The labels are kept in step with ``news_category_stage.LABELS`` by a test.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel, field_validator

from news_nlp.eval.judges import JudgeAgent, _invoke_with_repair, load_prompt

PROMPT_STAGE = "category_step"
STEP_LABELS: tuple[str, ...] = (
    "earnings_performance",
    "mergers_acquisitions",
    "leadership_governance",
    "legal_regulatory",
    "product_innovation",
    "capital_shareholder_returns",
    "labor_human_capital",
    "market_analyst_sentiment",
    "partnerships_business_dev",
    "reputational_harm",
    "investor_activity",
    "other",
)


class StepCategoryVerdict(BaseModel):
    """The judge's own label for one (text, company). An out-of-taxonomy slug is a parse failure, not ``other``."""

    ideal_slug: str
    second_slug: str | None = None
    rationale: str = ""
    parse_failed: bool = False

    @field_validator("ideal_slug")
    @classmethod
    def _known(cls, value: str) -> str:
        if value not in STEP_LABELS:
            raise ValueError(f"unknown label {value!r}")
        return value

    @field_validator("second_slug")
    @classmethod
    def _known_or_none(cls, value: str | None) -> str | None:
        return value if value in STEP_LABELS else None


def user_prompt(company: str, text: str) -> str:
    """What the judge reads: the company and the text the step reads, nothing of any model."""
    return f"COMPANY: {company}\n\nTEXT\n{text}\n\nReturn your JSON verdict."


def judge_step_category(agent: JudgeAgent, company: str, text: str) -> StepCategoryVerdict:
    fallback = StepCategoryVerdict(
        ideal_slug="other",
        rationale="judge reply could not be parsed",
        parse_failed=True,
    )
    return _invoke_with_repair(
        agent, user_prompt(company, text), StepCategoryVerdict, fallback=fallback
    )


def system_prompt() -> str:
    return load_prompt(PROMPT_STAGE)


def agreement(
    judge: Sequence[str], model: Sequence[str], second: Sequence[str | None]
) -> dict[str, float]:
    """Strict (the model's label is the judge's) and lenient (or the judge's second label) agreement of one model."""
    n = len(judge)
    strict = sum(m == j for m, j in zip(model, judge, strict=True))
    lenient = sum(m in (j, s) for m, j, s in zip(model, judge, second, strict=True))
    return {"n": n, "strict": strict / n if n else 0.0, "lenient": lenient / n if n else 0.0}


def bootstrap_ci(
    hits: Sequence[bool], *, boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval of a proportion (resampling the texts)."""
    rng = random.Random(seed)  # noqa: S311 -- a seeded resampling, not a secret
    n = len(hits)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(sum(hits[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    return (means[int(boot * alpha / 2)], means[min(boot - 1, int(boot * (1 - alpha / 2)))])


def sign_test(a_only: int, b_only: int) -> float:
    """Two-sided exact sign test on the texts where exactly one of two models agrees with the judge."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / float(2**n)
    return min(1.0, 2 * tail)
