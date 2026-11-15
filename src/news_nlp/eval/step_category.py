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


JudgedRow = dict[
    str, str
]  # keys: judge, second, parse_failed ("True"/"False"), tag, selected, untouched
MODELS: tuple[tuple[str, str], ...] = (
    ("untouched zero-shot (the step until now)", "untouched"),
    ("**fine-tune pinned in `config/category_step.json`**", "selected"),
)
LABELLERS: tuple[tuple[str, str], ...] = (("the assistant's tags (a second LLM labeller)", "tag"),)


def summary(rows: Sequence[JudgedRow]) -> dict[str, float | int]:
    """The headline figures of a judged sample (parse failures excluded); what the pin records."""
    ok = [r for r in rows if r["parse_failed"] != "True"]
    judge = [r["judge"] for r in ok]
    sel = [r["selected"] == j for r, j in zip(ok, judge, strict=True)]
    unt = [r["untouched"] == j for r, j in zip(ok, judge, strict=True)]
    sel_only = sum(s and not u for s, u in zip(sel, unt, strict=True))
    unt_only = sum(u and not s for s, u in zip(sel, unt, strict=True))
    n = len(ok)
    return {
        "n": n,
        "parse_failed": len(rows) - n,
        "selected": sum(sel) / n if n else 0.0,
        "untouched": sum(unt) / n if n else 0.0,
        "selected_only": sel_only,
        "untouched_only": unt_only,
        "sign_test_p": sign_test(sel_only, unt_only),
    }


def report(rows: Sequence[JudgedRow], header: Sequence[str]) -> str:
    """The markdown of a judged sample: agreement of each model with the judge, paired test, per class, confusions."""
    ok = [r for r in rows if r["parse_failed"] != "True"]
    judge = [r["judge"] for r in ok]
    second = [r["second"] or None for r in ok]
    lines = [
        f"# The LLM judge on step 7: {len(rows)} real sub-news, blind (T-209)\n",
        *header,
        f"- {len(ok)} judged, {len(rows) - len(ok)} parse failures (excluded). The judge is a model: agreement with a judge, not truth.",
        "- The judge never saw a model's label. The assistant's tags and the judge are both LLM output: their agreement is not an independent check.\n",
        "| labels | agree with the judge (strict) [95%] | with the judge's second label too |\n|---|---|---|",
    ]
    for name, col in (*MODELS, *LABELLERS):
        labels = [r[col] for r in ok]
        a = agreement(judge, labels, second)
        lo, hi = bootstrap_ci([m == j for m, j in zip(labels, judge, strict=True)])
        lines.append(f"| {name} | {a['strict']:.1%} [{lo:.1%}, {hi:.1%}] | {a['lenient']:.1%} |")
    s = summary(rows)
    lines.append(
        f"\nPaired: the fine-tune agrees with the judge where the untouched model does not on {s['selected_only']} texts, "
        f"the reverse on {s['untouched_only']}; exact sign test p = {s['sign_test_p']:.3f}."
    )
    lines.append(
        f"\n`other`: the judge says `other` on {sum(j == 'other' for j in judge)} texts, the pinned model on "
        f"{sum(r['selected'] == 'other' for r in ok)}."
    )
    lines.append(
        "\n## Per class of the judge (the texts the judge puts in the class)\n\n| class | texts | fine-tune right | untouched right |\n|---|---|---|---|"
    )
    for lab in STEP_LABELS:
        mine = [r for r in ok if r["judge"] == lab]
        if mine:
            lines.append(
                f"| {lab} | {len(mine)} | {sum(r['selected'] == lab for r in mine)} | {sum(r['untouched'] == lab for r in mine)} |"
            )
    pairs: dict[tuple[str, str], int] = {}
    for r in ok:
        if r["judge"] != r["selected"]:
            pairs[(r["judge"], r["selected"])] = pairs.get((r["judge"], r["selected"]), 0) + 1
    lines.append("\nLargest disagreements (judge -> pinned model):")
    lines += [
        f"- `{g}` -> `{p}`: {k}" for (g, p), k in sorted(pairs.items(), key=lambda x: -x[1])[:6]
    ]
    return "\n".join(lines) + "\n"
