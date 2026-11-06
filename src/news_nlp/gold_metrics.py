"""Figures from the owner's tagged files of Work item 18 (PLAN D27, D32): intervals, the
re-weighted sentiment figures of the stratified 180, the label check and the faithfulness check.

The 180-text sentiment gold is drawn 60 per *predicted* class, so its raw agreement is not the
production agreement: every row weighs ``population / drawn`` of its stratum, and the interval of
each figure is a bootstrap that resamples inside each stratum. Everything is seeded.
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from news_nlp import triage

Z95 = 1.96
RESAMPLES = 2000
CONTINUATION_BAR = 0.80  # D27: at most this share right -> train on `lead` only
SENTIMENT_CLASSES = ("positive", "negative", "neutral")


class MarkedFileError(ValueError):
    """An owner's file is not usable: a missing column, an unknown value or an unmarked row."""


@dataclass(frozen=True)
class Figure:
    """A point estimate with its 95% interval."""

    point: float
    low: float
    high: float

    def judge(self, bar: float) -> str:
        """``pass`` when the whole interval clears *bar*, ``fail`` when it is all below, else
        ``not decided`` (D32: reported with the interval, the cut still goes, D34)."""
        if self.low >= bar:
            return "pass"
        if self.high < bar:
            return "fail"
        return "not decided"

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


def wilson(successes: int, n: int, z: float = Z95) -> Figure:
    """Wilson score interval of a proportion; ``0 of 0`` has no estimate (NaN)."""
    low, high = triage.wilson(successes, n, z)
    return Figure(successes / n if n else math.nan, low, high)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _bootstrap(
    strata: Mapping[str, list[Mapping[str, str]]],
    statistic: Callable[[Mapping[str, list[Mapping[str, str]]]], float],
    seed: int,
    resamples: int,
) -> Figure:
    point = statistic(strata)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws: list[float] = []
    for _ in range(resamples):
        sample = {k: [rng.choice(v) for _ in v] for k, v in strata.items()}
        value = statistic(sample)
        if not math.isnan(value):
            draws.append(value)
    if not draws or math.isnan(point):
        return Figure(point, math.nan, math.nan)
    return Figure(point, _percentile(draws, 0.025), _percentile(draws, 0.975))


@dataclass(frozen=True)
class SentimentReport:
    agreement: Figure
    negative_precision: Figure
    negative_recall: Figure
    raw_agreement: Figure  # not re-weighted: for comparison only
    drawn: dict[str, int]


def sentiment_report(
    rows: Sequence[Mapping[str, str]],
    populations: Mapping[str, int],
    *,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> SentimentReport:
    """*rows* carry ``predicted`` (the stratum) and ``gold`` (the owner's class). *populations* is
    the size of each stratum in the pool the draw came from."""
    r = stratified_report(
        rows, populations, label="negative", stratum_key="predicted", seed=seed, resamples=resamples
    )
    return SentimentReport(r.agreement, r.precision, r.recall, r.raw_agreement, r.drawn)


@dataclass(frozen=True)
class StratifiedReport:
    agreement: Figure
    precision: Figure  # of *label*: of the rows predicted *label*, the share gold calls *label*
    recall: Figure  # of *label*: of the rows gold calls *label*, the share predicted *label*
    raw_agreement: Figure
    drawn: dict[str, int]
    gold_count: int  # rows the owner calls *label* in the sample (D33: under 40, not gated)


def stratified_report(
    rows: Sequence[Mapping[str, str]],
    populations: Mapping[str, int],
    *,
    label: str,
    stratum_key: str,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> StratifiedReport:
    """Agreement and the precision / recall of *label* of a draw stratified on *stratum_key*, each
    row weighing ``population / drawn`` of its stratum; the intervals are a seeded bootstrap that
    resamples inside each stratum. *rows* carry ``predicted`` and ``gold``."""
    strata: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        strata.setdefault(row[stratum_key], []).append(row)
    unknown = sorted(set(strata) - set(populations))
    if unknown:
        raise MarkedFileError(f"no population for the strata {unknown}")

    def weighted(
        s: Mapping[str, list[Mapping[str, str]]], keep: Callable[[Mapping[str, str]], bool]
    ) -> float:
        return sum(populations[k] / len(v) for k, v in s.items() for r in v if keep(r))

    def agreement(s: Mapping[str, list[Mapping[str, str]]]) -> float:
        total = weighted(s, lambda r: True)
        return weighted(s, lambda r: r["gold"] == r["predicted"]) / total if total else math.nan

    def precision(s: Mapping[str, list[Mapping[str, str]]]) -> float:
        predicted = weighted(s, lambda r: r["predicted"] == label)
        hit = weighted(s, lambda r: r["predicted"] == label and r["gold"] == label)
        return hit / predicted if predicted else math.nan

    def recall(s: Mapping[str, list[Mapping[str, str]]]) -> float:
        actual = weighted(s, lambda r: r["gold"] == label)
        hit = weighted(s, lambda r: r["predicted"] == label and r["gold"] == label)
        return hit / actual if actual else math.nan

    raw = wilson(sum(r["gold"] == r["predicted"] for r in rows), len(rows))
    return StratifiedReport(
        agreement=_bootstrap(strata, agreement, seed, resamples),
        precision=_bootstrap(strata, precision, seed + 1, resamples),
        recall=_bootstrap(strata, recall, seed + 2, resamples),
        raw_agreement=raw,
        drawn={k: len(v) for k, v in strata.items()},
        gold_count=sum(r["gold"] == label for r in rows),
    )


def paired_difference(
    a: Sequence[bool], b: Sequence[bool], *, seed: int = 20261106, resamples: int = RESAMPLES
) -> Figure:
    """Share of items *a* gets right minus the share *b* gets right, on the same items, with a
    bootstrap interval over the items (the extractor against the baseline on the same 30)."""
    if len(a) != len(b) or not a:
        raise ValueError("two non-empty lists of the same length are needed")
    diffs = [int(x) - int(y) for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws = [sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples)]
    return Figure(sum(diffs) / len(diffs), _percentile(draws, 0.025), _percentile(draws, 0.975))


def read_marked(
    path: Path, column: str, allowed: Sequence[str], required: Sequence[str] = ("article_id",)
) -> list[dict[str, str]]:
    """The rows of an owner's tagging file with *column* checked: every row marked, every value one
    of *allowed* (case and spaces ignored). Raises :class:`MarkedFileError` naming the first rows
    at fault, so a half-tagged or mistyped file is never turned into a figure."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    if not rows:
        raise MarkedFileError(f"{path.name} has no rows")
    missing = [c for c in (column, *required) if c not in rows[0]]
    if missing:
        raise MarkedFileError(f"{path.name} lacks the columns {missing}")
    empty, wrong = [], []
    for r in rows:
        value = (r[column] or "").strip().lower()
        r[column] = value
        if not value:
            empty.append(r["article_id"])
        elif value not in allowed:
            wrong.append(f"{r['article_id']}={value!r}")
    if empty or wrong:
        raise MarkedFileError(
            f"{path.name}: {len(empty)} rows without {column} (first: {empty[:5]}), "
            f"{len(wrong)} with a value outside {list(allowed)} (first: {wrong[:5]})"
        )
    return rows


def label_check_report(rows: Sequence[Mapping[str, str]]) -> dict[str, Figure]:
    """Precision of the automatic labels per kind (``lead`` / ``continuation``) from the marked
    ``correct`` column (yes / no), with Wilson intervals; ``<kind> 2022+`` repeats it for the rows
    published from 2022 on (``pub_date``), the window the backtest cut serves."""
    out: dict[str, Figure] = {}
    for kind in ("lead", "continuation"):
        part = [r for r in rows if r["kind"] == kind]
        out[kind] = wilson(sum(r["correct"] == "yes" for r in part), len(part))
        late = [r for r in part if (r.get("pub_date") or "")[:4] >= "2022"]
        out[f"{kind} 2022+"] = wilson(sum(r["correct"] == "yes" for r in late), len(late))
    return out


def train_on_lead_only(report: Mapping[str, Figure]) -> bool:
    """D27's rule, fixed before the owner read the sample: the `continuation` labels are dropped
    from training when at most :data:`CONTINUATION_BAR` of them are right."""
    return report["continuation"].point <= CONTINUATION_BAR


def faithfulness_report(
    marks: Mapping[str, bool], extractors_of: Mapping[str, Sequence[str]]
) -> dict[str, Figure]:
    """Share of faithful texts per extractor. *marks* is the owner's yes / no by text key and
    *extractors_of* says which extractors produced each text (one text can be both's)."""
    names = sorted({e for v in extractors_of.values() for e in v})
    out: dict[str, Figure] = {}
    for name in names:
        keys = [k for k, v in extractors_of.items() if name in v]
        out[name] = wilson(sum(marks[k] for k in keys), len(keys))
    return out
