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
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from news_nlp import triage

Z95 = 1.96
RESAMPLES = 2000
LATE_FROM_YEAR = 2022  # the backtest window (D34)
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
        """``pass`` when the whole interval clears *bar* (its low end is at or above it), ``fail``
        when the whole interval is below it, else ``not decided``: an interval that straddles the
        bar or only touches it from below (``high == bar``) is not a pass or a fail. The point
        estimate decides nothing alone (D32: reported with the interval, the cut still goes, D34)."""
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
    fixed: frozenset[str] = frozenset(),
) -> Figure:
    """Seeded percentile interval of *statistic*, resampling inside each stratum. A stratum in
    *fixed* was drawn whole (a census): it has no sampling error and is not resampled."""
    point = statistic(strata)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws: list[float] = []
    for _ in range(resamples):
        sample = {k: v if k in fixed else [rng.choice(v) for _ in v] for k, v in strata.items()}
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
    census: tuple[str, ...] = ()  # strata drawn whole: held fixed in the bootstrap


def sentiment_report(
    rows: Sequence[Mapping[str, str]],
    populations: Mapping[str, int],
    *,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> SentimentReport:
    """*rows* carry ``predicted`` and ``gold`` (the owner's class), and ``stratum`` when the draw has
    more than one part (otherwise ``predicted`` is the stratum). *populations* is the size of each
    stratum in the pool the draw came from."""
    key = "stratum" if rows and "stratum" in rows[0] else "predicted"
    r = stratified_report(
        rows, populations, label="negative", stratum_key=key, seed=seed, resamples=resamples
    )
    return SentimentReport(r.agreement, r.precision, r.recall, r.raw_agreement, r.drawn, r.census)


@dataclass(frozen=True)
class StratifiedReport:
    agreement: Figure
    precision: Figure  # of *label*: of the rows predicted *label*, the share gold calls *label*
    recall: Figure  # of *label*: of the rows gold calls *label*, the share predicted *label*
    raw_agreement: Figure
    drawn: dict[str, int]
    gold_count: int  # rows the owner calls *label* in the sample (D33: under 40, not gated)
    census: tuple[str, ...] = ()  # strata drawn whole (drawn >= population): not resampled


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
    resamples inside each stratum, except a stratum drawn whole (a census: no sampling error there).
    *rows* carry ``predicted`` and ``gold``."""
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
    census = frozenset(k for k, v in strata.items() if len(v) >= populations[k])
    return StratifiedReport(
        agreement=_bootstrap(strata, agreement, seed, resamples, census),
        precision=_bootstrap(strata, precision, seed + 1, resamples, census),
        recall=_bootstrap(strata, recall, seed + 2, resamples, census),
        raw_agreement=raw,
        drawn={k: len(v) for k, v in strata.items()},
        gold_count=sum(r["gold"] == label for r in rows),
        census=tuple(sorted(census)),
    )


def _cluster_ratio(clusters: Sequence[tuple[float, int]], seed: int, resamples: int) -> Figure:
    """The ratio ``sum(first) / sum(second)`` over *clusters*, with a bootstrap interval that
    resamples whole clusters: the units that are independent (the bodies), not their texts."""
    total = sum(n for _, n in clusters)
    if not total:
        return Figure(math.nan, math.nan, math.nan)
    point = sum(x for x, _ in clusters) / total
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws: list[float] = []
    for _ in range(resamples):
        sample = rng.choices(clusters, k=len(clusters))
        n = sum(c for _, c in sample)
        if n:
            draws.append(sum(x for x, _ in sample) / n)
    return Figure(point, _percentile(draws, 0.025), _percentile(draws, 0.975))


def paired_difference(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    clusters: Sequence[str] | None = None,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> Figure:
    """Share of items *a* gets right minus the share *b* gets right, on the same items (the
    extractor against the baseline). With *clusters* (the body of each item) the interval is a
    bootstrap over the bodies, because the company texts of one body are not independent;
    without it, over the items."""
    if len(a) != len(b) or not a:
        raise ValueError("two non-empty lists of the same length are needed")
    if clusters is not None and len(clusters) != len(a):
        raise ValueError("one cluster per item is needed")
    diffs = [int(x) - int(y) for x, y in zip(a, b, strict=True)]
    names = list(clusters) if clusters is not None else [str(i) for i in range(len(diffs))]
    grouped: dict[str, list[int]] = {}
    for name, d in zip(names, diffs, strict=True):
        grouped.setdefault(name, []).append(d)
    return _cluster_ratio([(sum(v), len(v)) for v in grouped.values()], seed, resamples)


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


def require_unique(keys: Sequence[str], what: str) -> None:
    """Stop when a key repeats: a repeated key would give two rows one weight and one mark."""
    repeated = sorted(k for k, n in Counter(keys).items() if n > 1)
    if repeated:
        raise MarkedFileError(f"{what}: repeated keys {repeated[:5]}")


def require_same_keys(found: Sequence[str], expected: Sequence[str], what: str) -> None:
    """Stop when the owner's file does not hold exactly the rows of the draw. A deleted row would
    silently change the weights (``population / rows tagged``) or drop a text from a share."""
    missing = sorted(set(expected) - set(found))
    if missing:
        raise MarkedFileError(
            f"{what}: {len(missing)} rows of the draw are not in the tagged file "
            f"(first: {missing[:5]})"
        )


def year_of(row: Mapping[str, str]) -> int | None:
    """The publication year of a row (``pub_date`` as ``YYYY-...``), ``None`` when it has none."""
    head = (row.get("pub_date") or "").strip()[:4]
    return int(head) if head.isdigit() else None


def rows_without_year(rows: Sequence[Mapping[str, str]]) -> int:
    """Rows the ``2022+`` figures cannot place because their ``pub_date`` is empty or malformed."""
    return sum(year_of(r) is None for r in rows)


def label_check_report(rows: Sequence[Mapping[str, str]]) -> dict[str, Figure]:
    """Precision of the automatic labels per kind (``lead`` / ``continuation``) from the marked
    ``correct`` column (yes / no), with Wilson intervals; ``<kind> 2022+`` repeats it for the rows
    published from 2022 on (``pub_date``), the window the backtest cut serves. A file without the
    ``pub_date`` column is refused: its ``2022+`` figures would silently be ``0 of 0``."""
    if rows and "pub_date" not in rows[0]:
        raise MarkedFileError("the label-check rows lack the column 'pub_date'")
    out: dict[str, Figure] = {}
    for kind in ("lead", "continuation"):
        part = [r for r in rows if r["kind"] == kind]
        out[kind] = wilson(sum(r["correct"] == "yes" for r in part), len(part))
        late = [r for r in part if (year_of(r) or 0) >= LATE_FROM_YEAR]
        out[f"{kind} 2022+"] = wilson(sum(r["correct"] == "yes" for r in late), len(late))
    return out


def train_on_lead_only(report: Mapping[str, Figure]) -> bool:
    """D27's rule, fixed before the owner read the sample: the `continuation` labels are dropped
    from training when at most :data:`CONTINUATION_BAR` of them are right. The rule is on the point
    estimate, as the owner fixed it; the interval is printed beside it, not used. With no
    `continuation` row marked there is no estimate and no rule: that raises, it never defaults."""
    if math.isnan(report["continuation"].point):
        raise MarkedFileError(
            "no `continuation` row in the label check: the D27 rule has no figure"
        )
    return report["continuation"].point <= CONTINUATION_BAR


def gain_verdict(gain: Figure) -> str:
    """D32: the extractor is ``better`` than the baseline when the whole interval of its gain is
    above 0, ``not better`` when the whole interval is below 0, else ``not decided``. The edges
    are those of :meth:`Figure.judge`: an interval that only touches 0 decides nothing."""
    if gain.low > 0:
        return "better"
    if gain.high < 0:
        return "not better"
    return "not decided"


def body_of_key(key: str) -> str:
    """The body (article) of a faithfulness text key ``<article>#<ticker>#<n>``."""
    return key.split("#", 1)[0]


def faithfulness_report(
    marks: Mapping[str, bool],
    extractors_of: Mapping[str, Sequence[str]],
    *,
    cluster_of: Callable[[str], str] = body_of_key,
    omitted: Mapping[str, Sequence[str]] | None = None,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> dict[str, Figure]:
    """Share of faithful texts per extractor. *marks* is the owner's yes / no by text key and
    *extractors_of* says which extractors produced each text (one text can be both's). The point
    is the share of texts; the interval is a bootstrap over bodies (*cluster_of*), because the
    company texts of one body are not independent (D32). *omitted* maps an extractor to the keys
    of the companies it named but wrote no text for; each counts as an unfaithful text, so an
    extractor cannot raise its share by leaving the hard companies out."""
    omitted = omitted or {}
    names = sorted({e for v in extractors_of.values() for e in v} | set(omitted))
    out: dict[str, Figure] = {}
    for name in names:
        grouped: dict[str, list[int]] = {}
        for key, owners in extractors_of.items():
            if name in owners:
                grouped.setdefault(cluster_of(key), []).append(int(marks[key]))
        for key in omitted.get(name, ()):
            grouped.setdefault(cluster_of(key), []).append(0)
        out[name] = _cluster_ratio([(sum(v), len(v)) for v in grouped.values()], seed, resamples)
    return out
