"""The report of a reading of the validation slice (Work item 18, T-148c3): what a run, or the pinned model,
did with the owner's 83 validation texts. Kept apart from ``sentiment_pair_train`` on purpose: the revision
of that module is part of the grid's fingerprint (a number depends on it), and a change to how a number is
*shown* must not invalidate the runs.

Two kinds of figure, and the difference matters. The validation slice was drawn **per class the pinned model
predicted** (30 each, from pools of different sizes), so its raw shares are shares of that draw, not of the
population: the confusion matrix, the per-class figures and McNemar are *raw descriptions of the 83*. The
**weighted** figures (``gold_metrics.stratified_report``, ``paired_stratified_gain``: each row weighs
``pool size / drawn`` of its stratum) are the ones the bar and the pin rule of D35 are read in, with the
interval and the verdict of the gold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from news_nlp import gold_metrics
from sentiment_pair_train import LABEL2ID

METRICS = {"agreement": "agreement", "negative precision": "precision", "negative recall": "recall"}
#: D32's bar for the cut's sentiment (not the pin rule: that one is a paired gain, ``weighted_gains``).
BARS = {"agreement": 0.80, "negative precision": 0.70, "negative recall": 0.70}


# --- raw, of the 83 -------------------------------------------------------------------------------


def confusion(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, dict[str, int]]:
    """``[owner's class][predicted class]`` -> texts, over the three classes."""
    table = {e: dict.fromkeys(LABEL2ID, 0) for e in LABEL2ID}
    for e, p in zip(expected, predicted, strict=True):
        table[e][p] += 1
    return table


def class_figures(table: dict[str, dict[str, int]]) -> dict[str, dict[str, gold_metrics.Figure]]:
    """Raw precision and recall of each class with a Wilson interval (as drawn, not re-weighted)."""
    figures = {}
    for label in LABEL2ID:
        tp = table[label][label]
        predicted = sum(row[label] for row in table.values())
        figures[label] = {
            "precision": gold_metrics.wilson(tp, predicted),
            "recall": gold_metrics.wilson(tp, sum(table[label].values())),
        }
    return figures


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """The exact two-sided McNemar p-value: of the texts the two readings do not share, how likely is a split
    as uneven as *only_a* against *only_b* if each reading were as likely to be the right one."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


class PairedReport(NamedTuple):
    both: int
    only_a: int
    only_b: int
    neither: int
    p_value: float
    gain: (
        gold_metrics.Figure
    )  # raw: share *a* gets right minus share *b* gets right, over the items


def paired_report(a_hits: Sequence[bool], b_hits: Sequence[bool]) -> PairedReport:
    """Two readings of the same texts: who is right where, the exact McNemar p-value, and the raw gain of *a*
    over *b* with its interval. **No verdict:** the pin rule is the weighted paired gain, below."""
    pairs = list(zip(a_hits, b_hits, strict=True))
    return PairedReport(
        both=sum(x and y for x, y in pairs),
        only_a=sum(x and not y for x, y in pairs),
        only_b=sum(y and not x for x, y in pairs),
        neither=sum(not x and not y for x, y in pairs),
        p_value=mcnemar_exact(
            sum(x and not y for x, y in pairs), sum(y and not x for x, y in pairs)
        ),
        gain=gold_metrics.paired_difference(a_hits, b_hits),
    )


def report_lines(name: str, expected: Sequence[str], predicted: Sequence[str]) -> list[str]:
    """The confusion matrix and the raw per-class figures of one reading, for printing."""
    table = confusion(expected, predicted)
    hits = sum(e == p for e, p in zip(expected, predicted, strict=True))
    lines = [
        f"{name}, RAW on the {len(expected)} texts: agreement {hits}/{len(expected)}",
        f"  rows: the owner's class; columns: {list(LABEL2ID)}",
    ]
    lines += [f"  {e:<9}" + " ".join(f"{table[e][p]:>4}" for p in LABEL2ID) for e in LABEL2ID]
    for label, f in class_figures(table).items():
        lines.append(f"  {label}: precision {f['precision']}, recall {f['recall']}")
    return lines


# --- weighted: the figures the bar and the pin rule are read in ------------------------------------


def weighted_figures(
    rows: Sequence[Mapping[str, str]], populations: Mapping[str, int], column: str
) -> dict[str, gold_metrics.Figure]:
    """Agreement and negative precision and recall of the reading in *column*, weighted by stratum (rows carry
    ``stratum``, ``gold`` and the column)."""
    report = gold_metrics.stratified_report(
        [{**r, "predicted": r[column]} for r in rows],
        populations,
        label="negative",
        stratum_key="stratum",
    )
    return {
        "agreement": report.agreement,
        "negative precision": report.precision,
        "negative recall": report.recall,
    }


def weighted_gains(
    rows: Sequence[Mapping[str, str]],
    populations: Mapping[str, int],
    candidate: str,
    baseline: str,
) -> dict[str, tuple[gold_metrics.Figure, str]]:
    """The paired weighted gain of *candidate* over *baseline* for agreement and negative precision and
    recall, each with the gold's verdict (``gain_verdict``): the figures of D35's pin rule."""
    gains = {}
    for name, metric in METRICS.items():
        gain = gold_metrics.paired_stratified_gain(
            rows,
            populations,
            candidate=candidate,
            baseline=baseline,
            metric=metric,
            label="negative",
            stratum_key="stratum",
        )
        gains[name] = (gain, gold_metrics.gain_verdict(gain))
    return gains


def bar_verdicts(figures: Mapping[str, gold_metrics.Figure]) -> dict[str, str]:
    """D32's verdict on each weighted figure (``Figure.judge``: ``pass`` when the whole interval is at or above
    the bar, ``fail`` when it is wholly below, else ``not decided``). It is a different test from the pin
    rule: a model can be better than the pinned one (the pin rule) and still below the bar (D32)."""
    return {name: figures[name].judge(bar) for name, bar in BARS.items()}


def weighted_agreement(
    rows: Sequence[Mapping[str, str]], populations: Mapping[str, int], column: str
) -> float:
    """The point estimate of the weighted agreement of the reading in *column* (no interval: for comparing
    the runs of the grid)."""
    strata: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        strata.setdefault(row["stratum"], []).append(row)
    total = sum(populations[k] for k in strata)
    right = sum(
        populations[k] * sum(r["gold"] == r[column] for r in v) / len(v) for k, v in strata.items()
    )
    return right / total if total else math.nan
