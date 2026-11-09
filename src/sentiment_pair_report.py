"""The report of a reading of the validation slice (Work item 18, T-148c3): what a run, or the pinned model,
did with the owner's 83 validation texts. Kept apart from ``sentiment_pair_train`` on purpose: the revision
of that module is part of the grid's fingerprint (a number depends on it), and a change to how a number is
*shown* must not invalidate the runs.

**The metrics (owner, 2026-11-09; PLAN D37): Precision, Recall and ROC OVR, per class, are what is recorded
and what model selection uses.** For each of the three classes (positive, negative, neutral): the precision
and the recall of the class, and the ROC AUC of the class against the rest (one-vs-rest, from the model's
probability of the class: the mean over a text's chunks). **The selection score** is the mean of those nine
figures (the macro of each of the three metrics, then their mean): one number, with no weight between the
metrics or the classes. A figure that is undefined (a class never predicted has no precision) counts as 0
there (D34 (a): undefined is not passed). Accuracy (agreement) is not a selection, bar or pin metric any
more: it is shown only as a description (the confusion matrix).

Two kinds of figure, and the difference matters. The validation slice was drawn **per class the pinned model
predicted** (30 each, from pools of different sizes), so its raw shares are shares of that draw, not of the
population: the confusion matrix, the raw per-class figures and McNemar are *raw descriptions of the 83*. The
**weighted** figures (each text weighs ``pool size / drawn`` of its stratum; intervals are a seeded bootstrap
inside the strata) are the ones the bar and the pin rule are read in.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from news_nlp import gold_metrics
from sentiment_pair_train import LABEL2ID

CLASSES = tuple(LABEL2ID)  # positive, negative, neutral
CLASS_METRICS = ("precision", "recall", "roc_ovr")
#: D32's bar for the cut's sentiment on the negative class (the class the backtest counts). ROC OVR is recorded
#: and used in the selection and the pin rule, but has no bar (the owner has given none).
BARS = {"negative precision": 0.70, "negative recall": 0.70}
#: What the pin rule reads (``pin_decision``): the selection score's gain, and the negative class's three.
PIN_STATISTICS = ("score", "negative precision", "negative recall", "negative roc_ovr")
RESAMPLES = gold_metrics.RESAMPLES
SEED = 20261109


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


# --- weighted precision, recall and ROC OVR -------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    """One validation text as a reading scored it: the owner's class, the stratum it was drawn from (the class
    the pinned model predicted then) and the reading's label and class probabilities."""

    key: str
    stratum: str
    gold: str
    predicted: str
    probs: Mapping[str, float]


def row_weights(scored: Sequence[Scored], populations: Mapping[str, int]) -> list[float]:
    """The weight of each text: its stratum's pool size over the number of its texts in the slice."""
    drawn: dict[str, int] = {}
    for s in scored:
        drawn[s.stratum] = drawn.get(s.stratum, 0) + 1
    unknown = sorted(set(drawn) - set(populations))
    if unknown:
        raise gold_metrics.MarkedFileError(f"no population for the strata {unknown}")
    return [populations[s.stratum] / drawn[s.stratum] for s in scored]


def weighted_roc_auc(
    scores: Sequence[float], positive: Sequence[bool], weights: Sequence[float]
) -> float:
    """The weighted ROC AUC of one class against the rest: the chance that a (weighted) positive text scores
    higher than a negative one, ties counting half. NaN if there is no positive or no negative text."""
    pos = [(sc, w) for sc, p, w in zip(scores, positive, weights, strict=True) if p]
    neg = [(sc, w) for sc, p, w in zip(scores, positive, weights, strict=True) if not p]
    total = sum(wp * wn for _, wp in pos for _, wn in neg)
    if not total:
        return math.nan
    won = sum(
        wp * wn * (1.0 if sp > sn else 0.5 if sp == sn else 0.0) for sp, wp in pos for sn, wn in neg
    )
    return won / total


def class_metrics(
    scored: Sequence[Scored], weights: Sequence[float]
) -> dict[str, dict[str, float]]:
    """Precision, recall and ROC OVR of each class, weighted; ``nan`` where undefined (a class never predicted
    has no precision; a class with no text of its own, or all texts, has no recall or ROC)."""
    out: dict[str, dict[str, float]] = {}
    for c in CLASSES:
        hit = sum(
            w for s, w in zip(scored, weights, strict=True) if s.predicted == c and s.gold == c
        )
        predicted = sum(w for s, w in zip(scored, weights, strict=True) if s.predicted == c)
        actual = sum(w for s, w in zip(scored, weights, strict=True) if s.gold == c)
        out[c] = {
            "precision": hit / predicted if predicted else math.nan,
            "recall": hit / actual if actual else math.nan,
            "roc_ovr": weighted_roc_auc(
                [s.probs[c] for s in scored], [s.gold == c for s in scored], weights
            ),
        }
    return out


def _zero_if_undefined(x: float) -> float:
    return 0.0 if math.isnan(x) else x


def macro(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """The mean over the three classes of each metric (an undefined figure counts as 0)."""
    return {
        m: sum(_zero_if_undefined(metrics[c][m]) for c in CLASSES) / len(CLASSES)
        for m in CLASS_METRICS
    }


def selection_score(metrics: Mapping[str, Mapping[str, float]]) -> float:
    """The selection score: the mean of the nine figures (equal weights; an undefined figure counts as 0)."""
    return sum(macro(metrics).values()) / len(CLASS_METRICS)


def statistics_of(scored: Sequence[Scored], weights: Sequence[float]) -> dict[str, float]:
    """Every figure the report reads, by name: ``score``, the three macros (``macro precision``, ...) and each
    class's three (``negative precision``, ``positive roc_ovr``, ...)."""
    metrics = class_metrics(scored, weights)
    out = {"score": selection_score(metrics)}
    out.update({f"macro {m}": v for m, v in macro(metrics).items()})
    out.update({f"{c} {m}": metrics[c][m] for c in CLASSES for m in CLASS_METRICS})
    return out


def detail_of(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float | None]]:
    """What a run records (JSON-able, ``None`` for an undefined figure): each class's three figures and the macro."""
    rows: dict[str, dict[str, float | None]] = {
        c: {m: None if math.isnan(v) else v for m, v in metrics[c].items()} for c in CLASSES
    }
    rows["macro"] = dict(macro(metrics))
    return rows


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _resampled(strata: Mapping[str, list[int]], rng: random.Random) -> list[int]:
    """Indices drawn with replacement inside each stratum (the strata keep their sizes)."""
    out = []
    for members in strata.values():
        out += [rng.choice(members) for _ in members]
    return out


def _figures(
    point: Mapping[str, float], draws: Mapping[str, list[float]]
) -> dict[str, gold_metrics.Figure]:
    figures = {}
    for name, value in point.items():
        kept = [d for d in draws[name] if not math.isnan(d)]
        if math.isnan(value) or not kept:
            figures[name] = gold_metrics.Figure(value, math.nan, math.nan)
        else:
            figures[name] = gold_metrics.Figure(
                value, _percentile(kept, 0.025), _percentile(kept, 0.975)
            )
    return figures


def weighted_figures(
    scored: Sequence[Scored],
    populations: Mapping[str, int],
    *,
    seed: int = SEED,
    resamples: int = RESAMPLES,
) -> dict[str, gold_metrics.Figure]:
    """Each figure of :func:`statistics_of` with a 95% interval: a seeded bootstrap inside the strata."""
    point = statistics_of(scored, row_weights(scored, populations))
    strata: dict[str, list[int]] = {}
    for i, s in enumerate(scored):
        strata.setdefault(s.stratum, []).append(i)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(resamples):
        sample = [scored[i] for i in _resampled(strata, rng)]
        for name, value in statistics_of(sample, row_weights(sample, populations)).items():
            draws[name].append(value)
    return _figures(point, draws)


def weighted_gains(
    candidate: Sequence[Scored],
    baseline: Sequence[Scored],
    populations: Mapping[str, int],
    *,
    seed: int = SEED,
    resamples: int = RESAMPLES,
) -> dict[str, tuple[gold_metrics.Figure, str]]:
    """The paired gain of the candidate over the baseline for each figure of :func:`statistics_of` (the same
    resample feeds both readings, so the interval is of the difference), each with the gold's verdict
    (``gain_verdict``): ``better`` only when the whole interval is above 0. The texts must be the same ones."""
    if [s.key for s in candidate] != [s.key for s in baseline]:
        raise gold_metrics.MarkedFileError("the two readings are not of the same texts")
    weights = row_weights(candidate, populations)
    point_a, point_b = statistics_of(candidate, weights), statistics_of(baseline, weights)
    point = {name: point_a[name] - point_b[name] for name in point_a}
    strata: dict[str, list[int]] = {}
    for i, s in enumerate(candidate):
        strata.setdefault(s.stratum, []).append(i)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(resamples):
        idx = _resampled(strata, rng)
        a, b = [candidate[i] for i in idx], [baseline[i] for i in idx]
        w = row_weights(a, populations)
        sa, sb = statistics_of(a, w), statistics_of(b, w)
        for name in point:
            draws[name].append(sa[name] - sb[name])
    return {
        name: (figure, gold_metrics.gain_verdict(figure))
        for name, figure in _figures(point, draws).items()
    }


def bar_verdicts(figures: Mapping[str, gold_metrics.Figure]) -> dict[str, str]:
    """D32's verdict on each barred figure (the negative class's precision and recall; ``Figure.judge``:
    ``pass`` when the whole interval is at or above the bar, ``fail`` when it is wholly below, else ``not
    decided``). ROC OVR has no bar. It is a different test from the pin rule: a model can be better than the
    pinned one (the pin rule) and still below the bar (D32)."""
    return {name: figures[name].judge(bar) for name, bar in BARS.items()}


def scored_from_rows(
    predictions: Sequence[Mapping[str, str]], marks: Sequence[Mapping[str, str]]
) -> tuple[list[Scored], dict[str, int]]:
    """The kept predictions of a reading (``key``, ``expected``, ``predicted``, ``p_<class>``) as :class:`Scored`
    with their strata, and the pool size of each stratum, from the owner's marks (``predicted`` and
    ``pool_class_size`` of the validation rows)."""
    by_key = {m["key"]: m for m in marks if m["slice"] == "validation"}
    populations: dict[str, int] = {}
    scored = []
    for p in predictions:
        m = by_key[p["key"]]
        size = populations.setdefault(m["predicted"], int(m["pool_class_size"]))
        if size != int(m["pool_class_size"]):
            raise gold_metrics.MarkedFileError(f"stratum {m['predicted']} has two pool sizes")
        scored.append(
            Scored(
                p["key"],
                m["predicted"],
                p["expected"],
                p["predicted"],
                {c: float(p[f"p_{c}"]) for c in CLASSES},
            )
        )
    return scored, populations


def pin_decision(verdicts: Mapping[str, str]) -> tuple[bool, str]:
    """D35's pin rule on the verdicts of the paired weighted gain (``weighted_gains``; keys ``score``,
    ``negative precision``, ``negative recall``, ``negative roc_ovr``), and the reason in words. (3) The
    candidate is pinned only if the selection score's gain is ``better``. (4) No pin if the negative class's
    precision, recall or ROC OVR gain is ``not better``. An ``undefined`` gain (no interval) of one of the three
    is **read as not passing** (the owner's answer of 2026-11-09, PLAN D35 (7)). This is D35's rule with the
    metrics of D37 in place of the agreement (the owner's 2026-11-09 instruction); how exactly the rule reads
    them is written in PLAN D37 for the owner to confirm. The rule is applied once, to the gold (T-148c4); on
    the validation slice it is a rehearsal."""
    if verdicts["score"] != "better":
        return False, f"the selection score gain is `{verdicts['score']}`, not `better` (D35 (3))"
    for name in ("negative precision", "negative recall", "negative roc_ovr"):
        if verdicts[name] == "not better":
            return False, f"the {name} gain is `not better` (D35 (4))"
        if verdicts[name] == "undefined":
            return (
                False,
                f"the {name} gain is `undefined`, read as not passing (owner, 2026-11-09)",
            )
    return True, (
        f"the selection score gain is `better` and none of the negative class's gains is `not better` "
        f"(precision `{verdicts['negative precision']}`, recall `{verdicts['negative recall']}`, "
        f"ROC OVR `{verdicts['negative roc_ovr']}`) (D35 (3), (4))"
    )
