"""What T-121(b) reports from the judged calibration sample (PLAN.md Work item 15 step 4).

Read-only: it takes the weighted truth labels of ``calibration_judge.load_truths`` and the
stored scores of the confirmed articles and returns

* per candidate loosening rule, the share of the articles it would take out of `other` whose
  truth is `other` (Horvitz-Thompson weights, stratified bootstrap), next to the cap T-123 puts
  on it (:func:`news_nlp.eval.calibration.loosening_bound`), with the two-sided 95% interval and
  the one-sided 99.17% bounds the staged design stops on (:func:`report_rule`);
* how well the maximum score, the margin over the runner-up and the group score separate
  truly-`other` from truly-slug articles (a weighted AUC with a bootstrap interval,
  :func:`separation`).

A rule's removed set is a union of whole sub-cells of the sample
(``calibration_sample.SUBCELLS``), so the estimator is a plain stratified one. Weights are pooled
over the fit and report halves: the candidate rules are fixed in advance, nothing is fitted
here (a threshold fitted on a half must use :class:`Judged.half`).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from news_nlp.eval.calibration import (
    ConfirmedArticle,
    Rule,
    _share_interval,
    _ShareRow,
    _top_two,
    apply_rule,
    band_of,
    loosening_bound,
)
from news_nlp.eval.calibration_judge import TruthSet
from news_nlp.eval.calibration_sample import stratum_of
from news_nlp.taxonomy import OTHER_LABEL

#: Measured on eval_run 43's `accept` partition (T-121(a)): the `other` precision and recall of
#: the shipped rule that T-123's co-requirement is expressed against.
BASELINE_OTHER_PRECISION = 0.319
BASELINE_OTHER_RECALL = 0.661
#: Two-sided level whose tails are the one-sided stop bounds of the staged design: three candidate
#: rules looked at twice (stage 1, then stage 2) make 6 looks, so each bound is one-sided at
#: 1 - 0.05 / 6 = 99.17% (the design first fixed 98.75% for two rules; a third, `t0.55+m0.00`,
#: sits in the same stage-1 sub-cells). The final decision uses 95%.
STOP_LEVEL = 1 - 0.10 / 6
#: Bootstrap repetitions. The stop bound sits 0.83% from the end of the sorted draws, so 1,000
#: repetitions would put it on the 8th value: noisy in the third digit. 5,000 puts it on the 42nd.
BOOTSTRAP_REPS = 5000
FINAL_LEVEL = 0.95
#: Score bands where a threshold or a margin moves labels (0.4-0.5 and 0.5-0.6).
THRESHOLD_BANDS = (1, 2)


class Judged(NamedTuple):
    article: ConfirmedArticle
    cell: str
    half: str
    weight: float  # pooled over both halves: cell population / usable articles of the cell
    truth: str  # the judge's best-fitting label


def judged_articles(truths: TruthSet, articles: Sequence[ConfirmedArticle]) -> list[Judged]:
    """The usable judged articles with their pooled weights. An id missing from *articles*
    raises ``KeyError`` (the sample was drawn from this population)."""
    by_id = {a.article_id: a for a in articles}
    per_cell = Counter(t.cell for t in truths.rows)
    return [
        Judged(
            by_id[t.article_id],
            t.cell,
            t.half,
            truths.populations[t.cell] / per_cell[t.cell],
            t.truth_slug,
        )
        for t in truths.rows
    ]


def moved(article: ConfirmedArticle, rule: Rule) -> bool:
    """True if *rule* takes the article out of `other` (stored `other`, a slug under *rule*)."""
    return (
        article.stored_label == OTHER_LABEL
        and apply_rule(article.group_score, article.leaf_scores, rule) != OTHER_LABEL
    )


class RuleReport(NamedTuple):
    rule: Rule
    removed: int  # articles in the population the rule takes out of `other`
    n: int  # judged articles inside that set
    n_other: int  # of which truly `other`
    rate: float  # weighted share truly `other`
    low: float  # two-sided 95% interval
    high: float
    stop_low: float  # one-sided 99.17% lower bound
    stop_high: float  # one-sided 99.17% upper bound
    cap: float  # the most T-123 allows to be truly `other` (point-estimate condition)
    design_effect: float | None
    verdict: str


def _kish(weights: Sequence[float]) -> float | None:
    if not weights:
        return None
    return len(weights) * sum(w * w for w in weights) / sum(weights) ** 2


def report_rule(
    judged: Sequence[Judged],
    articles: Sequence[ConfirmedArticle],
    rule: Rule,
    *,
    precision: float = BASELINE_OTHER_PRECISION,
    recall: float = BASELINE_OTHER_RECALL,
    final: bool = False,
    reps: int = BOOTSTRAP_REPS,
    seed: int = 0,
) -> RuleReport | None:
    """The truly-`other` rate of the set *rule* removes from `other`, against its cap.

    Verdicts: ``fails`` if the lower bound is above the cap (one-sided 99.17% while staged, the
    95% interval when *final*); ``below the cap`` if the upper bound is below it (a screen, not
    a pass: T-123 also needs the lower bound of the recall change within tolerance);
    ``undecided`` otherwise; ``not covered`` if some cell holding part of the removed set has no
    judged article yet (the estimate would ignore it). ``None`` if the rule removes nothing."""
    labelled_other = sum(a.stored_label == OTHER_LABEL for a in articles)
    removed_cells = {stratum_of(a.group_score, a.leaf_scores) for a in articles if moved(a, rule)}
    removed = sum(moved(a, rule) for a in articles)
    if not removed:
        return None
    if removed_cells - {j.cell for j in judged}:
        return RuleReport(
            rule,
            removed,
            0,
            0,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            None,
            "not covered",
        )
    rows: list[_ShareRow] = [
        (j.cell, j.weight, moved(j.article, rule), j.truth == OTHER_LABEL) for j in judged
    ]
    inside = [r for r in rows if r[2]]
    if not inside:
        return None
    seen = sum(r[1] for r in inside)
    rate = sum(r[1] for r in inside if r[3]) / seen
    low, high = _share_interval(rows, reps=reps, seed=seed, level=FINAL_LEVEL)
    stop_low, stop_high = _share_interval(rows, reps=reps, seed=seed, level=STOP_LEVEL)
    cap = loosening_bound(labelled_other, removed, precision, recall).max_rate
    lower, upper = (low, high) if final else (stop_low, stop_high)
    if lower > cap:
        verdict = "fails"
    elif upper < cap:
        verdict = "below the cap"
    else:
        verdict = "undecided"
    return RuleReport(
        rule,
        removed,
        len(inside),
        sum(r[3] for r in inside),
        rate,
        low,
        high,
        stop_low,
        stop_high,
        cap,
        _kish([r[1] for r in inside]),
        verdict,
    )


# --- separation -------------------------------------------------------------


def feature_values(article: ConfirmedArticle) -> dict[str, float]:
    """The three scores a rule can threshold: the maximum leaf score, its margin over the
    runner-up, and the group score."""
    _, best, runner_up = _top_two(article.leaf_scores)
    return {"max_score": best, "margin": best - runner_up, "group_score": article.group_score}


def weighted_auc(
    values: Sequence[float], positive: Sequence[bool], weights: Sequence[float]
) -> float:
    """P(value of a positive > value of a negative), ties counting half, by weight. ``nan`` if
    either class has no weight."""
    v = np.asarray(values, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    w = np.asarray(weights, dtype=float)
    pos_w, neg_w = w * pos, w * ~pos
    if pos_w.sum() <= 0 or neg_w.sum() <= 0:
        return math.nan
    uniq, inverse = np.unique(v, return_inverse=True)
    gp = np.bincount(inverse, weights=pos_w, minlength=len(uniq))
    gn = np.bincount(inverse, weights=neg_w, minlength=len(uniq))
    below = np.cumsum(gn) - gn
    return float((gp * (below + 0.5 * gn)).sum() / (pos_w.sum() * neg_w.sum()))


class Separation(NamedTuple):
    feature: str
    auc: float  # P(a truly-slug article scores higher than a truly-`other` one)
    low: float  # 95% stratified-bootstrap interval
    high: float
    n: int
    n_other: int


def separation(
    judged: Sequence[Judged],
    *,
    bands: Sequence[int] = THRESHOLD_BANDS,
    reps: int = 1000,
    seed: int = 0,
) -> list[Separation]:
    """Among the judged articles the shipped rule labels `other` in the given score *bands*, how
    well each feature separates truly-slug from truly-`other` articles (0.5 = not at all)."""
    pool = [
        j
        for j in judged
        if j.article.stored_label == OTHER_LABEL
        and band_of(max(j.article.leaf_scores)) in bands
        and not j.cell.startswith("short_circuit")
    ]
    by_cell: dict[str, list[Judged]] = {}
    for j in pool:
        by_cell.setdefault(j.cell, []).append(j)
    rng = random.Random(seed)  # noqa: S311 -- resampling, not cryptography
    out: list[Separation] = []
    for name in ("max_score", "margin", "group_score"):

        def auc_of(sample: Sequence[Judged], feature: str = name) -> float:
            return weighted_auc(
                [feature_values(j.article)[feature] for j in sample],
                [j.truth != OTHER_LABEL for j in sample],
                [j.weight for j in sample],
            )

        draws = sorted(
            a
            for a in (
                auc_of([x for g in by_cell.values() for x in rng.choices(g, k=len(g))])
                for _ in range(reps)
            )
            if not math.isnan(a)
        )
        low = draws[int(0.025 * (len(draws) - 1))] if draws else math.nan
        high = draws[round(0.975 * (len(draws) - 1))] if draws else math.nan
        out.append(
            Separation(
                name, auc_of(pool), low, high, len(pool), sum(j.truth == OTHER_LABEL for j in pool)
            )
        )
    return out
