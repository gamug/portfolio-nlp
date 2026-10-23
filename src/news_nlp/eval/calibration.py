"""Offline simulation of a category decision rule against stored scores (PLAN.md Work
item 15 step 4, TASKS T-121(a)). No model, no judge call, nothing written.

Three things, all from data already in the stores:

* :func:`relabel_table`: for each candidate rule, how many confirmed articles it
  re-labels, from the stored leaf scores alone, in total and per (slug x band) cell.
* :func:`verdict_table`: for the rules that only *tighten* the shipped one (they send
  items to ``other``, never away from it), the stored judgements of one eval run score
  the candidate exactly: the judge's ``ideal_slug`` is the truth label, so a rewritten
  prediction is graded by :func:`news_nlp.eval.metrics.aggregate_category` unchanged.
  A rule that moves items *away* from ``other`` cannot be scored this way (the judge
  graded ``other`` on those items, not the slug), which is why it needs the paid sample.
* :func:`simulate_power`: the T-123 estimator itself, the paired difference of the
  weighted ``other`` precision (a ratio of weighted sums, because the candidate changes
  the denominator), on report-half samples of the planned size drawn from the real
  confirmed articles, with a stated truth model and a within-cell bootstrap interval.

The truth model has to be assumed where nothing was measured: in the >= 0.6 cells the
stored label is the argmax slug and eval_run 43 measured how often it was right; in the
0.4-0.6 cells and the short-circuit stratum the stored label is ``other`` and the judge
graded ``other``, so the share of articles whose truth is the argmax slug is a swept
parameter, not a number: the probability that an article in the 0.5-0.6 band is truly
``other`` (:class:`TruthModel` ``low_other``), constrained so that the model reproduces the
measured `other` precision of the stored run.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from news_nlp.eval.metrics import DIAGNOSTIC_ONLY_BUCKETS, aggregate_category
from news_nlp.eval.partition import Pair, is_defined
from news_nlp.eval.sampling import EvalItem
from news_nlp.taxonomy import (
    CATEGORY_CONFIDENCE_THRESHOLD,
    CATEGORY_GROUP_FLOOR,
    CATEGORY_SLUGS,
    OTHER_LABEL,
)

#: Upper edges of the score bands; the last band is >= 0.8 (PLAN step 4).
BAND_EDGES = (0.4, 0.5, 0.6, 0.8)
BAND_NAMES = ("<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.8", ">=0.8")
SHORT_CIRCUIT = "short_circuit"
#: Report-half sample per cell (PLAN step 4: 90 per cell in the 0.4-0.6 bands, 60
#: elsewhere, split in halves; the short-circuit stratum 30 report).
REPORT_HALF_THRESHOLD_BANDS = 45
REPORT_HALF_OTHER_BANDS = 30
REPORT_HALF_SHORT_CIRCUIT = 30
#: The two bands where a threshold moves labels.
THRESHOLD_BANDS = (1, 2)
#: Band key of the short-circuit stratum among the low cells (it has no score band).
SHORT_CIRCUIT_BAND = -1
#: How the probability that an article is truly `other` varies over the cells whose stored
#: label is `other`. Whatever the shape, the average over the articles labelled `other` is
#: fixed at the stored run's measured `other` precision (0.319 on the `accept` partition): the
#: stored judgements already say that about two thirds of them have a real category in the
#: judge's view, spread over any of the nine slugs, and nothing a shape does may move that.
#:
#: * ``band2``: only the 0.5-0.6 band differs (the swept rate); every other low cell shares
#:   the rate that keeps the average at the measured value;
#: * ``graded``: the 0.5-0.6 band is the swept rate, 0.4-0.5 is halfway between it and the
#:   rate of the remaining cells (< 0.4 and the short-circuit stratum);
#: * ``flat``: every low cell has the measured rate (no cell is cleaner than another, so no
#:   rule can gain: the zero-gain check of the estimator).
SHAPES = ("graded", "band2", "flat")
DEFAULT_SHAPE = "graded"
#: The first band at or above the shipped threshold (CATEGORY_CONFIDENCE_THRESHOLD = 0.6).
FIRST_CONFIDENT_BAND = 3


class Rule(NamedTuple):
    """argmax slug if its score is >= ``threshold`` and leads the runner-up by >=
    ``margin``, else ``other``; the level-1 floor is unchanged."""

    threshold: float = CATEGORY_CONFIDENCE_THRESHOLD
    margin: float = 0.0

    @property
    def name(self) -> str:
        return f"t{self.threshold:.2f}+m{self.margin:.2f}"

    @property
    def tightens(self) -> bool:
        """True when every label this rule gives is the shipped rule's label or ``other``
        (a threshold at or above the shipped one). A margin can only remove labels, so it
        never decides this: a rule with a lower threshold loosens whatever its margin."""
        return self.threshold >= CATEGORY_CONFIDENCE_THRESHOLD


BASELINE = Rule()


class ConfirmedArticle(NamedTuple):
    article_id: int
    stored_label: str
    group_score: float
    leaf_scores: tuple[float, ...]  # in CATEGORY_SLUGS order


def band_of(score: float) -> int:
    return sum(score >= edge for edge in BAND_EDGES)


def _top_two(leaf_scores: Sequence[float]) -> tuple[int, float, float]:
    best = max(leaf_scores)
    index = list(leaf_scores).index(best)  # first of an exact tie, as the stored rule did
    rest = [s for i, s in enumerate(leaf_scores) if i != index]
    return index, best, max(rest) if rest else 0.0


def apply_rule(group_score: float, leaf_scores: Sequence[float], rule: Rule = BASELINE) -> str:
    """The label *rule* gives from stored scores. ``BASELINE`` reproduces every stored
    label (``scripts/measure_corrected_corpus_2026_10_22.py`` item 9 checks it on the
    real store)."""
    if group_score < CATEGORY_GROUP_FLOOR:
        return OTHER_LABEL
    index, best, runner_up = _top_two(leaf_scores)
    if best >= rule.threshold and best - runner_up >= rule.margin:
        return CATEGORY_SLUGS[index]
    return OTHER_LABEL


def cell_of(group_score: float, leaf_scores: Sequence[float]) -> str:
    """``"<slug>|<band>"`` of the argmax slug and its score, or :data:`SHORT_CIRCUIT`."""
    if group_score < CATEGORY_GROUP_FLOOR:
        return SHORT_CIRCUIT
    index, best, _ = _top_two(leaf_scores)
    return f"{CATEGORY_SLUGS[index]}|{band_of(best)}"


def _band_of_cell(cell: str) -> int | None:
    return None if cell == SHORT_CIRCUIT else int(cell.rsplit("|", 1)[1])


def report_half_size(cell: str) -> int:
    band = _band_of_cell(cell)
    if band is None:
        return REPORT_HALF_SHORT_CIRCUIT
    return REPORT_HALF_THRESHOLD_BANDS if band in THRESHOLD_BANDS else REPORT_HALF_OTHER_BANDS


def candidate_rules(
    thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90),
    margins: Sequence[float] = (0.0, 0.10, 0.20, 0.30),
) -> list[Rule]:
    """The grid of uniform rules (one threshold for all slugs). Per-slug thresholds are
    fitted on the paid sample's fit half (T-121(b)); there is too little per-slug data
    here to fit nine of them."""
    return [Rule(t, m) for t in thresholds for m in margins]


# --- 1. re-label counts, from stored scores alone -------------------------------------


@dataclass(frozen=True)
class RelabelRow:
    rule: Rule
    changed: int
    to_other: int
    from_other: int
    slug_to_slug: int
    by_cell: Mapping[str, int]


def relabel_table(articles: Sequence[ConfirmedArticle], rules: Sequence[Rule]) -> list[RelabelRow]:
    """How many of *articles* each rule re-labels versus the stored label."""
    cells = [cell_of(a.group_score, a.leaf_scores) for a in articles]
    rows = []
    for rule in rules:
        by_cell: Counter[str] = Counter()
        to_other = from_other = slug_to_slug = 0
        for a, cell in zip(articles, cells, strict=True):
            new = apply_rule(a.group_score, a.leaf_scores, rule)
            if new == a.stored_label:
                continue
            by_cell[cell] += 1
            if new == OTHER_LABEL:
                to_other += 1
            elif a.stored_label == OTHER_LABEL:
                from_other += 1
            else:
                slug_to_slug += 1
        rows.append(
            RelabelRow(rule, sum(by_cell.values()), to_other, from_other, slug_to_slug, by_cell)
        )
    return rows


# --- 2. stored verdicts, tightening rules only ----------------------------------------


def _scores_of(item: EvalItem) -> list[float]:
    dist = item.prediction.get("distribution", {})
    return [float(dist.get(slug, 0.0)) for slug in CATEGORY_SLUGS]


def retag(item: EvalItem, rule: Rule) -> EvalItem:
    """*item* with its predicted label replaced by what a tightening *rule* gives. An
    item the shipped rule already labelled ``other`` stays ``other``."""
    if not rule.tightens:
        raise ValueError(
            f"{rule.name} moves labels away from 'other'; stored verdicts cannot score it"
        )
    label = str(item.prediction.get("label", OTHER_LABEL))
    if label != OTHER_LABEL:
        _, best, runner_up = _top_two(_scores_of(item))
        if not (best >= rule.threshold and best - runner_up >= rule.margin):
            label = OTHER_LABEL
    return replace(item, prediction={**item.prediction, "label": label})


def verdict_table(pairs: Sequence[Pair], rules: Sequence[Rule]) -> dict[str, dict[str, float]]:
    """``aggregate_category`` of *pairs* (already restricted to one partition with
    :func:`news_nlp.eval.partition.domain_pairs`) for each tightening rule, plus the
    unweighted mean of the nine ``precision_<slug>`` (``mean_slug_precision``)."""
    out: dict[str, dict[str, float]] = {}
    for rule in rules:
        if not rule.tightens:
            continue
        retagged = [retag(it, rule) for it, _ in pairs]
        metrics = aggregate_category(retagged, [v for _, v in pairs])
        slug_precisions = [
            metrics[f"precision_{slug}"]
            for slug in CATEGORY_SLUGS
            if f"precision_{slug}" in metrics
        ]
        metrics["mean_slug_precision"] = (
            sum(slug_precisions) / len(slug_precisions) if slug_precisions else 0.0
        )
        out[rule.name] = metrics
    return out


# --- 3. power ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthModel:
    """P(truth = argmax slug), P(truth = other), else some third label, per cell.

    * cells with a stored label equal to the argmax slug (band >= 0.6): the measured
      agreement rate of the stored run (``agree``, weighted by the Horvitz-Thompson item
      weights; a cell with too few judged items falls back to its band's pooled rate and
      is listed in ``fallback_cells``); when the judge disagreed, ``other_given_wrong`` of
      the time the truth was ``other``.
    * cells whose stored label is ``other`` (bands < 0.6, short-circuit): the truth is
      ``other`` with probability ``low_other[band]`` (by default the measured weighted
      `other` precision of the stored run, ``other_agreement``; :func:`shaped` varies it
      across bands while keeping its average at the measured value), and otherwise the
      argmax slug with probability ``slug_fraction`` of the remaining mass and some other
      slug with the rest. ``slug_fraction`` only matters to accuracy and macro F1: neither
      `other` precision nor `other` recall depends on it.
    """

    agree: Mapping[str, float]
    other_given_wrong: float
    other_agreement: float
    low_other: Mapping[int, float] = field(default_factory=dict)
    slug_fraction: float = 0.5
    fallback_cells: tuple[str, ...] = ()

    def low_rate(self, band: int | None) -> float:
        """P(truth = other) of a low cell: *band* is its score band, ``None`` for the
        short-circuit stratum."""
        key = SHORT_CIRCUIT_BAND if band is None else band
        return self.low_other.get(key, self.other_agreement)

    def probabilities(self, cell: str) -> tuple[float, float]:
        """``(P(argmax slug), P(other))`` for *cell*."""
        band = _band_of_cell(cell)
        if band is None or band < FIRST_CONFIDENT_BAND:
            p_other = self.low_rate(band)
            return self.slug_fraction * (1 - p_other), p_other
        p_slug = self.agree[cell]
        return p_slug, (1 - p_slug) * self.other_given_wrong


LOW_BANDS = (0, 1, 2, SHORT_CIRCUIT_BAND)


def low_band_counts(articles: Sequence[ConfirmedArticle]) -> dict[int, int]:
    """How many of *articles* the shipped rule labels `other`, per low band (the
    short-circuit stratum under :data:`SHORT_CIRCUIT_BAND`): the weights of the average
    that :func:`shaped` keeps at the measured `other` precision."""
    counts = dict.fromkeys(LOW_BANDS, 0)
    for a in articles:
        if apply_rule(a.group_score, a.leaf_scores) != OTHER_LABEL:
            continue
        band = _band_of_cell(cell_of(a.group_score, a.leaf_scores))
        counts[SHORT_CIRCUIT_BAND if band is None else band] += 1
    return counts


def shaped(truth: TruthModel, counts: Mapping[int, int], rate: float, shape: str) -> TruthModel:
    """*truth* with the `other` probability of the 0.5-0.6 band set to *rate* and the other
    low cells set according to *shape* (:data:`SHAPES`), so that the average over the
    articles the shipped rule labels `other` stays at ``other_agreement`` exactly: the
    model reproduces the measured baseline `other` precision whatever the rate. Raises
    ``ValueError`` if no probability in [0, 1] does."""
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; valid: {', '.join(SHAPES)}")
    n0, n1, n2, nsc = (counts[0], counts[1], counts[2], counts[SHORT_CIRCUIT_BAND])
    total = truth.other_agreement * (n0 + n1 + n2 + nsc)
    if shape == "flat":
        rates = dict.fromkeys(LOW_BANDS, truth.other_agreement)
    else:
        rest_weight: float
        if shape == "band2":
            rest_weight, rest_mass = n0 + n1 + nsc, total - rate * n2
        else:  # graded: the 0.4-0.5 band sits halfway between the swept rate and the rest
            rest_weight, rest_mass = n1 / 2 + n0 + nsc, total - rate * n2 - rate * n1 / 2
        rest = rest_mass / rest_weight if rest_weight else 0.0
        rates = {0: rest, 1: rest if shape == "band2" else (rate + rest) / 2, 2: rate}
        rates[SHORT_CIRCUIT_BAND] = rest
    if any(not 0.0 <= r <= 1.0 for r in rates.values()):
        raise ValueError(
            f"no rate in [0, 1] for the other low cells keeps the average at "
            f"{truth.other_agreement:.3f} with a 0.5-0.6 rate of {rate:.3f} ({shape})"
        )
    return replace(truth, low_other=rates)


def fit_truth_model(
    accept_pairs: Sequence[Pair],
    *,
    slug_fraction: float = 0.5,
    min_judged: int = 10,
) -> TruthModel:
    """Measure the truth model's inputs on the stored judgements of one partition (the
    output of :func:`news_nlp.eval.partition.domain_pairs`). Diagnostic-only buckets are
    left out. Agreement rates and the `other` share of wrong labels are weighted by the
    Horvitz-Thompson weight ``N_h / n_h`` of each item (the buckets are not uniform-weight);
    the `other` agreement is the weighted ``precision_other`` of
    :func:`news_nlp.eval.metrics.aggregate_category`. Raises ``ValueError`` if a confident
    band has no judged item at all: there is nothing to fall back on, and a 0.0
    agreement would read as "every confident label is wrong"."""
    ok = [(it, v) for it, v in accept_pairs if it.bucket not in DIAGNOSTIC_ONLY_BUCKETS]
    n_bucket = Counter(it.bucket for it, _ in ok)
    hits: defaultdict[str, float] = defaultdict(float)
    seen: defaultdict[str, float] = defaultdict(float)
    count: Counter[str] = Counter()
    band_hits: defaultdict[int, float] = defaultdict(float)
    band_seen: defaultdict[int, float] = defaultdict(float)
    wrong = wrong_other = 0.0
    for it, v in ok:
        label = str(it.prediction.get("label", OTHER_LABEL))
        if label == OTHER_LABEL:
            continue
        weight = it.stratum_population / n_bucket[it.bucket]
        cell = cell_of(1.0, _scores_of(it))
        band = int(cell.rsplit("|", 1)[1])
        right = v.ideal_slug == label
        hits[cell] += weight * right
        seen[cell] += weight
        count[cell] += 1
        band_hits[band] += weight * right
        band_seen[band] += weight
        if not right:
            wrong += weight
            wrong_other += weight * (v.ideal_slug == OTHER_LABEL)
    agree: dict[str, float] = {}
    fallback: list[str] = []
    for band in range(FIRST_CONFIDENT_BAND, len(BAND_NAMES)):
        if not band_seen[band]:
            raise ValueError(
                f"no judged item with a stored slug label in band {BAND_NAMES[band]}: "
                "the truth model has no agreement rate for it"
            )
        for slug in CATEGORY_SLUGS:
            cell = f"{slug}|{band}"
            if count[cell] >= min_judged:
                agree[cell] = hits[cell] / seen[cell]
            else:
                agree[cell] = band_hits[band] / band_seen[band]
                fallback.append(cell)
    return TruthModel(
        agree=agree,
        other_given_wrong=wrong_other / wrong if wrong else 0.0,
        other_agreement=aggregate_category([it for it, _ in ok], [v for _, v in ok]).get(
            "precision_other", 0.0
        ),
        slug_fraction=slug_fraction,
        fallback_cells=tuple(fallback),
    )


class BandMeasure(NamedTuple):
    """What the stored judgements say about a set of judged articles that carry one kind of
    stored label (`other`, or a slug): one score band, or the articles a rule would
    re-label."""

    n: int  # judged items
    n_other: int  # of which the judge's `ideal_slug` is `other`
    other: float  # weighted share whose truth is `other` (Horvitz-Thompson item weights)
    low: float  # 95% stratified-bootstrap interval of that weighted share
    high: float
    top: float  # weighted share whose truth is the band's argmax slug (0 for a rule's set)


#: One judged item as the bootstrap sees it: bucket, Horvitz-Thompson weight, whether it is in
#: the set being measured, whether its truth is `other`.
_ShareRow = tuple[str, float, bool, bool]


def _share_interval(
    rows: Sequence[_ShareRow], *, reps: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of the weighted share of `other` truths in the set, by a
    stratified bootstrap: each replicate redraws every bucket's items with replacement at
    the bucket's own size (the weights stay ``N_h / n_h``). A replicate with no item in the
    set is skipped; ``(nan, nan)`` if every one is."""
    groups: dict[str, list[_ShareRow]] = {}
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    rng = random.Random(seed)  # noqa: S311 -- resampling, not cryptography
    draws: list[float] = []
    for _ in range(reps):
        num = den = 0.0
        for group in groups.values():
            for _b, weight, inside, hit in rng.choices(group, k=len(group)):
                if inside:
                    den += weight
                    num += weight * hit
        if den > 0:
            draws.append(num / den)
    if not draws:
        return math.nan, math.nan
    draws.sort()
    tail = (1 - level) / 2
    return draws[int(tail * (len(draws) - 1))], draws[round((1 - tail) * (len(draws) - 1))]


def _stored_other(it: EvalItem) -> bool:
    return str(it.prediction.get("label", OTHER_LABEL)) == OTHER_LABEL


def measure_bands(accept_pairs: Sequence[Pair], *, stored_other: bool) -> dict[int, BandMeasure]:
    """Per score band, the share of judged articles whose truth is `other` and whose truth
    is the argmax slug, among those the shipped rule labelled `other` (*stored_other*, the
    low bands 0-2) or a slug (the confident bands 3-4). Measured directly on the judge's
    ``ideal_slug``: no model. The point estimate is Horvitz-Thompson weighted (the buckets
    are not uniform-weight) and so is its stratified-bootstrap interval. Diagnostic-only
    buckets are left out."""
    ok = [(it, v) for it, v in accept_pairs if it.bucket not in DIAGNOSTIC_ONLY_BUCKETS]
    n_bucket = Counter(it.bucket for it, _ in ok)
    items: list[tuple[str, float, int, bool, bool]] = []
    for it, v in ok:
        if _stored_other(it) != stored_other:
            continue
        slug, band_text = cell_of(1.0, _scores_of(it)).rsplit("|", 1)
        items.append(
            (
                it.bucket,
                it.stratum_population / n_bucket[it.bucket],
                int(band_text),
                v.ideal_slug == OTHER_LABEL,
                v.ideal_slug == slug,
            )
        )
    out: dict[int, BandMeasure] = {}
    for band in sorted({i[2] for i in items}):
        inside = [i for i in items if i[2] == band]
        seen = sum(i[1] for i in inside)
        low, high = _share_interval([(i[0], i[1], i[2] == band, i[3]) for i in items])
        out[band] = BandMeasure(
            len(inside),
            sum(i[3] for i in inside),
            sum(i[1] for i in inside if i[3]) / seen,
            low,
            high,
            sum(i[1] for i in inside if i[4]) / seen,
        )
    return out


def measure_rule(accept_pairs: Sequence[Pair], rule: Rule) -> BandMeasure | None:
    """For a loosening *rule*, the judged articles it would take out of `other` (stored
    label `other`, label under *rule* a slug): how many, and the weighted share whose truth
    is `other` with a stratified-bootstrap interval, the quantity T-123's bound
    (:func:`loosening_bound`) is a cap on. ``None`` if no judged article is re-labelled."""
    ok = [(it, v) for it, v in accept_pairs if it.bucket not in DIAGNOSTIC_ONLY_BUCKETS]
    n_bucket = Counter(it.bucket for it, _ in ok)
    rows: list[_ShareRow] = []
    for it, v in ok:
        moved = _stored_other(it) and apply_rule(1.0, _scores_of(it), rule) != OTHER_LABEL
        rows.append(
            (
                it.bucket,
                it.stratum_population / n_bucket[it.bucket],
                moved,
                v.ideal_slug == OTHER_LABEL,
            )
        )
    inside = [r for r in rows if r[2]]
    if not inside:
        return None
    seen = sum(r[1] for r in inside)
    low, high = _share_interval(rows)
    return BandMeasure(
        len(inside),
        sum(r[3] for r in inside),
        sum(r[1] for r in inside if r[3]) / seen,
        low,
        high,
        0.0,
    )


class SubsetEffect(NamedTuple):
    """The change a loosening rule brings if the articles it removes are truly `other` at
    a given rate: ``recall`` and ``precision`` differences of the `other` class."""

    recall: float
    precision: float


def subset_effect(
    labelled_other: int, removed: int, rate: float, precision: float, recall: float
) -> SubsetEffect:
    """Closed-form effect of removing *removed* of the *labelled_other* articles from
    `other` when a share *rate* of them is truly `other` (X = rate x removed), from the
    baseline *precision* and *recall* alone, as in :func:`loosening_bound`: recall changes
    by ``-X / T`` and precision to ``(H - X) / (labelled_other - removed)``."""
    hits = precision * labelled_other
    total = hits / recall
    lost = rate * removed
    return SubsetEffect(-lost / total, (hits - lost) / (labelled_other - removed) - precision)


def pooled_slug_fraction(
    measured: Mapping[int, BandMeasure], counts: Mapping[int, int]
) -> float | None:
    """The measured share of the non-`other` mass whose truth is the argmax slug, pooled
    over the low bands 0-2 by the number of articles the shipped rule labels `other` there
    (what :attr:`TruthModel.slug_fraction` stands for). ``None`` if nothing was measured."""
    top = rest = 0.0
    for band, m in measured.items():
        if band in LOW_BANDS and band != SHORT_CIRCUIT_BAND:
            top += counts.get(band, 0) * m.top
            rest += counts.get(band, 0) * (1 - m.other)
    return top / rest if rest else None


class PowerRow(NamedTuple):
    rule: Rule
    rate: float  # the probability that a 0.5-0.6 article is truly `other`
    implied_gain: float  # population difference of the weighted `other` precision
    power: float  # share of draws whose interval's lower bound is above 0
    defined: int  # draws with a usable interval (the bootstrap kept a replicate)
    mean_diff: float = 0.0  # mean over draws of the sample's paired point difference
    sd_diff: float = 0.0  # its standard deviation over draws (the sampling spread)
    mean_lower: float = 0.0  # mean over draws of the interval's lower bound
    critical: float = 0.0  # the lower-bound value a draw has to exceed to count
    joint_power: float = 0.0  # share of draws that also keep recall and accuracy within tolerance
    recall_diff: float = 0.0  # mean over draws of the sample's paired `other`-recall difference
    accuracy_diff: float = 0.0  # the same for overall accuracy

    @property
    def power_se(self) -> float:
        """Monte Carlo standard error of :attr:`power` (binomial over the draws)."""
        return math.sqrt(self.power * (1 - self.power) / self.defined) if self.defined else 0.0


@dataclass(frozen=True)
class Frame:
    """The articles under one rule: the cell index of each article and whether the
    shipped rule (``base_other``) and the candidate (``cand_other``) label it ``other``.
    Built once per rule; the truth model is applied to it separately."""

    rule: Rule
    names: tuple[str, ...]
    cell: np.ndarray
    base_other: np.ndarray
    cand_other: np.ndarray

    @property
    def signature(self) -> bytes:
        """Two rules with the same signature label every article alike."""
        return np.packbits(self.cand_other).tobytes()


def frame_for(articles: Sequence[ConfirmedArticle], rule: Rule) -> Frame:
    cells = [cell_of(a.group_score, a.leaf_scores) for a in articles]
    names = tuple(sorted(set(cells)))
    index = {c: i for i, c in enumerate(names)}
    return Frame(
        rule,
        names,
        np.array([index[c] for c in cells]),
        np.array([apply_rule(a.group_score, a.leaf_scores) == OTHER_LABEL for a in articles]),
        np.array([apply_rule(a.group_score, a.leaf_scores, rule) == OTHER_LABEL for a in articles]),
    )


def _cell_probabilities(frame: Frame, truth: TruthModel) -> tuple[np.ndarray, np.ndarray]:
    probs = [truth.probabilities(name) for name in frame.names]
    return np.array([p[0] for p in probs]), np.array([p[1] for p in probs])


class Implied(NamedTuple):
    """Candidate minus shipped rule, in expectation under a truth model (no sampling)."""

    precision: float  # weighted `other` precision
    recall: float  # `other` recall
    accuracy: float  # overall accuracy against the truth label


#: T-123's tolerance (owner-confirmed, TASKS.md): the paired differences in `other` recall,
#: overall accuracy and macro F1 must each have a 95% lower bound no worse than this.
RECALL_TOLERANCE = -0.02


def implied_changes(frame: Frame, truth: TruthModel) -> Implied:
    """The change in `other` precision, `other` recall and overall accuracy the rule
    brings under *truth*, as expectations: precision is the sum of P(truth = other) over
    the articles labelled ``other`` over their count, recall the same sum over the sum of
    P(truth = other) of all articles, accuracy the mean probability that the label is the
    truth (``other`` is right with P(other), the argmax slug with P(argmax slug)). Macro F1
    is not derived: the truth model puts the rest of the mass on an unspecified third
    label, so a per-class F1 has no class to land on; T-121(b) measures it."""
    p_slug_cell, p_other_cell = _cell_probabilities(frame, truth)
    p_slug, p_other = p_slug_cell[frame.cell], p_other_cell[frame.cell]
    total_other = p_other.sum()

    def stats(flag: np.ndarray) -> tuple[float, float, float]:
        hit = float((p_other * flag).sum())
        precision = hit / flag.sum() if flag.sum() else 0.0
        recall = hit / total_other if total_other else 0.0
        accuracy = float((np.where(flag, p_other, p_slug)).mean())
        return precision, recall, accuracy

    cand, base = stats(frame.cand_other), stats(frame.base_other)
    return Implied(cand[0] - base[0], cand[1] - base[1], cand[2] - base[2])


def implied_gain(frame: Frame, truth: TruthModel) -> float:
    """Population difference of the weighted `other` precision under *truth* (an
    expectation, no sampling): :attr:`Implied.precision` of :func:`implied_changes`."""
    return implied_changes(frame, truth).precision


def _at_rate(truth: TruthModel, counts: Mapping[int, int], shape: str, rate: float) -> TruthModel:
    return shaped(truth, counts, rate, shape)


def rate_for_gain(
    frame: Frame,
    truth: TruthModel,
    counts: Mapping[int, int],
    shape: str,
    gain: float,
    *,
    tol: float = 1e-5,
) -> float | None:
    """The probability that a 0.5-0.6 article is truly `other` at which the rule's implied
    `other`-precision gain equals *gain* (:func:`shaped` fixes the other cells), by bisection
    on [0, the measured `other` precision]: the gain is 0 at the measured value (nothing is
    cleaner than anything else) and grows as the rate falls. ``None`` if even a rate of 0
    does not reach it."""
    hi = truth.other_agreement
    lo = 0.0
    if implied_gain(frame, _at_rate(truth, counts, shape, lo)) < gain:
        return None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if implied_gain(frame, _at_rate(truth, counts, shape, mid)) >= gain:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def rate_for_recall(
    frame: Frame,
    truth: TruthModel,
    counts: Mapping[int, int],
    shape: str,
    bound: float = RECALL_TOLERANCE,
    *,
    tol: float = 1e-5,
) -> float | None:
    """The largest probability that a 0.5-0.6 article is truly `other` at which the rule's
    implied `other`-recall change is still at least *bound*, by bisection on [0, the
    measured `other` precision]; ``None`` if no rate on it keeps recall that high, and the
    measured value itself if even that does (a rule that removes nothing). The recall change
    falls as the rate rises: more of the removed articles were `other`."""
    hi = truth.other_agreement
    if implied_changes(frame, _at_rate(truth, counts, shape, hi)).recall >= bound:
        return hi
    lo = 0.0
    if implied_changes(frame, _at_rate(truth, counts, shape, lo)).recall < bound:
        return None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if implied_changes(frame, _at_rate(truth, counts, shape, mid)).recall >= bound:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def baseline_under(frame: Frame, truth: TruthModel) -> tuple[float, float]:
    """The shipped rule's `other` precision and recall under *truth*: what a calibrated
    model must reproduce from the stored run (measured 0.319 and 0.661 on eval_run 43's
    `accept` partition)."""
    p_other = _cell_probabilities(frame, truth)[1][frame.cell]
    hit = float((p_other * frame.base_other).sum())
    return hit / frame.base_other.sum(), hit / float(p_other.sum())


def confident_other_rates(
    truth: TruthModel, articles: Sequence[ConfirmedArticle]
) -> dict[int, float]:
    """*Modelled* probability that an article the shipped rule labels with a slug is truly
    `other`, per confident band (3: 0.6-0.8, 4: >= 0.8), averaged over *articles*:
    ``(1 - agreement) x other_given_wrong`` with one pooled `other` share for all bands.
    :func:`measure_bands` gives the measured value; the script prints both."""
    totals: dict[int, float] = dict.fromkeys(range(FIRST_CONFIDENT_BAND, len(BAND_NAMES)), 0.0)
    counts: dict[int, int] = dict.fromkeys(totals, 0)
    for a in articles:
        cell = cell_of(a.group_score, a.leaf_scores)
        band = _band_of_cell(cell)
        if band is None or band < FIRST_CONFIDENT_BAND:
            continue
        totals[band] += truth.probabilities(cell)[1]
        counts[band] += 1
    return {b: totals[b] / counts[b] for b in totals if counts[b]}


class LooseningBound(NamedTuple):
    """What a loosening rule must achieve to meet T-123, from the measured aggregates alone
    (no truth model): of the *removed* articles at most ``max_truth_other`` may be truly
    `other` (``max_rate`` of them), and at least ``min_removed`` must be removed."""

    labelled_other: int  # articles the shipped rule labels `other`
    removed: int  # articles the rule takes out of `other`
    truth_other_total: float  # truly-`other` articles in the population
    max_truth_other: float  # allowed truly-`other` articles among the removed
    max_rate: float  # the same as a share of the removed
    min_removed: float  # the fewest articles that can reach the precision gain


def loosening_bound(
    labelled_other: int,
    removed: int,
    precision: float,
    recall: float,
    *,
    gain: float = 0.05,
    tolerance: float = -RECALL_TOLERANCE,
) -> LooseningBound:
    """The bound a loosening rule has to meet to raise `other` precision by *gain* while
    keeping `other` recall within *tolerance*, given the measured baseline *precision* and
    *recall* of the articles labelled `other`. With ``H = precision x labelled_other``
    truly-`other` articles among them and ``T = H / recall`` in the population, removing
    *removed* articles of which X are truly `other` gives recall change ``-X / T`` and
    precision ``(H - X) / (labelled_other - removed)``. Both conditions hold iff
    ``X <= tolerance x T`` and ``X <= H - (precision + gain) x (labelled_other - removed)``;
    the second is looser for the large removals this task considers. It is a necessary
    condition: T-123 asks for the 95% *lower* bound of the recall change to be within the
    tolerance, which needs more room than the point estimate."""
    hits = precision * labelled_other
    total = hits / recall
    by_recall = tolerance * total
    by_precision = hits - (precision + gain) * (labelled_other - removed)
    allowed = max(0.0, min(by_recall, by_precision))
    min_removed = (labelled_other - (hits - by_recall) / (precision + gain)) if by_recall else 0.0
    return LooseningBound(
        labelled_other,
        removed,
        total,
        allowed,
        allowed / removed if removed else 0.0,
        min_removed,
    )


#: T-121(a)'s detectability bar and the width of the zone around it that is re-run with more
#: draws (a 200-draw power has a standard error up to 0.035, too close to the bar to call).
POWER_TARGET = 0.80
BORDERLINE = 0.10


def classify_power(power: float, se: float, *, target: float = POWER_TARGET) -> str:
    """``"detectable"`` when the power clears *target* by two standard errors,
    ``"not detectable"`` when it falls short by two, else ``"borderline"``. A borderline
    candidate is treated like a miss: it takes the discordant top-up (TASKS.md T-121(a))."""
    if power - 2 * se >= target:
        return "detectable"
    if power + 2 * se < target:
        return "not detectable"
    return "borderline"


def needs_refinement(power: float, *, target: float = POWER_TARGET) -> bool:
    """Whether a power is within :data:`BORDERLINE` of *target* and is re-run with more draws."""
    return abs(power - target) < BORDERLINE


def null_truth(truth: TruthModel) -> TruthModel:
    """*truth* with the same ``other`` probability in every cell and no argmax-slug share:
    exchangeable articles, so every rule's gain is exactly 0. Its power is the estimator's
    false-positive rate."""
    return replace(
        truth,
        agree=dict.fromkeys(truth.agree, 0.0),
        other_given_wrong=truth.other_agreement,
        low_other=dict.fromkeys(LOW_BANDS, truth.other_agreement),
        slug_fraction=0.0,
    )


@dataclass(frozen=True)
class _Draws:
    """Per defined draw: the lower bound of the precision, recall and accuracy differences
    (an undefined recall or accuracy bound is ``-inf``, a failure) and the sample's paired
    point differences."""

    lower: list[float]
    point: list[float]
    recall_lower: list[float]
    recall_point: list[float]
    accuracy_lower: list[float]
    accuracy_point: list[float]


def _lower(diff: np.ndarray, level: float) -> float:
    """Nearest-rank lower percentile of the usable replicates."""
    k = max(1, math.ceil((1 - level) / 2 * len(diff)))
    return float(np.sort(diff)[min(k, len(diff)) - 1])


def _draw_summaries(
    frame: Frame,
    truth: TruthModel,
    *,
    draws: int,
    boot_reps: int,
    seed: int,
    size_of: Callable[[str], int],
    level: float,
) -> _Draws:
    """Simulate *draws* report-half samples. Draws come from one random stream, so a
    longer run starts with the draws of a shorter one that has the same seed."""
    rng = np.random.default_rng(seed)
    p_slug_cell, p_other_cell = _cell_probabilities(frame, truth)
    p_slug, p_other = p_slug_cell[frame.cell], p_other_cell[frame.cell]
    members = [np.flatnonzero(frame.cell == i) for i in range(len(frame.names))]
    take = [min(size_of(name), len(m)) for name, m in zip(frame.names, members, strict=True)]
    pops = [len(m) for m in members]
    out = _Draws([], [], [], [], [], [])
    for _ in range(draws):
        idx, weight = [], []
        for m, n, pop in zip(members, take, pops, strict=True):
            chosen = rng.choice(m, size=n, replace=False)
            idx.append(chosen)
            weight.append(np.full(n, pop / n))
        sel = np.concatenate(idx)
        w = np.concatenate(weight)
        roll = rng.random(len(sel))
        # P(other) is stored as an absolute probability; resolve the three-way draw
        # from one uniform: [0, p_slug) slug, [p_slug, p_slug + p_other) other, else third.
        is_slug = roll < p_slug[sel]
        is_other = (roll >= p_slug[sel]) & (roll < p_slug[sel] + p_other[sel])
        base = frame.base_other[sel].astype(float)
        cand = frame.cand_other[sel].astype(float)
        base_hit = base * is_other
        cand_hit = cand * is_other
        # an article is labelled correctly if `other` and the truth is `other`, or the slug
        # and the truth is the argmax slug
        base_right = base_hit + (1 - base) * is_slug
        cand_right = cand_hit + (1 - cand) * is_slug
        # bootstrap counts, within cell
        counts = np.zeros((boot_reps, len(sel)))
        start = 0
        for n in take:
            if n:
                counts[:, start : start + n] = rng.multinomial(n, np.full(n, 1 / n), size=boot_reps)
            start += n
        wc = counts * w
        den_b, den_c = wc @ base, wc @ cand
        ok = (den_b > 0) & (den_c > 0)
        if not ok.any():
            continue
        diff = (wc @ cand_hit)[ok] / den_c[ok] - (wc @ base_hit)[ok] / den_b[ok]
        out.lower.append(_lower(diff, level))
        out.point.append(float(w @ cand_hit / (w @ cand) - w @ base_hit / (w @ base)))
        # `other` recall: the same truth-other total under both rules
        truth_other = wc @ is_other
        rok = truth_other > 0
        if rok.any():
            rdiff = (wc @ (cand_hit - base_hit))[rok] / truth_other[rok]
            out.recall_lower.append(_lower(rdiff, level))
        else:
            out.recall_lower.append(-math.inf)
        total = wc.sum(axis=1)
        adiff = (wc @ (cand_right - base_right)) / total
        out.accuracy_lower.append(_lower(adiff, level))
        out.recall_point.append(
            float(w @ (cand_hit - base_hit) / (w @ is_other)) if is_other.any() else 0.0
        )
        out.accuracy_point.append(float(w @ (cand_right - base_right) / w.sum()))
    return out


def null_critical(
    frame: Frame,
    truth: TruthModel,
    *,
    draws: int = 1000,
    boot_reps: int = 300,
    seed: int = 0,
    size_of: Callable[[str], int] = report_half_size,
    level: float = 0.95,
) -> float:
    """The lower-bound value that only ``(1 - level) / 2`` of the draws exceed when the
    true gain is 0 (:func:`null_truth`): the critical value that makes this interval's
    false-positive rate nominal at this sample size. Pass it to :func:`simulate_power` as
    ``critical`` for a power that does not benefit from a liberal interval. Its own
    sampling error is that of a 2.5% tail of *draws* draws: vary *seed* to see it."""
    lowers = _draw_summaries(
        frame,
        null_truth(truth),
        draws=draws,
        boot_reps=boot_reps,
        seed=seed,
        size_of=size_of,
        level=level,
    ).lower
    if not lowers:
        return math.nan
    rank = math.ceil((1 - (1 - level) / 2) * len(lowers))
    return max(0.0, sorted(lowers)[rank - 1])


def simulate_power(
    frame: Frame,
    truth: TruthModel,
    *,
    draws: int = 200,
    boot_reps: int = 300,
    seed: int = 0,
    size_of: Callable[[str], int] = report_half_size,
    level: float = 0.95,
    critical: float = 0.0,
    tolerance: float = RECALL_TOLERANCE,
) -> PowerRow:
    """Power of T-123's estimator to detect the frame rule's gain under *truth*.

    Returns a :class:`PowerRow`; ``defined`` counts the draws on which the bootstrap kept
    at least one replicate (a replicate is dropped when either rule has no `other`
    label in the resample), and the point-difference and lower-bound summaries explain a
    power: a high power with a small spread is a nearly deterministic difference, not a
    luckier estimator.

    Each draw takes the planned report-half sample per cell without replacement (all of
    a smaller cell), draws every sampled article's truth from the truth model, forms
    the paired difference of the population-weighted ``other`` precision, and takes a
    stratified (within-cell) bootstrap interval with nearest-rank percentiles. ``power``
    is the share of draws whose lower bound is above *critical* (0 by default; the nominal
    test). ``joint_power`` also requires the lower bounds of the paired `other`-recall and
    overall-accuracy differences to be no worse than *tolerance* (T-123's conditions; macro
    F1 is not modelled, see :func:`implied_changes`). A draw whose bootstrap leaves the
    denominator empty in every replicate is not counted as defined.
    """
    d = _draw_summaries(
        frame,
        truth,
        draws=draws,
        boot_reps=boot_reps,
        seed=seed,
        size_of=size_of,
        level=level,
    )
    defined = len(d.lower)
    joint = sum(
        lo > critical and rl >= tolerance and al >= tolerance
        for lo, rl, al in zip(d.lower, d.recall_lower, d.accuracy_lower, strict=True)
    )
    return PowerRow(
        frame.rule,
        truth.low_rate(2),
        implied_gain(frame, truth),
        sum(lo > critical for lo in d.lower) / defined if defined else 0.0,
        defined,
        float(np.mean(d.point)) if d.point else 0.0,
        float(np.std(d.point)) if d.point else 0.0,
        float(np.mean(d.lower)) if d.lower else 0.0,
        critical,
        joint / defined if defined else 0.0,
        float(np.mean(d.recall_point)) if d.recall_point else 0.0,
        float(np.mean(d.accuracy_point)) if d.accuracy_point else 0.0,
    )


class PairedInterval(NamedTuple):
    """Rule minus shipped rule on one metric: the point difference, the bootstrap bounds
    and how many of the ``reps`` replicates were usable."""

    difference: float
    low: float
    high: float
    used: int
    reps: int


def paired_rule_interval(
    pairs: Sequence[Pair],
    rule: Rule,
    key: str,
    *,
    reps: int = 500,
    seed: int = 0,
    level: float = 0.95,
) -> PairedInterval:
    """``metric(rule) - metric(shipped rule)`` on the same stored items, with a stratified
    bootstrap (each replicate redraws every bucket's items with replacement at the
    bucket's own size and scores both rules on that one resample, so the difference is
    paired). *pairs* is one partition from :func:`news_nlp.eval.partition.domain_pairs`;
    *rule* must tighten the shipped one (see :func:`retag`). A replicate is skipped when a
    ``precision_``/``recall_`` metric has an empty denominator under either rule.
    Nearest-rank percentiles, approximate to about a hundredth. Free: no judge call."""

    def metric(sample: Sequence[Pair], tagged: bool) -> float:
        items = [retag(it, rule) if tagged else it for it, _ in sample]
        if not is_defined(
            key, [(it, v) for it, (_, v) in zip(items, sample, strict=True)], {}, "all"
        ):
            return math.nan
        return aggregate_category(items, [v for _, v in sample]).get(key, math.nan)

    def difference(sample: Sequence[Pair]) -> float:
        return metric(sample, True) - metric(sample, False)

    by_bucket: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_bucket.setdefault(pair[0].bucket, []).append(pair)
    rng = random.Random(seed)  # noqa: S311 -- resampling, not cryptography
    draws: list[float] = []
    for _ in range(reps):
        value = difference([p for g in by_bucket.values() for p in rng.choices(g, k=len(g))])
        if math.isfinite(value):
            draws.append(value)
    point = difference(pairs)
    if not draws:
        return PairedInterval(point, math.nan, math.nan, 0, reps)
    draws.sort()
    tail = (1 - level) / 2
    return PairedInterval(
        point,
        draws[int(tail * (len(draws) - 1))],
        draws[round((1 - tail) * (len(draws) - 1))],
        len(draws),
        reps,
    )
