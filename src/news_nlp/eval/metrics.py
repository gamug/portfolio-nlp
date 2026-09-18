"""Pure aggregation of per-row judge verdicts into flat metric dicts.

Every ``aggregate_*`` returns ``dict[str, float]`` ready for
``mlflow.log_metrics``. Rows the judge could not parse (``parse_failed``) are
excluded from the accuracy numbers and counted in ``parse_fail_rate``. The judge
is treated as ground truth, so "F1 vs judge" / "accuracy vs judge" are
agreement measures, not truth measures -- see ``docs/evaluation.md``.
``aggregate_sentiment``/``aggregate_category`` also gain one-vs-rest
``roc_auc_<class>`` (TASKS.md T-093, SPEC.md FR-016), computed from each
sampled row's own stored per-class probability -- see ``_weighted_auc``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from statistics import fmean
from typing import Any

from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.verdicts import (
    CategoryVerdict,
    NerVerdict,
    SectorIntroVerdict,
    SentimentVerdict,
    SummaryVerdict,
)
from news_nlp.taxonomy import CATEGORY_SLUGS

#: The single metric ``regression.check_regression`` compares between runs.
#:
#: ``sentiment`` is ``recall_negative``: missing a real negative-sentiment
#: article costs more here than over-flagging a neutral one as negative --
#: negative sentiment is the signal portfolio construction leans on, so false
#: negatives are the regression that matters. See docs/evaluation.md's "Why
#: recall, not F1, for sentiment negative" note. Unaffected by
#: ``aggregate_sentiment`` being narrowed to one-vs-rest-only metrics
#: (2026-09-15, constitution AI behavior #12) -- ``recall_negative`` was
#: already a per-class metric, not a ``macro_f1_vs_judge``-style aggregate.
HEADLINE: dict[str, str] = {
    "sentiment": "recall_negative",
    "category": "accuracy_vs_judge",
    "ner": "micro_f1",
    "c_summary": "mean_faithfulness",
    "sector_summary": "mean_faithfulness",
}

_SENTIMENT_CLASSES = ("positive", "negative", "neutral")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _macro_f1(
    pairs: Sequence[tuple[str, str]], classes: Iterable[str]
) -> tuple[float, dict[str, tuple[float, float, float]]]:
    """``pairs`` = ``(truth, pred)``. Macro-F1 over ``classes`` + per-class PRF."""
    per: dict[str, tuple[float, float, float]] = {}
    for c in classes:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        per[c] = _prf(tp, fp, fn)
    macro = fmean(f1 for _, _, f1 in per.values()) if per else 0.0
    return macro, per


def _rate(flags: Sequence[bool]) -> float:
    return fmean(1.0 if f else 0.0 for f in flags) if flags else 0.0


#: Strata that are deliberately non-representative by construction (worst-case
#: stress test / soft-probability class targeting) and must never feed a
#: population-estimate metric. See sampling.py's module docstring and
#: docs/evaluation.md's "Statistical methodology" section.
_DIAGNOSTIC_ONLY_BUCKETS: frozenset[str] = frozenset({"low_conf"})


def _ht_sum(
    items: Sequence[EvalItem],
    flags: Sequence[float],
    *,
    exclude: frozenset[str] = _DIAGNOSTIC_ONLY_BUCKETS,
) -> float:
    """Horvitz-Thompson population-total estimate of *flags* from a disjoint
    stratified sample: each non-excluded stratum ``h`` contributes
    ``(N_h / n_h') * sum(flags within h)``, where ``n_h'`` is the count of
    items actually present here (already parse-failure-filtered by the
    caller -- ``aggregate_*`` passes ``ok``-filtered items) -- NOT the
    original draw count. This is a deliberate MCAR assumption: a judge parse
    failure is treated as independent of the row's true label, so survivors
    of stratum ``h`` are still ~a uniform sample of size ``n_h'`` from ``N_h``
    (see docs/evaluation.md). A stratum with ``n_h' == 0`` contributes nothing
    (never a ``ZeroDivisionError``)."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    populations: dict[str, int] = {}
    for it, flag in zip(items, flags, strict=True):
        if it.bucket in exclude:
            continue
        totals[it.bucket] = totals.get(it.bucket, 0.0) + flag
        counts[it.bucket] = counts.get(it.bucket, 0) + 1
        populations[it.bucket] = it.stratum_population
    return sum((populations[b] / n) * totals[b] for b, n in counts.items() if n)


def _ht_ratio(
    items: Sequence[EvalItem],
    numerator_flags: Sequence[float],
    denominator_flags: Sequence[float],
    *,
    exclude: frozenset[str] = _DIAGNOSTIC_ONLY_BUCKETS,
) -> float:
    """HT estimate of ``sum(numerator)/sum(denominator)`` over the population.
    A weighted mean is the special case
    ``_ht_ratio(items, values, [1.0] * len(items))``."""
    num = _ht_sum(items, numerator_flags, exclude=exclude)
    den = _ht_sum(items, denominator_flags, exclude=exclude)
    return num / den if den else 0.0


def _prf_ht(
    items: Sequence[EvalItem],
    tp_flags: Sequence[float],
    fp_flags: Sequence[float],
    fn_flags: Sequence[float],
    *,
    exclude: frozenset[str] = _DIAGNOSTIC_ONLY_BUCKETS,
) -> tuple[float, float, float]:
    tp = _ht_sum(items, tp_flags, exclude=exclude)
    fp = _ht_sum(items, fp_flags, exclude=exclude)
    fn = _ht_sum(items, fn_flags, exclude=exclude)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _macro_f1_ht(
    items: Sequence[EvalItem], pairs: Sequence[tuple[str, str]], classes: Iterable[str]
) -> tuple[float, dict[str, tuple[float, float, float]]]:
    """HT-aware drop-in for ``_macro_f1`` -- same per-class shape, weighted
    sums instead of flat counts. ``pairs`` unchanged: ``(truth, pred)``."""
    per: dict[str, tuple[float, float, float]] = {}
    for c in classes:
        tp = [1.0 if t == c and p == c else 0.0 for t, p in pairs]
        fp = [1.0 if t != c and p == c else 0.0 for t, p in pairs]
        fn = [1.0 if t == c and p != c else 0.0 for t, p in pairs]
        per[c] = _prf_ht(items, tp, fp, fn)
    macro = fmean(f1 for _, _, f1 in per.values()) if per else 0.0
    return macro, per


def _per_bucket(items: Sequence[EvalItem], values: Sequence[float]) -> dict[str, float]:
    """Diagnostic per-stratum mean -- never HT-reweighted (describes only that
    stratum's own drawn sample, not the population). Generalizes the old
    2-hardcoded-bucket ``low_conf``/``random`` split to however many strata a
    run actually used (dynamic ``target_<x>`` names included)."""
    out: dict[str, float] = {}
    for bucket in sorted({it.bucket for it in items}):
        vals = [v for it, v in zip(items, values, strict=True) if it.bucket == bucket]
        out[bucket] = fmean(vals) if vals else 0.0
    return out


def _ht_weights(
    items: Sequence[EvalItem], *, exclude: frozenset[str] = _DIAGNOSTIC_ONLY_BUCKETS
) -> list[float]:
    """Per-item Horvitz-Thompson weight (``population_h / n_h'``), 0.0 for an
    excluded (diagnostic-only) bucket -- the same per-bucket reweighting
    ``_ht_sum`` folds into one running total, exposed per-item here because
    ``_weighted_auc``'s rank statistic needs a weight per row, not a single
    population-total sum."""
    counts: dict[str, int] = {}
    for it in items:
        if it.bucket in exclude:
            continue
        counts[it.bucket] = counts.get(it.bucket, 0) + 1
    return [
        (it.stratum_population / counts[it.bucket]) if it.bucket not in exclude else 0.0
        for it in items
    ]


def _weighted_auc(pos: Sequence[tuple[float, float]], neg: Sequence[tuple[float, float]]) -> float:
    """One-vs-rest ROC AUC via the weighted Mann-Whitney U statistic -- the
    share of (positive, negative) weight ever outranked by the positive
    item's score (ties count half), which is exactly the area under the ROC
    curve traced by sweeping a threshold over every observed score.
    Reduces to the standard rank-based AUC when every weight is 1.0 (this
    module's ``_naive_pooled`` convention); with HT weights (``_ht_weights``)
    it's the natural generalization of the same reweighting ``_ht_sum``/
    ``_ht_ratio`` apply to flag-sums elsewhere in this module. O(n log n)
    (sort once, single pass grouping ties) rather than the O(n^2) the
    definition above suggests -- real eval runs sample thousands of rows
    (docs/evaluation.md), so this matters. 0.0 (undefined, same
    zero-denominator convention as ``_prf``/``_ht_ratio``) when either class
    carries no weight in this sample."""
    total_pos_w = sum(w for _, w in pos)
    total_neg_w = sum(w for _, w in neg)
    total_pairs_w = total_pos_w * total_neg_w
    if not total_pairs_w:
        return 0.0
    tagged = sorted(
        [(s, w, True) for s, w in pos] + [(s, w, False) for s, w in neg], key=lambda t: t[0]
    )
    concordant = 0.0
    cum_neg_before = 0.0
    i, n = 0, len(tagged)
    while i < n:
        score = tagged[i][0]
        j = i
        group_pos_w = group_neg_w = 0.0
        while j < n and tagged[j][0] == score:
            if tagged[j][2]:
                group_pos_w += tagged[j][1]
            else:
                group_neg_w += tagged[j][1]
            j += 1
        concordant += group_pos_w * cum_neg_before + 0.5 * group_pos_w * group_neg_w
        cum_neg_before += group_neg_w
        i = j
    return concordant / total_pairs_w


def _roc_auc_per_class(
    it_ok: Sequence[EvalItem],
    truths: Sequence[str],
    classes: Iterable[str],
    score_of: Callable[[EvalItem, str], float],
) -> dict[str, float]:
    """``roc_auc_<cls>`` (HT-weighted) + ``roc_auc_<cls>_naive_pooled`` for
    every *classes*, one-vs-rest against *truths* using each item's own raw
    per-class probability (*score_of*). Shared by ``aggregate_sentiment``/
    ``aggregate_category`` -- unlike ``confusion_pairs``'s deliberate 2-line
    duplication, this loop (build pos/neg, call ``_weighted_auc`` twice) is
    identical between the two callers, only the score lookup and class set
    differ."""
    ht_weights = _ht_weights(it_ok)
    naive_weights = [1.0] * len(it_ok)
    out: dict[str, float] = {}
    for cls in classes:
        scores = [score_of(it, cls) for it in it_ok]
        for weights, suffix in ((ht_weights, ""), (naive_weights, "_naive_pooled")):
            pos = [(s, w) for s, w, t in zip(scores, weights, truths, strict=True) if t == cls]
            neg = [(s, w) for s, w, t in zip(scores, weights, truths, strict=True) if t != cls]
            out[f"roc_auc_{cls}{suffix}"] = _weighted_auc(pos, neg)
    return out


def aggregate_sentiment(
    items: Sequence[EvalItem], verdicts: Sequence[SentimentVerdict]
) -> dict[str, float]:
    """One-vs-rest metrics only -- precision/recall/F1/accuracy_ovr per class
    (HT-weighted and naive-pooled), plus `n`/`parse_fail_rate` run bookkeeping.
    No aggregate/multi-class summary (`agreement_rate`, `macro_f1_vs_judge`,
    `mean_severity`) is computed for sentiment -- constitution.md AI behavior
    #12, amended 2026-09-15 after repeated confusion mixing per-class and
    blended-multi-class numbers in the same report (see docs/evaluation.md's
    2026-09-15 "OVR-only" follow-up). `HEADLINE["sentiment"]` (recall_negative)
    is unaffected -- it was already a per-class metric, not an aggregate one.
    category/NER/c_summary keep their existing complete metric set, aggregate
    included -- this narrowing is sentiment-specific, not project-wide."""
    ok = [(it, v) for it, v in zip(items, verdicts, strict=True) if not v.parse_failed]
    total = len(verdicts)
    out: dict[str, float] = {
        "n": float(len(ok)),
        "parse_fail_rate": (total - len(ok)) / total if total else 0.0,
    }
    if not ok:
        return out
    it_ok = [it for it, _ in ok]
    pairs = [(v.ideal_label, str(it.prediction.get("label", "neutral"))) for it, v in ok]
    _, per_ht = _macro_f1_ht(it_ok, pairs, _SENTIMENT_CLASSES)
    _, per_naive = _macro_f1(pairs, _SENTIMENT_CLASSES)
    for cls, (prec, rec, f1) in per_ht.items():
        out[f"precision_{cls}"] = prec
        out[f"recall_{cls}"] = rec
        out[f"f1_{cls}"] = f1
    for cls, (prec, rec, f1) in per_naive.items():
        out[f"precision_{cls}_naive_pooled"] = prec
        out[f"recall_{cls}_naive_pooled"] = rec
        out[f"f1_{cls}_naive_pooled"] = f1
    # One-vs-rest binary accuracy per class: "is this label or not," collapsing
    # the other two classes into a single negative class -- same formula/
    # naming as aggregate_category's accuracy_ovr_<slug>, kept consistent
    # across every stage (constitution.md AI behavior #12).
    ones = [1.0] * len(it_ok)
    for cls in _SENTIMENT_CLASSES:
        ovr_hit_f = [1.0 if (t == cls) == (p == cls) else 0.0 for t, p in pairs]
        out[f"accuracy_ovr_{cls}"] = _ht_ratio(it_ok, ovr_hit_f, ones)
        out[f"accuracy_ovr_{cls}_naive_pooled"] = _rate([bool(f) for f in ovr_hit_f])
    # One-vs-rest ROC AUC per class, from each row's own raw softmax score
    # (article_sentiment's positive/negative/neutral columns) -- TASKS.md
    # T-093, SPEC.md FR-016.
    truths = [t for t, _ in pairs]
    out.update(
        _roc_auc_per_class(
            it_ok, truths, _SENTIMENT_CLASSES, lambda it, cls: float(it.prediction.get(cls, 0.0))
        )
    )
    return out


def aggregate_category(
    items: Sequence[EvalItem], verdicts: Sequence[CategoryVerdict]
) -> dict[str, float]:
    ok = [(it, v) for it, v in zip(items, verdicts, strict=True) if not v.parse_failed]
    total = len(verdicts)
    out: dict[str, float] = {
        "n": float(len(ok)),
        "parse_fail_rate": (total - len(ok)) / total if total else 0.0,
    }
    if not ok:
        return out
    it_ok = [it for it, _ in ok]
    correct = [str(it.prediction.get("label", "other")) == v.ideal_slug for it, v in ok]
    correct_f = [1.0 if c else 0.0 for c in correct]
    ones = [1.0] * len(it_ok)
    pairs = [(v.ideal_slug, str(it.prediction.get("label", "other"))) for it, v in ok]
    classes = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    macro_ht, per_ht = _macro_f1_ht(it_ok, pairs, classes)
    macro_naive, per_naive = _macro_f1(pairs, classes)
    out["accuracy_vs_judge"] = _ht_ratio(it_ok, correct_f, ones)
    out["accuracy_vs_judge_naive_pooled"] = _rate(correct)
    for bucket, rate in _per_bucket(it_ok, correct_f).items():
        out[f"accuracy_vs_judge_{bucket}"] = rate
    out["macro_f1"] = macro_ht
    out["macro_f1_naive_pooled"] = macro_naive
    model_other_f = [1.0 if str(it.prediction.get("label")) == "other" else 0.0 for it, _ in ok]
    judge_other_f = [1.0 if v.ideal_slug == "other" else 0.0 for _, v in ok]
    out["other_rate_model"] = _ht_ratio(it_ok, model_other_f, ones)
    out["other_rate_model_naive_pooled"] = _rate([bool(f) for f in model_other_f])
    out["other_rate_judge"] = _ht_ratio(it_ok, judge_other_f, ones)
    out["other_rate_judge_naive_pooled"] = _rate([bool(f) for f in judge_other_f])
    out["mean_severity"] = fmean(v.severity for _, v in ok)
    # per-judge-slug accuracy (recall of the model on that slug) -- kept for
    # backward compat with earlier runs/dashboards; identical to recall_<slug>
    # below (same HT algebra), just computed by hand instead of reused from
    # per_ht.
    for slug in sorted({v.ideal_slug for _, v in ok}):
        truth_f = [1.0 if v.ideal_slug == slug else 0.0 for _, v in ok]
        hit_f = [c if t else 0.0 for c, t in zip(correct_f, truth_f, strict=True)]
        out[f"acc_{slug}"] = _ht_ratio(it_ok, hit_f, truth_f)
        hits = [c for (it, v), c in zip(ok, correct, strict=True) if v.ideal_slug == slug]
        out[f"acc_{slug}_naive_pooled"] = _rate(hits)
    # One-vs-rest per-slug precision/recall/F1 (already computed inside
    # _macro_f1_ht/_macro_f1 for the macro average, previously discarded) --
    # precision is the number `acc_<slug>` above can't give you: of the times
    # the model predicted this slug, how often was that actually right.
    for slug, (prec, rec, f1) in per_ht.items():
        out[f"precision_{slug}"] = prec
        out[f"recall_{slug}"] = rec
        out[f"f1_{slug}"] = f1
    for slug, (prec, rec, f1) in per_naive.items():
        out[f"precision_{slug}_naive_pooled"] = prec
        out[f"recall_{slug}_naive_pooled"] = rec
        out[f"f1_{slug}_naive_pooled"] = f1
    # One-vs-rest binary accuracy per slug: "is this article <slug> or not",
    # collapsing the other 9 possible labels into a single negative class --
    # the literal "this category vs. the total others" framing. Skews high
    # for rare slugs (dominated by true negatives), so read alongside
    # precision/recall above, not instead of them.
    for slug in classes:
        ovr_hit_f = [1.0 if (t == slug) == (p == slug) else 0.0 for t, p in pairs]
        out[f"accuracy_ovr_{slug}"] = _ht_ratio(it_ok, ovr_hit_f, ones)
        out[f"accuracy_ovr_{slug}_naive_pooled"] = _rate([bool(f) for f in ovr_hit_f])
    # One-vs-rest ROC AUC per class, from each row's own raw NLI score
    # (article_category's 9-way distribution) -- TASKS.md T-093, SPEC.md
    # FR-016. Iterates the fixed CATEGORY_SLUGS, NOT the dynamic `classes`
    # set the loops above use: "other" is a threshold fallback with no NLI
    # hypothesis/score column of its own (taxonomy.py), so roc_auc_other
    # isn't computable the way roc_auc_<slug> is for the 9 substantive
    # slugs -- it's simply never a key in this function's output.
    out.update(
        _roc_auc_per_class(
            it_ok,
            [t for t, _ in pairs],
            CATEGORY_SLUGS,
            lambda it, cls: float(it.prediction.get("distribution", {}).get(cls, 0.0)),
        )
    )
    return out


def aggregate_ner(items: Sequence[EvalItem], verdicts: Sequence[NerVerdict]) -> dict[str, float]:
    """Error-only judge contract: FP = judge-flagged ``wrong`` (clamped to the
    predicted count), FN = ``missed``, TP = predicted - FP. Per-type buckets use
    the model's own type for ``wrong`` and the judge's type for ``missed``."""
    ok = [(it, v) for it, v in zip(items, verdicts, strict=True) if not v.parse_failed]
    total = len(verdicts)
    out: dict[str, float] = {
        "n": float(len(ok)),
        "parse_fail_rate": (total - len(ok)) / total if total else 0.0,
    }
    if not ok:
        return out
    tp = fp = fn = 0
    predicted_total = 0
    per_type: dict[str, list[int]] = {}  # type -> [tp, fp, fn]
    for it, v in ok:
        preds = it.prediction.get("entities", [])
        pred_by_type = Counter(str(p.get("entity_type", "")) for p in preds)
        n_pred = len(preds)
        n_wrong = min(len(v.wrong), n_pred)
        predicted_total += n_pred
        tp += n_pred - n_wrong
        fp += n_wrong
        fn += len(v.missed)

        wrong_by_type = Counter(w.entity_type for w in v.wrong if w.entity_type)
        for etype, count in pred_by_type.items():
            slot = per_type.setdefault(etype, [0, 0, 0])
            w = min(wrong_by_type.get(etype, 0), count)
            slot[0] += count - w
            slot[1] += w
        for miss in v.missed:
            if miss.entity_type:
                per_type.setdefault(miss.entity_type, [0, 0, 0])[2] += 1

    micro_p, micro_r, micro_f1 = _prf(tp, fp, fn)
    out["micro_precision"] = micro_p
    out["micro_recall"] = micro_r
    out["micro_f1"] = micro_f1
    per_type_f1 = [_prf(t, f, n)[2] for t, f, n in per_type.values()]
    out["macro_f1"] = fmean(per_type_f1) if per_type_f1 else 0.0
    out["hallucination_rate"] = fp / (tp + fp) if (tp + fp) else 0.0
    out["miss_rate"] = fn / (tp + fn) if (tp + fn) else 0.0
    out["mean_entities_per_article"] = predicted_total / len(ok)
    for etype, (t, f, n) in sorted(per_type.items()):
        out[f"f1_{etype}"] = _prf(t, f, n)[2]
    return out


def aggregate_c_summary(
    items: Sequence[EvalItem], verdicts: Sequence[SummaryVerdict]
) -> dict[str, float]:
    ok = [(it, v) for it, v in zip(items, verdicts, strict=True) if not v.parse_failed]
    total = len(verdicts)
    out: dict[str, float] = {
        "n": float(len(ok)),
        "parse_fail_rate": (total - len(ok)) / total if total else 0.0,
    }
    if not ok:
        return out
    it_ok = [it for it, _ in ok]
    ones = [1.0] * len(it_ok)
    faith = [float(v.faithfulness) for _, v in ok]
    cov = [float(v.coverage) for _, v in ok]
    conc = [float(v.conciseness) for _, v in ok]
    halluc = [1.0 if v.hallucinations else 0.0 for _, v in ok]
    out["mean_faithfulness"] = _ht_ratio(it_ok, faith, ones)
    out["mean_faithfulness_naive_pooled"] = fmean(faith)
    out["mean_coverage"] = _ht_ratio(it_ok, cov, ones)
    out["mean_coverage_naive_pooled"] = fmean(cov)
    out["mean_conciseness"] = _ht_ratio(it_ok, conc, ones)
    out["mean_conciseness_naive_pooled"] = fmean(conc)
    out["pct_with_hallucination"] = _ht_ratio(it_ok, halluc, ones)
    out["pct_with_hallucination_naive_pooled"] = _rate([bool(h) for h in halluc])
    for bucket, rate in _per_bucket(it_ok, faith).items():
        out[f"mean_faithfulness_{bucket}"] = rate
    for bucket, rate in _per_bucket(it_ok, cov).items():
        out[f"mean_coverage_{bucket}"] = rate
    return out


def aggregate_sector_summary(
    items: Sequence[EvalItem], verdicts: Sequence[SectorIntroVerdict]
) -> dict[str, float]:
    """Full-population census, not a sample (T-054: 3,628 rows total is
    affordable to judge every run -- see sample_for_stage's sector_summary
    branch). Every item's ``stratum_population`` equals ``len(items)``, so
    the HT reweighting below is mathematically a no-op (weight 1 for every
    row); kept anyway for the same aggregator shape every other stage uses,
    rather than hand-rolling a plain mean as a special case."""
    ok = [(it, v) for it, v in zip(items, verdicts, strict=True) if not v.parse_failed]
    total = len(verdicts)
    out: dict[str, float] = {
        "n": float(len(ok)),
        "parse_fail_rate": (total - len(ok)) / total if total else 0.0,
    }
    if not ok:
        return out
    it_ok = [it for it, _ in ok]
    ones = [1.0] * len(it_ok)
    faith = [float(v.faithfulness) for _, v in ok]
    halluc = [1.0 if v.hallucinations else 0.0 for _, v in ok]
    out["mean_faithfulness"] = _ht_ratio(it_ok, faith, ones)
    out["mean_faithfulness_naive_pooled"] = fmean(faith)
    out["pct_with_hallucination"] = _ht_ratio(it_ok, halluc, ones)
    out["pct_with_hallucination_naive_pooled"] = _rate([bool(h) for h in halluc])
    return out


_Aggregator = Callable[[Sequence[EvalItem], Sequence[Any]], dict[str, float]]

_AGGREGATORS: dict[str, _Aggregator] = {
    "sentiment": aggregate_sentiment,
    "category": aggregate_category,
    "ner": aggregate_ner,
    "c_summary": aggregate_c_summary,
    "sector_summary": aggregate_sector_summary,
}


def aggregate(stage: str, items: Sequence[EvalItem], verdicts: Sequence[Any]) -> dict[str, float]:
    """Dispatch to the stage's aggregator."""
    try:
        fn = _AGGREGATORS[stage]
    except KeyError:
        raise ValueError(f"unknown stage {stage!r}") from None
    return fn(items, verdicts)


def confusion_pairs(
    stage: str, items: Sequence[EvalItem], verdicts: Sequence[Any]
) -> list[tuple[str, str]] | None:
    """``(true_label, predicted_label)`` pairs, ``parse_failed`` rows
    excluded -- sentiment/category only (the only two stages with a
    discrete predicted/ideal label shape); ``None`` for every other stage
    (TASKS.md T-092, SPEC.md FR-016). Mirrors `aggregate_sentiment`/
    `aggregate_category`'s own ``pairs``/``parse_failed``-filter derivation
    exactly -- kept separate rather than shared, since refactoring those
    two tested aggregate functions to reuse an extraction helper isn't
    worth it for a two-line derivation."""
    if stage == "sentiment":
        return [
            (v.ideal_label, str(it.prediction.get("label", "neutral")))
            for it, v in zip(items, verdicts, strict=True)
            if not v.parse_failed
        ]
    if stage == "category":
        return [
            (v.ideal_slug, str(it.prediction.get("label", "other")))
            for it, v in zip(items, verdicts, strict=True)
            if not v.parse_failed
        ]
    return None
