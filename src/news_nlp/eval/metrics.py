"""Pure aggregation of per-row judge verdicts into flat metric dicts.

Every ``aggregate_*`` returns ``dict[str, float]`` ready for
``mlflow.log_metrics``. Rows the judge could not parse (``parse_failed``) are
excluded from the accuracy numbers and counted in ``parse_fail_rate``. The judge
is treated as ground truth, so "F1 vs judge" / "accuracy vs judge" are
agreement measures, not truth measures -- see ``docs/evaluation.md``.
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
    SentimentVerdict,
    SummaryVerdict,
)

#: The single metric ``regression.check_regression`` compares between runs.
#:
#: ``sentiment`` is ``recall_negative``, not the balanced ``macro_f1_vs_judge``
#: (still computed and logged, just not the gate): missing a real
#: negative-sentiment article costs more here than over-flagging a neutral one
#: as negative -- negative sentiment is the signal portfolio construction
#: leans on, so false negatives are the regression that matters. See
#: docs/evaluation.md's "Why recall, not F1, for sentiment negative" note.
HEADLINE: dict[str, str] = {
    "sentiment": "recall_negative",
    "category": "accuracy_vs_judge",
    "ner": "micro_f1",
    "c_summary": "mean_faithfulness",
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


def _split(items: Sequence[EvalItem], flags: Sequence[bool], bucket: str) -> float:
    picked = [f for it, f in zip(items, flags, strict=True) if it.bucket == bucket]
    return _rate(picked)


def aggregate_sentiment(
    items: Sequence[EvalItem], verdicts: Sequence[SentimentVerdict]
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
    agree = [v.agrees for _, v in ok]
    pairs = [(v.ideal_label, str(it.prediction.get("label", "neutral"))) for it, v in ok]
    macro, per = _macro_f1(pairs, _SENTIMENT_CLASSES)
    out["agreement_rate"] = _rate(agree)
    out["agreement_rate_low_conf"] = _split(it_ok, agree, "low_conf")
    out["agreement_rate_random"] = _split(it_ok, agree, "random")
    out["macro_f1_vs_judge"] = macro
    out["mean_severity"] = fmean(v.severity for _, v in ok)
    for cls, (prec, rec, f1) in per.items():
        out[f"precision_{cls}"] = prec
        out[f"recall_{cls}"] = rec
        out[f"f1_{cls}"] = f1
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
    pairs = [(v.ideal_slug, str(it.prediction.get("label", "other"))) for it, v in ok]
    classes = {t for t, _ in pairs} | {p for _, p in pairs}
    macro, _ = _macro_f1(pairs, sorted(classes))
    out["accuracy_vs_judge"] = _rate(correct)
    out["accuracy_vs_judge_low_conf"] = _split(it_ok, correct, "low_conf")
    out["accuracy_vs_judge_random"] = _split(it_ok, correct, "random")
    out["macro_f1"] = macro
    out["other_rate_model"] = _rate([str(it.prediction.get("label")) == "other" for it, _ in ok])
    out["other_rate_judge"] = _rate([v.ideal_slug == "other" for _, v in ok])
    out["mean_severity"] = fmean(v.severity for _, v in ok)
    # per-judge-slug accuracy (recall of the model on that slug)
    for slug in sorted({v.ideal_slug for _, v in ok}):
        hits = [c for (it, v), c in zip(ok, correct, strict=True) if v.ideal_slug == slug]
        out[f"acc_{slug}"] = _rate(hits)
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
    out["mean_faithfulness"] = fmean(v.faithfulness for _, v in ok)
    out["mean_coverage"] = fmean(v.coverage for _, v in ok)
    out["mean_conciseness"] = fmean(v.conciseness for _, v in ok)
    out["pct_with_hallucination"] = _rate([bool(v.hallucinations) for _, v in ok])
    faith_flags = [v.faithfulness for _, v in ok]
    lc = [f for it, f in zip(it_ok, faith_flags, strict=True) if it.bucket == "low_conf"]
    rnd = [f for it, f in zip(it_ok, faith_flags, strict=True) if it.bucket == "random"]
    out["mean_faithfulness_low_conf"] = fmean(lc) if lc else 0.0
    out["mean_faithfulness_random"] = fmean(rnd) if rnd else 0.0
    return out


_Aggregator = Callable[[Sequence[EvalItem], Sequence[Any]], dict[str, float]]

_AGGREGATORS: dict[str, _Aggregator] = {
    "sentiment": aggregate_sentiment,
    "category": aggregate_category,
    "ner": aggregate_ner,
    "c_summary": aggregate_c_summary,
}


def aggregate(stage: str, items: Sequence[EvalItem], verdicts: Sequence[Any]) -> dict[str, float]:
    """Dispatch to the stage's aggregator."""
    try:
        fn = _AGGREGATORS[stage]
    except KeyError:
        raise ValueError(f"unknown stage {stage!r}") from None
    return fn(items, verdicts)
