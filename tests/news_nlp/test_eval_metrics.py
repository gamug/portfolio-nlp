"""Pure aggregation maths for news_nlp.eval.metrics -- no DB, no LLM."""

from __future__ import annotations

import pytest

from news_nlp.eval import metrics
from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.verdicts import (
    CategoryVerdict,
    EntityRef,
    NerVerdict,
    SentimentVerdict,
    SummaryVerdict,
)


def _item(article_id: int, bucket: str, prediction: dict) -> EvalItem:
    return EvalItem(
        article_id=article_id,
        bucket=bucket,
        title="t",
        body_text="b",
        prediction=prediction,
    )


def test_sentiment_agreement_and_macro_f1() -> None:
    items = [
        _item(1, "low_conf", {"label": "positive"}),
        _item(2, "low_conf", {"label": "negative"}),
        _item(3, "random", {"label": "neutral"}),
        _item(4, "random", {"label": "positive"}),
    ]
    verdicts = [
        SentimentVerdict(agrees=True, ideal_label="positive", severity=0),
        SentimentVerdict(agrees=False, ideal_label="positive", severity=2),  # model said negative
        SentimentVerdict(agrees=True, ideal_label="neutral", severity=0),
        SentimentVerdict(agrees=True, ideal_label="positive", severity=0),
    ]
    out = metrics.aggregate_sentiment(items, verdicts)

    assert out["n"] == 4.0
    assert out["parse_fail_rate"] == 0.0
    assert out["agreement_rate"] == 0.75
    assert out["agreement_rate_low_conf"] == 0.5
    assert out["agreement_rate_random"] == 1.0
    # judge truth: pos,pos,neu,pos ; model pred: pos,neg,neu,pos
    # positive: tp=2 fp=0 fn=1 -> P=1 R=.667 F1=.8
    # negative: tp=0 fp=1 fn=0 -> F1=0
    # neutral:  tp=1 fp=0 fn=0 -> F1=1
    assert round(out["f1_positive"], 3) == 0.8
    assert out["f1_negative"] == 0.0
    assert out["f1_neutral"] == 1.0
    assert round(out["macro_f1_vs_judge"], 3) == round((0.8 + 0.0 + 1.0) / 3, 3)


def test_sentiment_recall_negative_is_the_headline_metric() -> None:
    """sentiment's HEADLINE is recall_negative (metrics.HEADLINE), not F1: a
    missed real negative costs more here than an over-flagged neutral, so the
    metric that matters is "how many true negatives did the model catch," not
    a precision/recall balance. This locks in that recall_negative stays
    perfect even when over-flagging tanks precision (and drags F1 down with
    it) -- exactly the trade-off the headline choice is meant to reward."""
    items = [
        _item(1, "low_conf", {"label": "negative"}),  # true negative, caught
        _item(2, "low_conf", {"label": "negative"}),  # true negative, caught
        _item(3, "random", {"label": "negative"}),  # true neutral, over-flagged
        _item(4, "random", {"label": "negative"}),  # true neutral, over-flagged
        _item(5, "random", {"label": "negative"}),  # true positive, over-flagged
    ]
    verdicts = [
        SentimentVerdict(agrees=True, ideal_label="negative", severity=0),
        SentimentVerdict(agrees=True, ideal_label="negative", severity=0),
        SentimentVerdict(agrees=False, ideal_label="neutral", severity=1),
        SentimentVerdict(agrees=False, ideal_label="neutral", severity=1),
        SentimentVerdict(agrees=False, ideal_label="positive", severity=2),
    ]
    out = metrics.aggregate_sentiment(items, verdicts)

    # negative: tp=2 (articles 1,2) fp=3 (articles 3,4,5) fn=0 -> recall=1.0,
    # but precision=0.4 and f1~0.571 -- F1 would read this as mediocre.
    assert out["recall_negative"] == 1.0
    assert round(out["precision_negative"], 3) == 0.4
    assert round(out["f1_negative"], 3) == round(2 * 1.0 * 0.4 / (1.0 + 0.4), 3)
    assert metrics.HEADLINE["sentiment"] == "recall_negative"


def test_sentiment_excludes_parse_failures() -> None:
    items = [_item(1, "random", {"label": "positive"}), _item(2, "random", {"label": "negative"})]
    verdicts = [
        SentimentVerdict(agrees=True, ideal_label="positive"),
        SentimentVerdict(agrees=False, ideal_label="neutral", parse_failed=True),
    ]
    out = metrics.aggregate_sentiment(items, verdicts)
    assert out["n"] == 1.0
    assert out["parse_fail_rate"] == 0.5
    assert out["agreement_rate"] == 1.0


def test_category_accuracy_and_other_rates() -> None:
    items = [
        _item(1, "low_conf", {"label": "earnings_performance"}),
        _item(2, "low_conf", {"label": "other"}),
        _item(3, "random", {"label": "legal_regulatory"}),
    ]
    verdicts = [
        CategoryVerdict(agrees=True, ideal_slug="earnings_performance"),
        CategoryVerdict(agrees=False, ideal_slug="mergers_acquisitions"),
        CategoryVerdict(agrees=True, ideal_slug="legal_regulatory"),
    ]
    out = metrics.aggregate_category(items, verdicts)
    assert round(out["accuracy_vs_judge"], 4) == round(2 / 3, 4)
    assert out["accuracy_vs_judge_low_conf"] == 0.5
    assert out["accuracy_vs_judge_random"] == 1.0
    assert round(out["other_rate_model"], 4) == round(1 / 3, 4)
    assert out["other_rate_judge"] == 0.0


def test_category_coerces_unknown_ideal_slug_to_other() -> None:
    v = CategoryVerdict(agrees=False, ideal_slug="not_a_real_slug")
    assert v.ideal_slug == "other"


def test_ner_error_only_contract_prf() -> None:
    items = [
        _item(
            1,
            "random",
            {
                "entities": [
                    {"entity_type": "ORG", "text": "Acme"},
                    {"entity_type": "ORG", "text": "the"},
                    {"entity_type": "PER", "text": "Bob"},
                ]
            },
        ),
        _item(
            2,
            "random",
            {
                "entities": [
                    {"entity_type": "PER", "text": "Sue"},
                    {"entity_type": "ORG", "text": "IBM"},
                ]
            },
        ),
    ]
    verdicts = [
        NerVerdict(
            wrong=[EntityRef(text="the", entity_type="ORG")],
            missed=[EntityRef(text="Berlin", entity_type="LOC")],
        ),
        NerVerdict(wrong=[], missed=[]),
    ]
    out = metrics.aggregate_ner(items, verdicts)
    # predicted 5, wrong 1 -> tp=4 fp=1 fn=1
    assert out["micro_precision"] == pytest.approx(0.8)
    assert out["micro_recall"] == pytest.approx(0.8)
    assert out["micro_f1"] == pytest.approx(0.8)
    assert out["hallucination_rate"] == pytest.approx(0.2)
    assert out["miss_rate"] == pytest.approx(0.2)
    assert out["mean_entities_per_article"] == 2.5
    # ORG: tp=2 fp=1 -> F1 .8 ; PER: tp=2 -> 1.0 ; LOC: fn=1 -> 0.0
    assert out["f1_ORG"] == pytest.approx(0.8)
    assert out["f1_PER"] == 1.0
    assert out["f1_LOC"] == 0.0
    assert out["macro_f1"] == pytest.approx(0.6)


def test_c_summary_scales_and_hallucination_flag() -> None:
    items = [_item(1, "low_conf", {}), _item(2, "random", {})]
    verdicts = [
        SummaryVerdict(
            faithfulness=2, coverage=3, conciseness=4, hallucinations=["made up a number"]
        ),
        SummaryVerdict(faithfulness=5, coverage=5, conciseness=5, hallucinations=[]),
    ]
    out = metrics.aggregate_c_summary(items, verdicts)
    assert out["mean_faithfulness"] == 3.5
    assert out["pct_with_hallucination"] == 0.5
    assert out["mean_faithfulness_low_conf"] == 2.0
    assert out["mean_faithfulness_random"] == 5.0


def test_aggregate_dispatch_and_empty() -> None:
    assert metrics.aggregate("sentiment", [], []) == {"n": 0.0, "parse_fail_rate": 0.0}
    assert metrics.HEADLINE["ner"] == "micro_f1"
    with pytest.raises(ValueError, match="unknown stage"):
        metrics.aggregate("bogus", [], [])
