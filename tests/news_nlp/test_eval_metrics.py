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


def _item(
    article_id: int, bucket: str, prediction: dict, *, stratum_population: int = 1
) -> EvalItem:
    return EvalItem(
        article_id=article_id,
        bucket=bucket,
        stratum_population=stratum_population,
        title="t",
        body_text="b",
        prediction=prediction,
    )


def test_sentiment_agreement_and_macro_f1() -> None:
    """Uses a single non-excluded bucket ("representative") so the
    HT-weighted headline metrics reduce to the plain pooled numbers -- see
    test_aggregate_sentiment_recall_negative_uses_ht_estimator for a
    multi-stratum example where they genuinely differ. low_conf is included
    here only to exercise its exclusion from the *_naive_pooled companions,
    which still pool everything (unchanged _macro_f1/_rate)."""
    items = [
        _item(1, "low_conf", {"label": "positive"}),
        _item(2, "low_conf", {"label": "negative"}),
        _item(3, "representative", {"label": "neutral"}),
        _item(4, "representative", {"label": "positive"}),
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
    assert out["agreement_rate_representative"] == 1.0
    # naive pooled (all 4 items): judge truth pos,pos,neu,pos ; model pred pos,neg,neu,pos
    # positive: tp=2 fp=0 fn=1 -> P=1 R=.667 F1=.8 ; negative: tp=0 fp=1 fn=0 -> F1=0 ; neutral: F1=1
    assert round(out["f1_positive_naive_pooled"], 3) == 0.8
    assert out["f1_negative_naive_pooled"] == 0.0
    assert out["f1_neutral_naive_pooled"] == 1.0
    assert round(out["macro_f1_vs_judge_naive_pooled"], 3) == round((0.8 + 0.0 + 1.0) / 3, 3)
    # HT-weighted (excludes low_conf; single non-excluded bucket -> reduces to
    # that bucket's own pooled numbers): only articles 3,4, both correct.
    assert out["f1_positive"] == 1.0
    assert out["f1_negative"] == 0.0
    assert out["f1_neutral"] == 1.0
    assert round(out["macro_f1_vs_judge"], 3) == round((1.0 + 0.0 + 1.0) / 3, 3)


def test_sentiment_recall_negative_is_the_headline_metric() -> None:
    """sentiment's HEADLINE is recall_negative (metrics.HEADLINE), not F1: a
    missed real negative costs more here than an over-flagged neutral, so the
    metric that matters is "how many true negatives did the model catch," not
    a precision/recall balance. This locks in that recall_negative stays
    perfect even when over-flagging tanks precision (and drags F1 down with
    it) -- exactly the trade-off the headline choice is meant to reward.
    Single bucket ("representative") throughout -- HT reduces to naive pooled
    here, so it isolates the recall-vs-F1 property from the HT reweighting
    property (covered separately below)."""
    items = [
        _item(1, "representative", {"label": "negative"}),  # true negative, caught
        _item(2, "representative", {"label": "negative"}),  # true negative, caught
        _item(3, "representative", {"label": "negative"}),  # true neutral, over-flagged
        _item(4, "representative", {"label": "negative"}),  # true neutral, over-flagged
        _item(5, "representative", {"label": "negative"}),  # true positive, over-flagged
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


def test_aggregate_sentiment_recall_negative_uses_ht_estimator() -> None:
    """A genuinely multi-stratum sample: recall_negative (HT-weighted) must
    differ from recall_negative_naive_pooled once strata are weighted
    unevenly -- proves the Horvitz-Thompson reweighting actually engages
    rather than silently degenerating to a flat pooled count. Matches the
    worked example in docs/evaluation.md: target_negative N=100,n=4 (TP=2,
    FN=1); representative N=900,n=6 (TP=1)."""
    items = [
        _item(1, "target_negative", {"label": "negative"}, stratum_population=100),  # TP
        _item(2, "target_negative", {"label": "negative"}, stratum_population=100),  # TP
        _item(3, "target_negative", {"label": "positive"}, stratum_population=100),  # FN
        _item(4, "target_negative", {"label": "positive"}, stratum_population=100),  # off-class
        _item(5, "representative", {"label": "negative"}, stratum_population=900),  # TP
        _item(6, "representative", {"label": "positive"}, stratum_population=900),  # off-class
        _item(7, "representative", {"label": "positive"}, stratum_population=900),  # off-class
        _item(8, "representative", {"label": "positive"}, stratum_population=900),  # off-class
        _item(9, "representative", {"label": "positive"}, stratum_population=900),  # off-class
        _item(10, "representative", {"label": "positive"}, stratum_population=900),  # off-class
    ]
    verdicts = [
        SentimentVerdict(agrees=True, ideal_label="negative", severity=0),
        SentimentVerdict(agrees=True, ideal_label="negative", severity=0),
        SentimentVerdict(agrees=False, ideal_label="negative", severity=2),
        SentimentVerdict(agrees=True, ideal_label="positive", severity=0),
        SentimentVerdict(agrees=True, ideal_label="negative", severity=0),
        SentimentVerdict(agrees=True, ideal_label="positive", severity=0),
        SentimentVerdict(agrees=True, ideal_label="positive", severity=0),
        SentimentVerdict(agrees=True, ideal_label="positive", severity=0),
        SentimentVerdict(agrees=True, ideal_label="positive", severity=0),
        SentimentVerdict(agrees=True, ideal_label="positive", severity=0),
    ]
    out = metrics.aggregate_sentiment(items, verdicts)

    # TP_hat = (100/4)*2 + (900/6)*1 = 50+150=200 ; FN_hat = (100/4)*1 + (900/6)*0 = 25
    assert round(out["recall_negative"], 4) == round(200 / 225, 4)
    # naive pooled: tp=3 (articles 1,2,5) fn=1 (article 3) -> 3/4
    assert out["recall_negative_naive_pooled"] == 0.75
    assert out["recall_negative"] != out["recall_negative_naive_pooled"]


def test_sentiment_excludes_parse_failures() -> None:
    items = [
        _item(1, "representative", {"label": "positive"}),
        _item(2, "representative", {"label": "negative"}),
    ]
    verdicts = [
        SentimentVerdict(agrees=True, ideal_label="positive"),
        SentimentVerdict(agrees=False, ideal_label="neutral", parse_failed=True),
    ]
    out = metrics.aggregate_sentiment(items, verdicts)
    assert out["n"] == 1.0
    assert out["parse_fail_rate"] == 0.5
    assert out["agreement_rate"] == 1.0


def test_ht_sum_and_ht_ratio_hand_computed_example() -> None:
    """The worked stratified-estimator example from docs/evaluation.md:
    stratum A N=100,n=4 (TP=2,FN=1); stratum B N=900,n=6 (TP=1,FN=0)."""
    items = [_item(i, "target_negative", {}, stratum_population=100) for i in range(1, 5)] + [
        _item(i, "representative", {}, stratum_population=900) for i in range(5, 11)
    ]
    tp = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fn = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    denom = [t + f for t, f in zip(tp, fn, strict=True)]
    recall_hat = metrics._ht_ratio(items, tp, denom)
    assert round(recall_hat, 4) == round(200 / 225, 4)
    naive = sum(tp) / (sum(tp) + sum(fn))
    assert round(naive, 4) == 0.75


def test_ht_ratio_excludes_low_conf_bucket() -> None:
    items = [
        _item(1, "low_conf", {}, stratum_population=50),  # would skew the result if included
        _item(2, "representative", {}, stratum_population=100),
    ]
    tp = [0.0, 1.0]
    fn = [1.0, 0.0]
    denom = [t + f for t, f in zip(tp, fn, strict=True)]
    assert metrics._ht_ratio(items, tp, denom) == 1.0  # only the representative item counts


def test_ht_ratio_zero_survivors_stratum_contributes_nothing() -> None:
    """A stratum with no surviving (post-parse-failure) rows contributes
    nothing, never a ZeroDivisionError -- aggregate_* already filters
    parse_failed rows out of `ok` before this is ever called, so a stratum
    with 100% parse failures is simply absent from `items` entirely."""
    items = [_item(1, "representative", {}, stratum_population=100)]
    assert metrics._ht_ratio(items, [1.0], [1.0]) == 1.0
    assert metrics._ht_ratio([], [], []) == 0.0


def test_category_accuracy_and_other_rates() -> None:
    """accuracy_vs_judge/other_rate_* are HT-weighted, excluding low_conf --
    with a single non-excluded bucket ("representative", 1 item), they reduce
    to that item's own value; *_naive_pooled preserves the old combined
    numbers. The per-bucket accuracy_vs_judge_<bucket> diagnostics are
    unaffected either way (they only ever describe their own bucket)."""
    items = [
        _item(1, "low_conf", {"label": "earnings_performance"}),
        _item(2, "low_conf", {"label": "other"}),
        _item(3, "representative", {"label": "legal_regulatory"}),
    ]
    verdicts = [
        CategoryVerdict(agrees=True, ideal_slug="earnings_performance"),
        CategoryVerdict(agrees=False, ideal_slug="mergers_acquisitions"),
        CategoryVerdict(agrees=True, ideal_slug="legal_regulatory"),
    ]
    out = metrics.aggregate_category(items, verdicts)
    assert out["accuracy_vs_judge"] == 1.0
    assert round(out["accuracy_vs_judge_naive_pooled"], 4) == round(2 / 3, 4)
    assert out["accuracy_vs_judge_low_conf"] == 0.5
    assert out["accuracy_vs_judge_representative"] == 1.0
    assert out["other_rate_model"] == 0.0  # only the representative item counts; not "other"
    assert round(out["other_rate_model_naive_pooled"], 4) == round(1 / 3, 4)
    assert out["other_rate_judge"] == 0.0
    assert out["other_rate_judge_naive_pooled"] == 0.0


def test_aggregate_category_accuracy_uses_ht_estimator() -> None:
    """A genuinely multi-stratum sample: accuracy_vs_judge (HT-weighted) must
    differ from its naive_pooled companion once strata are weighted
    unevenly."""
    items = [
        _item(
            1,
            "target_partnerships_business_dev",
            {"label": "partnerships_business_dev"},
            stratum_population=50,
        ),
        _item(
            2,
            "target_partnerships_business_dev",
            {"label": "partnerships_business_dev"},
            stratum_population=50,
        ),
        _item(3, "representative", {"label": "other"}, stratum_population=500),
    ]
    verdicts = [
        CategoryVerdict(agrees=True, ideal_slug="partnerships_business_dev"),
        CategoryVerdict(agrees=False, ideal_slug="other"),
        CategoryVerdict(agrees=True, ideal_slug="other"),
    ]
    out = metrics.aggregate_category(items, verdicts)
    # correct_hat = (50/2)*1 + (500/1)*1 = 25+500=525 ; n_hat = (50/2)*2 + (500/1)*1 = 550
    assert round(out["accuracy_vs_judge"], 4) == round(525 / 550, 4)
    assert round(out["accuracy_vs_judge_naive_pooled"], 4) == round(2 / 3, 4)
    assert out["accuracy_vs_judge"] != out["accuracy_vs_judge_naive_pooled"]


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
    """mean_faithfulness/pct_with_hallucination are HT-weighted, excluding
    low_conf -- single non-excluded bucket ("representative", 1 item) reduces
    to that item's own value; *_naive_pooled preserves the old combined
    numbers. Per-bucket mean_faithfulness_<bucket> is unaffected either way."""
    items = [_item(1, "low_conf", {}), _item(2, "representative", {})]
    verdicts = [
        SummaryVerdict(
            faithfulness=2, coverage=3, conciseness=4, hallucinations=["made up a number"]
        ),
        SummaryVerdict(faithfulness=5, coverage=5, conciseness=5, hallucinations=[]),
    ]
    out = metrics.aggregate_c_summary(items, verdicts)
    assert out["mean_faithfulness"] == 5.0
    assert out["mean_faithfulness_naive_pooled"] == 3.5
    assert out["pct_with_hallucination"] == 0.0
    assert out["pct_with_hallucination_naive_pooled"] == 0.5
    assert out["mean_faithfulness_low_conf"] == 2.0
    assert out["mean_faithfulness_representative"] == 5.0


def test_aggregate_c_summary_coverage_uses_ht_estimator() -> None:
    """A genuinely multi-stratum sample: mean_coverage (HT-weighted) must
    differ from its naive_pooled companion once strata are weighted
    unevenly."""
    items = [
        _item(1, "target_chunks_ge3", {}, stratum_population=20),
        _item(2, "target_chunks_ge3", {}, stratum_population=20),
        _item(3, "representative", {}, stratum_population=200),
    ]
    verdicts = [
        SummaryVerdict(faithfulness=5, coverage=2, conciseness=5, hallucinations=[]),
        SummaryVerdict(faithfulness=5, coverage=2, conciseness=5, hallucinations=[]),
        SummaryVerdict(faithfulness=5, coverage=5, conciseness=5, hallucinations=[]),
    ]
    out = metrics.aggregate_c_summary(items, verdicts)
    # cov_hat_num = (20/2)*(2+2) + (200/1)*5 = 40+1000=1040 ; cov_hat_den = (20/2)*2 + (200/1)*1 = 220
    assert round(out["mean_coverage"], 4) == round(1040 / 220, 4)
    assert round(out["mean_coverage_naive_pooled"], 4) == round((2 + 2 + 5) / 3, 4)
    assert out["mean_coverage"] != out["mean_coverage_naive_pooled"]


def test_aggregate_dispatch_and_empty() -> None:
    assert metrics.aggregate("sentiment", [], []) == {"n": 0.0, "parse_fail_rate": 0.0}
    assert metrics.HEADLINE["ner"] == "micro_f1"
    with pytest.raises(ValueError, match="unknown stage"):
        metrics.aggregate("bogus", [], [])
