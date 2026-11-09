"""The report of a reading of the validation slice (T-148c3): precision, recall and ROC OVR (PLAN D37), weighted,
with the raw description of the 83."""

from __future__ import annotations

import math

import pytest

import sentiment_pair_report as rep
from news_nlp import gold_metrics


def test_the_confusion_matrix_and_the_class_figures_count_what_was_read_as_what() -> None:
    expected = ["negative"] * 4 + ["positive"] * 3 + ["neutral"] * 3
    predicted = (
        ["negative", "negative", "neutral", "positive"]
        + ["positive"] * 3
        + ["negative"]
        + ["neutral"] * 2
    )
    table = rep.confusion(expected, predicted)
    assert table["negative"] == {"positive": 1, "negative": 2, "neutral": 1}
    assert table["neutral"] == {"positive": 0, "negative": 1, "neutral": 2}
    negative = rep.class_figures(table)["negative"]
    assert negative["precision"].point == pytest.approx(2 / 3)  # 3 read as negative, 2 of them are
    assert negative["recall"].point == pytest.approx(2 / 4)
    assert negative["recall"].low < 0.5 < negative["recall"].high
    lines = rep.report_lines("x", expected, predicted)
    assert lines[0] == "x, RAW on the 10 texts: agreement 7/10" and any(
        "negative: precision" in line for line in lines
    )


def test_the_exact_mcnemar_p_value_and_the_paired_report() -> None:
    assert rep.mcnemar_exact(0, 0) == 1.0 and rep.mcnemar_exact(3, 3) == 1.0
    assert rep.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert rep.mcnemar_exact(11, 0) == pytest.approx(2 / 2**11)
    a = [True] * 12 + [False] * 8
    b = (
        [True] * 2 + [False] * 10 + [True] * 2 + [False] * 6
    )  # 2 shared, 10 only a, 2 only b, 6 neither
    report = rep.paired_report(a, b)
    assert (report.both, report.only_a, report.only_b, report.neither) == (2, 10, 2, 6)
    assert report.p_value == pytest.approx(rep.mcnemar_exact(10, 2)) and report.p_value < 0.05
    assert report.gain.point == pytest.approx(0.4)
    assert not hasattr(report, "verdict")  # the pin rule is the weighted gain, not this one
    same = rep.paired_report(a, a)
    assert same.p_value == 1.0 and same.gain.point == 0.0


def _scored(reading: str) -> tuple[list[rep.Scored], dict[str, int]]:
    """Two strata (the class the pinned model predicted), the small one a pool of 100 and the big one of 900,
    10 texts each. *reading* 'good' gives the owner's class the top probability, 'bad' gives it the lowest."""
    out = []
    for stratum, count in (("negative", 10), ("positive", 10)):
        for i in range(count):
            gold = (
                ("negative" if i < 6 else "positive")
                if stratum == "negative"
                else (
                    (
                        "positive",
                        "positive",
                        "positive",
                        "positive",
                        "neutral",
                        "neutral",
                        "negative",
                        "positive",
                        "positive",
                        "neutral",
                    )[i]
                )
            )
            good = {c: 0.1 for c in rep.CLASSES} | {gold: 0.8}
            bad = {c: 0.8 for c in rep.CLASSES} | {gold: 0.1}
            probs = good if reading == "good" else bad
            predicted = max(probs, key=lambda c: probs[c])
            out.append(rep.Scored(f"{stratum[0]}{i}", stratum, gold, predicted, probs))
    return out, {"negative": 100, "positive": 900}


def test_the_weighted_roc_auc_of_a_class_against_the_rest() -> None:
    assert rep.weighted_roc_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False], [1] * 4) == 1.0
    assert rep.weighted_roc_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False], [1] * 4) == 0.0
    assert rep.weighted_roc_auc([0.5, 0.5], [True, False], [1, 1]) == 0.5  # a tie counts half
    # weights: the heavy negative sits above the positive, the light one below: 1 / (1 + 3) of the weight is won
    assert rep.weighted_roc_auc([0.5, 0.9, 0.1], [True, False, False], [1, 3, 1]) == pytest.approx(
        1 / 4
    )
    assert math.isnan(rep.weighted_roc_auc([0.5, 0.6], [True, True], [1, 1]))  # no negative text
    assert math.isnan(rep.weighted_roc_auc([0.5, 0.6], [False, False], [1, 1]))  # no positive text


def test_precision_recall_and_roc_ovr_of_each_class_weighted_by_the_strata() -> None:
    scored, populations = _scored("good")
    weights = rep.row_weights(scored, populations)
    assert weights[0] == 10.0 and weights[-1] == 90.0  # pool / drawn
    metrics = rep.class_metrics(scored, weights)
    for c in rep.CLASSES:
        assert metrics[c] == {"precision": 1.0, "recall": 1.0, "roc_ovr": 1.0}
    assert rep.selection_score(metrics) == 1.0
    bad_scored, _ = _scored("bad")
    bad = rep.class_metrics(bad_scored, rep.row_weights(bad_scored, populations))
    assert bad["negative"]["roc_ovr"] == 0.0  # the owner's class always has the lowest probability
    # the owner's class never has the top probability: no class is ever the right answer
    assert all(bad[c]["recall"] == 0.0 and bad[c]["roc_ovr"] == 0.0 for c in rep.CLASSES)
    assert rep.selection_score(bad) < 0.1


def test_an_undefined_figure_counts_as_zero_in_the_macro_and_the_selection_score() -> None:
    metrics = {
        "positive": {"precision": 0.9, "recall": 0.6, "roc_ovr": 0.9},
        "negative": {"precision": math.nan, "recall": 0.0, "roc_ovr": 0.3},
        "neutral": {"precision": 0.3, "recall": 0.3, "roc_ovr": 0.6},
    }
    macro = rep.macro(metrics)
    assert macro == {
        "precision": pytest.approx((0.9 + 0 + 0.3) / 3),
        "recall": pytest.approx((0.6 + 0 + 0.3) / 3),
        "roc_ovr": pytest.approx((0.9 + 0.3 + 0.6) / 3),
    }
    assert rep.selection_score(metrics) == pytest.approx(sum(macro.values()) / 3)
    detail = rep.detail_of(metrics)
    assert detail["negative"]["precision"] is None and detail["macro"][
        "precision"
    ] == pytest.approx(0.4)
    assert set(detail) == {"positive", "negative", "neutral", "macro"}


def test_the_weighted_figures_have_an_interval_that_holds_the_point_and_do_not_change_with_the_seed() -> (
    None
):
    scored, populations = _scored("good")
    figures = rep.weighted_figures(scored, populations, resamples=60)
    assert figures["score"].point == 1.0 and (figures["score"].low, figures["score"].high) == (
        1.0,
        1.0,
    )
    assert set(figures) == {"score", *(f"macro {m}" for m in rep.CLASS_METRICS)} | {
        f"{c} {m}" for c in rep.CLASSES for m in rep.CLASS_METRICS
    }
    mixed = (
        [s if i % 4 else s._replace(predicted="neutral") for i, s in enumerate(scored)]
        if False
        else [
            rep.Scored(s.key, s.stratum, s.gold, "neutral" if i % 4 == 0 else s.predicted, s.probs)
            for i, s in enumerate(scored)
        ]
    )
    a = rep.weighted_figures(mixed, populations, resamples=60)
    b = rep.weighted_figures(mixed, populations, resamples=60)
    assert a == b  # seeded
    assert (
        a["score"].low <= a["score"].point <= a["score"].high and a["score"].low < a["score"].high
    )
    assert rep.weighted_figures(mixed, populations, resamples=60, seed=7) != a


def test_the_weighted_gain_is_paired_and_comes_with_the_gold_verdict() -> None:
    good, populations = _scored("good")
    bad, _ = _scored("bad")
    gains = rep.weighted_gains(good, bad, populations, resamples=60)
    gain, verdict = gains["score"]
    assert gain.point > 0.5 and verdict == "better"
    gain, verdict = gains["negative roc_ovr"]
    assert gain.point == 1.0 and verdict == "better"
    same = rep.weighted_gains(good, good, populations, resamples=60)
    assert all(g.point == 0.0 and v == "not decided" for g, v in same.values())
    with pytest.raises(gold_metrics.MarkedFileError, match="not of the same texts"):
        rep.weighted_gains(good, [*bad[:-1], bad[0]], populations, resamples=5)
    with pytest.raises(gold_metrics.MarkedFileError, match="no population for the strata"):
        rep.row_weights(good, {"negative": 1})


def test_d32_verdicts_are_the_bars_judge_of_the_negative_class_and_roc_has_no_bar() -> None:
    figures = {
        "negative precision": gold_metrics.Figure(0.46, 0.278, 0.652),  # wholly below 70%
        "negative recall": gold_metrics.Figure(0.892, 0.719, 1.0),  # wholly at or above 70%
        "negative roc_ovr": gold_metrics.Figure(0.5, 0.4, 0.6),
    }
    assert rep.bar_verdicts(figures) == {"negative precision": "fail", "negative recall": "pass"}
    figures["negative recall"] = gold_metrics.Figure(0.8, 0.6, 0.95)  # straddles the bar
    assert rep.bar_verdicts(figures)["negative recall"] == "not decided"


def test_the_kept_predictions_become_scored_texts_with_their_strata_and_pools() -> None:
    marks = [
        {"key": "a", "slice": "validation", "predicted": "negative", "pool_class_size": "100"},
        {"key": "b", "slice": "validation", "predicted": "positive", "pool_class_size": "900"},
        {"key": "c", "slice": "train", "predicted": "positive", "pool_class_size": "900"},
    ]
    predictions = [
        {
            "key": k,
            "expected": "negative",
            "predicted": "positive",
            "p_positive": "0.7",
            "p_negative": "0.2",
            "p_neutral": "0.1",
        }
        for k in ("a", "b")
    ]
    scored, populations = rep.scored_from_rows(predictions, marks)
    assert populations == {"negative": 100, "positive": 900}
    assert scored[0] == rep.Scored(
        "a", "negative", "negative", "positive", {"positive": 0.7, "negative": 0.2, "neutral": 0.1}
    )
    marks[1]["pool_class_size"] = "901"
    marks.append(
        {"key": "d", "slice": "validation", "predicted": "positive", "pool_class_size": "900"}
    )
    with pytest.raises(gold_metrics.MarkedFileError, match="stratum positive has two pool sizes"):
        rep.scored_from_rows([*predictions, {**predictions[0], "key": "d"}], marks)


def _verdicts(score: str, precision: str, recall: str, roc: str) -> dict[str, str]:
    return {
        "score": score,
        "negative precision": precision,
        "negative recall": recall,
        "negative roc_ovr": roc,
    }


def test_the_pin_rule_pins_on_a_better_score_gain_and_no_not_better_negative_precision_recall_or_roc() -> (
    None
):
    pins, reason = rep.pin_decision(_verdicts("better", "not decided", "better", "better"))
    assert pins and "precision `not decided`, recall `better`, ROC OVR `better`" in reason
    assert rep.pin_decision(_verdicts("better", "not decided", "not decided", "not decided"))[0]
    for score in ("not better", "not decided", "undefined"):  # (3): only `better` pins
        pins, reason = rep.pin_decision(_verdicts(score, "better", "better", "better"))
        assert (
            not pins
            and f"the selection score gain is `{score}`" in reason
            and "(D35 (3))" in reason
        )
    for name, args in (
        ("negative precision", ("not better", "better", "better")),
        ("negative recall", ("better", "not better", "better")),
        ("negative roc_ovr", ("better", "better", "not better")),
    ):
        pins, reason = rep.pin_decision(_verdicts("better", *args))  # (4)
        assert not pins and f"the {name} gain is `not better` (D35 (4))" in reason


def test_an_undefined_negative_gain_is_read_as_not_passing_by_the_owners_answer() -> None:
    for args in (
        ("undefined", "better", "better"),
        ("better", "undefined", "better"),
        ("better", "better", "undefined"),
    ):
        pins, reason = rep.pin_decision(_verdicts("better", *args))
        assert not pins and "`undefined`, read as not passing (owner, 2026-11-09)" in reason
