"""The report of a reading of the validation slice (T-148c3): raw description and weighted figures."""

from __future__ import annotations

import math

import pytest

import sentiment_pair_report as rep


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


def _rows() -> tuple[list[dict[str, str]], dict[str, int]]:
    """Two strata (the class the pinned model predicted), 10 texts each; the stratum 'neg' is a small pool."""
    rows = []
    for i in range(10):
        gold = "negative" if i < 6 else "positive"
        rows.append(
            {"key": f"n{i}", "stratum": "neg", "gold": gold, "good": gold, "bad": "negative"}
        )
    for i in range(10):
        rows.append(
            {
                "key": f"p{i}",
                "stratum": "pos",
                "gold": "positive",
                "good": "positive",
                "bad": "negative",
            }
        )
    return rows, {"neg": 100, "pos": 900}


def test_the_weighted_figures_weigh_each_text_by_its_strata_pool_not_by_the_draw() -> None:
    rows, populations = _rows()
    good = rep.weighted_figures(rows, populations, "good")
    assert good["agreement"].point == 1.0 and good["negative precision"].point == 1.0
    bad = rep.weighted_figures(rows, populations, "bad")
    # raw: 6 of 20 right (30%); weighted: the 'pos' stratum (900 of 1000) is all wrong, 'neg' is 60% right
    assert bad["agreement"].point == pytest.approx((100 * 0.6 + 0) / 1000)
    # every text read as negative: precision is the weighted share of negatives among them
    assert bad["negative precision"].point == pytest.approx(60 / 1000)
    assert bad["negative recall"].point == 1.0


def test_the_weighted_gain_is_paired_and_comes_with_the_gold_verdict() -> None:
    rows, populations = _rows()
    gains = rep.weighted_gains(rows, populations, "good", "bad")
    assert set(gains) == {"agreement", "negative precision", "negative recall"}
    gain, verdict = gains["agreement"]
    assert gain.point == pytest.approx(1 - 0.06) and verdict == "better"
    gain, verdict = gains["negative recall"]  # both read every negative as negative
    assert gain.point == 0.0 and verdict == "not decided"
    same = rep.weighted_gains(rows, populations, "good", "good")
    assert all(g.point == 0.0 and v == "not decided" for g, v in same.values())


def test_d32_verdicts_are_the_bars_judge_of_each_figure_and_a_better_model_can_still_fail_them() -> (
    None
):
    from news_nlp.gold_metrics import Figure  # noqa: PLC0415

    figures = {
        "agreement": Figure(0.666, 0.569, 0.763),  # wholly below 80%
        "negative precision": Figure(0.46, 0.278, 0.652),  # wholly below 70%
        "negative recall": Figure(0.892, 0.719, 1.0),  # wholly at or above 70%
    }
    assert rep.bar_verdicts(figures) == {
        "agreement": "fail",
        "negative precision": "fail",
        "negative recall": "pass",
    }
    figures["negative recall"] = Figure(0.8, 0.6, 0.95)  # straddles the bar
    assert rep.bar_verdicts(figures)["negative recall"] == "not decided"


def test_the_weighted_agreement_of_a_run_weighs_its_strata_by_the_pool() -> None:
    rows, populations = _rows()
    assert rep.weighted_agreement(rows, populations, "good") == 1.0
    assert rep.weighted_agreement(rows, populations, "bad") == pytest.approx(0.06)
    assert math.isnan(rep.weighted_agreement([], {}, "good"))


def _verdicts(agreement: str, precision: str, recall: str) -> dict[str, str]:
    return {"agreement": agreement, "negative precision": precision, "negative recall": recall}


def test_the_pin_rule_pins_only_on_a_better_agreement_gain_and_no_not_better_precision_or_recall() -> (
    None
):
    pins, reason = rep.pin_decision(_verdicts("better", "not decided", "better"))
    assert pins and "neither the precision (`not decided`) nor the recall (`better`)" in reason
    assert rep.pin_decision(_verdicts("better", "better", "not decided"))[0]
    assert rep.pin_decision(_verdicts("better", "better", "better"))[0]
    for agreement in ("not better", "not decided", "undefined"):  # (3): only `better` pins
        pins, reason = rep.pin_decision(_verdicts(agreement, "better", "better"))
        assert (
            not pins and f"the agreement gain is `{agreement}`" in reason and "(D35 (3))" in reason
        )
    pins, reason = rep.pin_decision(_verdicts("better", "not better", "better"))  # (4)
    assert not pins and "negative precision gain is `not better` (D35 (4))" in reason
    pins, reason = rep.pin_decision(_verdicts("better", "better", "not better"))
    assert not pins and "negative recall gain is `not better` (D35 (4))" in reason


def test_an_undefined_precision_or_recall_gain_is_read_as_not_passing_by_the_owners_answer() -> (
    None
):
    for verdicts in (
        _verdicts("better", "undefined", "better"),
        _verdicts("better", "better", "undefined"),
    ):
        pins, reason = rep.pin_decision(verdicts)
        assert not pins and "`undefined`, read as not passing (owner, 2026-11-09)" in reason


def _row(**changes: str) -> dict[str, str]:
    row = {
        "role": "final",
        "gain_verdict": "better",
        "gain_negative_precision_verdict": "better",
        "gain_negative_recall_verdict": "better",
    }
    return {**row, **changes}


def test_the_pin_is_decided_from_the_three_verdict_columns_each_read_for_its_own_metric() -> None:
    from news_nlp.gold_metrics import MarkedFileError  # noqa: PLC0415

    assert rep.pin_decision_from_row(_row())[0]
    # each column moves only its own metric: swapping them would change who blocks the pin
    pins, reason = rep.pin_decision_from_row(_row(gain_negative_precision_verdict="not better"))
    assert not pins and "negative precision gain is `not better`" in reason
    pins, reason = rep.pin_decision_from_row(_row(gain_negative_recall_verdict="not better"))
    assert not pins and "negative recall gain is `not better`" in reason
    pins, reason = rep.pin_decision_from_row(_row(gain_verdict="not decided"))
    assert not pins and "the agreement gain is `not decided`" in reason
    pins, reason = rep.pin_decision_from_row(_row(gain_negative_recall_verdict="undefined"))
    assert not pins and "read as not passing (owner, 2026-11-09)" in reason
    # a verdict that is `not decided` for precision does not block (the validation figures' case)
    assert rep.pin_decision_from_row(_row(gain_negative_precision_verdict="not decided"))[0]
    for column in rep.ROW_VERDICT_COLUMNS.values():
        missing = _row()
        del missing[column]
        with pytest.raises(MarkedFileError, match=f"{column}: '' is not a verdict"):
            rep.pin_decision_from_row(missing)
        with pytest.raises(MarkedFileError, match=f"{column}: 'good' is not a verdict"):
            rep.pin_decision_from_row(_row(**{column: "good"}))
        with pytest.raises(MarkedFileError, match=f"{column}: '' is not a verdict"):
            rep.pin_decision_from_row(_row(**{column: ""}))  # a baseline row has no gain


def test_a_row_that_is_not_a_final_run_is_refused_and_a_row_without_a_role_is_read() -> None:
    from news_nlp.gold_metrics import MarkedFileError  # noqa: PLC0415

    with pytest.raises(MarkedFileError, match="`final` row, this one is 'baseline'"):
        rep.pin_decision_from_row(_row(role="baseline"))
    without_role = {k: v for k, v in _row().items() if k != "role"}
    assert rep.pin_decision_from_row(without_role)[0]


def test_the_row_columns_are_the_ones_the_scorer_logs() -> None:
    from news_nlp import retag  # noqa: PLC0415

    assert set(rep.ROW_VERDICT_COLUMNS.values()) <= set(retag.SENTIMENT_SCORING_LOG_COLUMNS)
