"""news_nlp.gold_metrics and news_nlp.attribution_baseline (Work item 18, D27, D32)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from news_nlp import attribution_baseline, gold_metrics
from news_nlp.gold_metrics import Figure, MarkedFileError
from news_nlp.triage import Company, CompanyResolver


def test_wilson_matches_the_known_interval_and_handles_no_data() -> None:
    f = gold_metrics.wilson(54, 60)  # D26: an observed 90% at n = 60 has a lower bound near 80%
    assert round(f.point, 3) == 0.9 and 0.78 < f.low < 0.81
    assert 0.84 < gold_metrics.wilson(180, 200).low < 0.86  # D30: near 85%
    assert math.isnan(gold_metrics.wilson(0, 0).point)


def test_a_figure_is_judged_against_a_bar_by_its_whole_interval() -> None:
    assert Figure(0.9, 0.85, 0.95).judge(0.8) == "pass"
    assert Figure(0.6, 0.5, 0.7).judge(0.8) == "fail"
    assert Figure(0.75, 0.6, 0.9).judge(0.7) == "not decided"
    assert Figure(0.65, 0.5, 0.8).judge(0.7) == "not decided"  # point under the bar, interval over


def _rows() -> list[dict[str, str]]:
    # 10 per predicted class. positive stratum: 8 right; negative: 6 right; neutral: 8 right, and
    # one negative hides in the neutral stratum (the hidden true negative recall needs).
    rows = []
    for i in range(10):
        rows.append({"predicted": "positive", "gold": "positive" if i < 8 else "neutral"})
        rows.append({"predicted": "negative", "gold": "negative" if i < 6 else "neutral"})
        gold = "neutral" if i < 8 else ("negative" if i == 8 else "positive")
        rows.append({"predicted": "neutral", "gold": gold})
    return rows


def test_sentiment_figures_are_reweighted_by_the_stratum_sizes() -> None:
    pops = {"positive": 100, "negative": 20, "neutral": 880}
    report = gold_metrics.sentiment_report(_rows(), pops, resamples=300)
    assert report.drawn == {"positive": 10, "negative": 10, "neutral": 10}
    assert report.raw_agreement.point == pytest.approx(22 / 30)
    # neutral is 88% of the pool and 8 of its 10 drawn agree: the weighted figure follows it
    expected = (100 * 0.8 + 20 * 0.6 + 880 * 0.8) / 1000
    assert report.agreement.point == pytest.approx(expected)
    assert report.negative_precision.point == pytest.approx(0.6)
    # recall: true negatives = 20*0.6 + 880*0.1 = 100 (the one in the neutral stratum weighs 88)
    assert report.negative_recall.point == pytest.approx(12 / (12 + 88))
    assert report.negative_recall.high - report.negative_recall.low > 0.1  # a wide interval
    again = gold_metrics.sentiment_report(_rows(), pops, resamples=300)
    assert again.agreement == report.agreement  # seeded


def test_a_stratum_without_a_population_is_refused() -> None:
    with pytest.raises(MarkedFileError, match="no population"):
        gold_metrics.sentiment_report(_rows(), {"positive": 1}, resamples=10)


def test_paired_difference_is_the_gain_over_the_baseline_on_the_same_items() -> None:
    model = [True] * 27 + [False] * 3
    base = [True] * 20 + [False] * 10
    f = gold_metrics.paired_difference(model, base, resamples=500)
    assert f.point == pytest.approx(7 / 30) and f.low > 0
    with pytest.raises(ValueError, match="same length"):
        gold_metrics.paired_difference([True], [True, False])


def _file(tmp_path: Path, values: list[str]) -> Path:
    path = tmp_path / "f.csv"
    lines = ["article_id,kind,pub_date,correct"] + [
        f"{i},{'lead' if i % 2 else 'continuation'},202{i % 5}-01-01,{v}"
        for i, v in enumerate(values, 1)
    ]
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def test_a_half_marked_or_mistyped_file_is_refused_with_the_rows_at_fault(tmp_path: Path) -> None:
    with pytest.raises(MarkedFileError, match="1 rows without correct"):
        gold_metrics.read_marked(_file(tmp_path, ["yes", ""]), "correct", ("yes", "no"))
    with pytest.raises(MarkedFileError, match="'maybe'"):
        gold_metrics.read_marked(_file(tmp_path, ["yes", "maybe"]), "correct", ("yes", "no"))
    with pytest.raises(MarkedFileError, match="lacks the columns"):
        gold_metrics.read_marked(_file(tmp_path, ["yes"]), "sentiment", ("yes",))
    rows = gold_metrics.read_marked(_file(tmp_path, [" YES ", "No"]), "correct", ("yes", "no"))
    assert [r["correct"] for r in rows] == ["yes", "no"]


def test_the_label_check_drops_continuation_only_at_or_under_the_bar() -> None:
    def rows(kind: str, right: int, total: int) -> list[dict[str, str]]:
        return [
            {"kind": kind, "correct": "yes" if i < right else "no", "pub_date": "2023-01-01"}
            for i in range(total)
        ]

    low = gold_metrics.label_check_report(rows("lead", 95, 100) + rows("continuation", 80, 100))
    assert gold_metrics.train_on_lead_only(low) is True  # exactly 80%: "at most 80%" drops them
    high = gold_metrics.label_check_report(rows("lead", 95, 100) + rows("continuation", 81, 100))
    assert gold_metrics.train_on_lead_only(high) is False
    assert high["lead 2022+"].point == pytest.approx(0.95)


def test_a_bar_is_judged_on_the_whole_interval_including_the_edges() -> None:
    fig = gold_metrics.Figure
    assert fig(0.9, 0.80, 0.95).judge(0.80) == "pass"  # low end exactly at the bar
    assert fig(0.7, 0.60, 0.80).judge(0.80) == "not decided"  # high end exactly at the bar
    assert fig(0.7, 0.60, 0.79).judge(0.80) == "fail"
    assert fig(0.82, 0.74, 0.89).judge(0.80) == "not decided"  # point above, interval straddles


def test_the_gain_over_the_baseline_has_a_verdict() -> None:
    fig = gold_metrics.Figure
    assert gold_metrics.gain_verdict(fig(0.2, 0.05, 0.35)) == "better"
    assert gold_metrics.gain_verdict(fig(0.0, -0.1, 0.1)) == "not decided"
    assert gold_metrics.gain_verdict(fig(-0.1, -0.2, 0.0)) == "not decided"  # only touches 0
    assert gold_metrics.gain_verdict(fig(-0.1, -0.2, -0.01)) == "not better"


def test_the_label_check_refuses_what_it_cannot_judge() -> None:
    with pytest.raises(MarkedFileError, match="pub_date"):
        gold_metrics.label_check_report([{"kind": "lead", "correct": "yes"}])
    only_lead = gold_metrics.label_check_report(
        [{"kind": "lead", "correct": "yes", "pub_date": "2023-01-01"}]
    )
    with pytest.raises(MarkedFileError, match="continuation"):
        gold_metrics.train_on_lead_only(only_lead)
    rows = [
        {"kind": "lead", "correct": "yes", "pub_date": ""},
        {"kind": "lead", "correct": "yes", "pub_date": "2021-05-01"},
        {"kind": "lead", "correct": "yes", "pub_date": "2022-01-01"},
    ]
    assert gold_metrics.rows_without_year(rows) == 1
    assert gold_metrics.label_check_report(rows)["lead 2022+"].point == 1.0


def test_faithfulness_is_counted_per_extractor_when_a_text_is_shared() -> None:
    marks = {"a": True, "b": False, "c": True, "d": True}
    owners = {"a": ["baseline", "model"], "b": ["baseline"], "c": ["model"], "d": ["model"]}
    got = gold_metrics.faithfulness_report(marks, owners)
    assert got["baseline"].point == pytest.approx(0.5) and got["model"].point == 1.0


UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
]


def test_the_baseline_gives_a_sentence_to_the_companies_it_names_and_then_by_recency() -> None:
    body = (
        "Markets were quiet on Tuesday. Apple raised its dividend by four percent. "
        "The move surprised analysts. Microsoft said cloud growth slowed. "
        "Investors reacted badly. Apple and Microsoft both fell."
    )
    resolver = CompanyResolver(UNIVERSE)
    got = attribution_baseline.attribute_sentences(body, resolver)
    assert got[0][1] == ()  # before any mention: nobody's
    assert got[2][1] == ("AAPL",)  # no name: the company of the sentence before
    assert got[4][1] == ("MSFT",)
    assert got[5][1] == ("AAPL", "MSFT")  # names two: both
    texts = attribution_baseline.baseline_company_texts(body, resolver)
    assert list(texts) == ["AAPL", "MSFT"]
    assert "surprised analysts" in texts["AAPL"] and "Markets were quiet" not in texts["AAPL"]
    assert "Investors reacted badly" in texts["MSFT"]


def test_faithfulness_interval_is_over_bodies_not_texts() -> None:
    # 10 bodies of 4 texts: six bodies are all faithful, four are all unfaithful (the texts of a
    # body stand or fall together), so 40 texts carry the information of about 10.
    marks, owners = {}, {}
    for body in range(10):
        for n in range(4):
            key = f"{body}#T#{n}"
            marks[key], owners[key] = body < 6, ["model"]
    by_body = gold_metrics.faithfulness_report(marks, owners, resamples=500)["model"]
    by_text = gold_metrics.wilson(24, 40)
    assert by_body.point == pytest.approx(0.6)
    assert by_body.high - by_body.low > by_text.high - by_text.low  # honestly wider


def test_the_gain_over_the_baseline_can_resample_bodies() -> None:
    model = [True] * 20 + [False] * 20
    base = [False] * 20 + [True] * 20
    items = [f"{i // 4}" for i in range(40)]  # 10 bodies of 4 texts
    per_text = gold_metrics.paired_difference(model, base, resamples=500)
    per_body = gold_metrics.paired_difference(model, base, clusters=items, resamples=500)
    assert per_body.point == per_text.point
    assert per_body.high - per_body.low >= per_text.high - per_text.low
    with pytest.raises(ValueError, match="one cluster per item"):
        gold_metrics.paired_difference(model, base, clusters=["a"])


def test_the_body_of_a_text_key() -> None:
    assert gold_metrics.body_of_key("123#AAPL#2") == "123" and gold_metrics.body_of_key("x") == "x"


def test_an_omitted_company_counts_as_an_unfaithful_text() -> None:
    marks = {"1#T#1": True, "2#T#1": True}
    owners = {"1#T#1": ["model"], "2#T#1": ["model"]}
    plain = gold_metrics.faithfulness_report(marks, owners, resamples=200)["model"]
    abstains = gold_metrics.faithfulness_report(
        marks, owners, omitted={"model": ["3#T", "4#T", "5#T"]}, resamples=200
    )["model"]
    assert plain.point == 1.0 and abstains.point == pytest.approx(2 / 5)


def test_the_guards_name_duplicate_and_missing_rows() -> None:
    with pytest.raises(MarkedFileError, match="repeated keys"):
        gold_metrics.require_unique(["a", "b", "a"], "f.csv")
    gold_metrics.require_unique(["a", "b"], "f.csv")
    with pytest.raises(MarkedFileError, match=r"2 rows of the draw are not in the tagged file"):
        gold_metrics.require_same_keys(["a"], ["a", "b", "c"], "f.csv")
    gold_metrics.require_same_keys(["a", "b"], ["a", "b"], "f.csv")


def test_a_stratum_drawn_whole_is_not_resampled() -> None:
    rows = [{"predicted": "negative", "gold": "negative", "stratum": "a"} for _ in range(5)]
    rows += [
        {"predicted": "neutral", "gold": "neutral" if i % 2 else "negative", "stratum": "b"}
        for i in range(10)
    ]
    whole = gold_metrics.stratified_report(
        rows, {"a": 5, "b": 100}, label="negative", stratum_key="stratum", resamples=300
    )
    assert whole.census == ("a",)
    sampled = gold_metrics.stratified_report(
        rows, {"a": 50, "b": 100}, label="negative", stratum_key="stratum", resamples=300
    )
    assert sampled.census == ()
    # with every stratum drawn whole there is no sampling error at all
    every = gold_metrics.stratified_report(
        rows, {"a": 5, "b": 10}, label="negative", stratum_key="stratum", resamples=300
    )
    assert every.agreement.low == every.agreement.high == every.agreement.point


def test_a_figure_with_no_interval_is_undefined_not_undecided() -> None:
    """A precision with no text predicted is NaN: it is not an interval that straddles the bar
    (that goes to the cut, D34), it is a figure that does not exist."""
    nan = math.nan
    assert Figure(nan, nan, nan).judge(0.7) == "undefined"
    assert Figure(0.6, nan, nan).judge(0.7) == "undefined"  # a point, but no interval to judge by
    assert Figure(0.6, 0.5, nan).judge(0.7) == "undefined"
    assert gold_metrics.gain_verdict(Figure(nan, nan, nan)) == "undefined"
    assert gold_metrics.gain_verdict(Figure(0.0, -0.1, 0.1)) == "not decided"  # unchanged


def _stratified_rows(
    candidate_right: int, baseline_right: int, n: int = 40
) -> list[dict[str, str]]:
    """*n* rows of one stratum, the owner's class cycling over the three; the first
    *candidate_right* are right for the candidate and the first *baseline_right* for the baseline."""
    classes = ["positive", "negative", "neutral"]
    rows = []
    for i in range(n):
        gold = classes[i % 3]
        wrong = classes[(i + 1) % 3]
        rows.append(
            {
                "stratum": "checked/positive",
                "gold": gold,
                "predicted": gold if i < candidate_right else wrong,
                "baseline": gold if i < baseline_right else wrong,
            }
        )
    return rows


def test_the_paired_stratified_gain_is_better_only_when_the_whole_interval_is_above_zero() -> None:
    populations = {"checked/positive": 400}
    big = gold_metrics.paired_stratified_gain(_stratified_rows(38, 10), populations, resamples=400)
    assert big.point == pytest.approx((38 - 10) / 40)
    assert gold_metrics.gain_verdict(big) == "better"
    same = gold_metrics.paired_stratified_gain(_stratified_rows(20, 20), populations, resamples=400)
    assert same.point == 0 and gold_metrics.gain_verdict(same) == "not decided"
    worse = gold_metrics.paired_stratified_gain(_stratified_rows(5, 35), populations, resamples=400)
    assert gold_metrics.gain_verdict(worse) == "not better"


def test_the_paired_stratified_gain_of_precision_and_recall_of_a_label() -> None:
    populations = {"checked/positive": 40}
    rows = [
        {
            "stratum": "checked/positive",
            "gold": "negative",
            "predicted": "negative",
            "baseline": "neutral",
        },
        {
            "stratum": "checked/positive",
            "gold": "negative",
            "predicted": "negative",
            "baseline": "negative",
        },
        {
            "stratum": "checked/positive",
            "gold": "neutral",
            "predicted": "neutral",
            "baseline": "negative",
        },
        {
            "stratum": "checked/positive",
            "gold": "neutral",
            "predicted": "neutral",
            "baseline": "neutral",
        },
    ]
    recall = gold_metrics.paired_stratified_gain(rows, populations, metric="recall", resamples=200)
    assert recall.point == pytest.approx(1.0 - 0.5)  # candidate finds both negatives, baseline one
    precision = gold_metrics.paired_stratified_gain(
        rows, populations, metric="precision", resamples=200
    )
    assert precision.point == pytest.approx(1.0 - 0.5)  # baseline: 1 of its 2 negatives is right


def test_the_paired_stratified_gain_refuses_an_unknown_stratum_and_an_unknown_metric() -> None:
    rows = _stratified_rows(10, 10)
    with pytest.raises(MarkedFileError, match="no population"):
        gold_metrics.paired_stratified_gain(rows, {"other": 5})
    with pytest.raises(ValueError, match="metric"):
        gold_metrics.paired_stratified_gain(rows, {"checked/positive": 5}, metric="f1")
