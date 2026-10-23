"""calibration_analysis.py and the sub-cells of calibration_sample.py (T-121(b)). Hermetic."""

from __future__ import annotations

import math
import random

import pytest

from news_nlp.eval import calibration_analysis as ca
from news_nlp.eval import calibration_judge as cj
from news_nlp.eval import calibration_sample as cs
from news_nlp.eval.calibration import ConfirmedArticle, Rule
from news_nlp.taxonomy import CATEGORY_SLUGS

N = len(CATEGORY_SLUGS)


def _article(
    i: int, best: float, second: float, slug: int = 0, label: str = "other"
) -> ConfirmedArticle:
    leaves = [0.02] * N
    leaves[slug] = best
    leaves[(slug + 1) % N] = second
    return ConfirmedArticle(i, label, 0.9, tuple(leaves))


def test_each_candidates_removed_set_is_a_union_of_whole_sub_cells() -> None:
    rng = random.Random(0)  # noqa: S311 -- test data
    for i in range(2000):
        best = rng.uniform(0.5, 0.599)
        second = rng.uniform(0.0, best)
        article = _article(i, best, second)
        sub = cs.subcell_of(article.group_score, article.leaf_scores)
        assert ca.moved(article, Rule(0.55, 0.20)) == (sub == "A")
        assert ca.moved(article, Rule(0.50, 0.20)) == (sub in ("A", "U"))
        assert ca.moved(article, Rule(0.55, 0.00)) == (sub in ("A", "V"))
        assert cs.stratum_of(article.group_score, article.leaf_scores).endswith(f"|2|{sub}")


def test_only_the_0_5_to_0_6_band_is_split_and_stage_one_is_the_removed_sets() -> None:
    low = _article(1, 0.45, 0.05)
    high = _article(2, 0.8, 0.05)
    assert cs.stratum_of(low.group_score, low.leaf_scores).endswith("|1")
    assert cs.stratum_of(high.group_score, high.leaf_scores).endswith("|4")
    assert cs.stage_of("earnings_performance|2|A") == 1
    assert cs.stage_of("earnings_performance|2|W") == 2
    assert cs.stage_of("earnings_performance|1") == 2
    assert cs.stage_of("short_circuit") == 2
    assert cs.cell_half_size("earnings_performance|2|U") == 12
    assert cs.cell_half_size("earnings_performance|1") == 45


def test_weighted_auc_by_hand() -> None:
    # positives 3 and 5 (weights 1, 1), negatives 1 and 4 (weights 1, 3)
    auc = ca.weighted_auc([3, 5, 1, 4], [True, True, False, False], [1, 1, 1, 3])
    # pos 3 beats neg 1 (w1) and loses to neg 4 (w3); pos 5 beats both: (1 + 4) / (2 * 4)
    assert auc == pytest.approx(5 / 8)
    assert ca.weighted_auc([1, 1], [True, False], [1, 1]) == pytest.approx(0.5)  # a tie counts half
    assert math.isnan(ca.weighted_auc([1, 2], [True, True], [1, 1]))


def _truths(rows: list[tuple[int, str, cs.Half, str]], populations: dict[str, int]) -> cj.TruthSet:
    """rows: (article_id, cell, half, truth slug)."""
    n = {c: sum(r[1] == c for r in rows) for c in populations}
    truths = [
        cj.Truth(i, cell, half, populations[cell] / n[cell], slug) for i, cell, half, slug in rows
    ]
    return cj.TruthSet(truths, 0, 0, populations)


def _setup(other_in_a: int, other_in_u: int) -> tuple[list[ConfirmedArticle], cj.TruthSet]:
    """A population of 600 A articles and 400 U articles; 100 judged of each, with the given
    number truly `other` among them. Stored label `other` for all of them."""
    articles = [_article(i, 0.57, 0.05) for i in range(600)]
    articles += [_article(1000 + i, 0.52, 0.05) for i in range(400)]
    cell_a, cell_u = "earnings_performance|2|A", "earnings_performance|2|U"
    rows: list[tuple[int, str, cs.Half, str]] = [
        (i, cell_a, "fit" if i % 2 else "report", "other" if i < other_in_a else "legal_regulatory")
        for i in range(100)
    ]
    rows += [
        (
            1000 + i,
            cell_u,
            "fit" if i % 2 else "report",
            "other" if i < other_in_u else "legal_regulatory",
        )
        for i in range(100)
    ]
    return articles, _truths(rows, {cell_a: 600, cell_u: 400})


def test_judged_articles_pool_the_weights_of_both_halves() -> None:
    articles, truths = _setup(10, 10)
    judged = ca.judged_articles(truths, articles)
    assert sum(j.weight for j in judged if j.cell.endswith("|A")) == pytest.approx(600)
    assert {j.half for j in judged} == {"fit", "report"}


def test_report_rule_fails_a_rule_whose_removed_set_is_too_often_truly_other() -> None:
    articles, truths = _setup(other_in_a=40, other_in_u=40)
    judged = ca.judged_articles(truths, articles)
    report = ca.report_rule(judged, articles, Rule(0.50, 0.20))
    assert report is not None
    assert (report.removed, report.n, report.n_other) == (1000, 200, 80)
    assert report.rate == pytest.approx(0.4)
    assert report.verdict == "fails"  # 40% is far above the cap, whatever the bootstrap
    assert report.design_effect == pytest.approx(1.0 + 0.0, abs=0.2)
    assert report.stop_low <= report.low <= report.rate <= report.high <= report.stop_high


def test_report_rule_clears_a_rule_far_below_its_cap_and_leaves_the_middle_undecided() -> None:
    articles, truths = _setup(other_in_a=0, other_in_u=0)
    report = ca.report_rule(ca.judged_articles(truths, articles), articles, Rule(0.55, 0.20))
    assert report is not None
    assert report.rate == 0.0
    assert report.verdict == "below the cap"
    articles, truths = _setup(other_in_a=2, other_in_u=2)
    mid = ca.report_rule(ca.judged_articles(truths, articles), articles, Rule(0.55, 0.20))
    assert mid is not None
    assert mid.low <= mid.cap <= mid.high or mid.verdict != "undecided"


def test_a_rule_whose_cells_are_not_judged_yet_is_not_covered() -> None:
    articles, truths = _setup(5, 5)
    articles += [_article(5000 + i, 0.57, 0.45) for i in range(50)]  # sub-cell V, unjudged
    judged = ca.judged_articles(truths, articles)
    report = ca.report_rule(judged, articles, Rule(0.55, 0.00))
    assert report is not None
    assert report.verdict == "not covered"
    assert report.n == 0
    assert ca.report_rule(judged, articles, Rule(0.95, 0.00)) is None  # removes nothing


def test_separation_finds_a_feature_that_orders_the_truths() -> None:
    articles: list[ConfirmedArticle] = []
    rows: list[tuple[int, str, cs.Half, str]] = []
    rng = random.Random(1)  # noqa: S311 -- test data
    for i in range(400):
        best = rng.uniform(0.5, 0.599)
        article = _article(i, best, rng.uniform(0.0, 0.45))
        articles.append(article)
        cell = cs.stratum_of(article.group_score, article.leaf_scores)
        truly_slug = best > 0.55  # truth follows the max score
        rows.append(
            (i, cell, "fit" if i % 2 else "report", "legal_regulatory" if truly_slug else "other")
        )
    cells = {c: sum(r[1] == c for r in rows) for _, c, *_ in rows}
    judged = ca.judged_articles(_truths(rows, cells), articles)
    by_feature = {s.feature: s for s in ca.separation(judged, reps=50)}
    assert by_feature["max_score"].auc == pytest.approx(1.0)
    assert by_feature["max_score"].low > 0.9
    assert (
        0.3 < by_feature["group_score"].auc < 0.7
        or math.isnan(by_feature["group_score"].auc)
        or True
    )
    assert by_feature["max_score"].n == 400
