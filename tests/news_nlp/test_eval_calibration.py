"""Offline simulation of category decision rules (TASKS.md T-121(a)): hermetic, no judge
call, no model."""

import numpy as np
import pytest

from news_nlp.eval.calibration import (
    BASELINE,
    ConfirmedArticle,
    Rule,
    TruthModel,
    apply_rule,
    band_of,
    candidate_rules,
    cell_of,
    fit_truth_model,
    frame_for,
    implied_gain,
    relabel_table,
    report_half_size,
    retag,
    simulate_power,
    slug_share_for_gain,
    verdict_table,
)
from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.verdicts import CategoryVerdict
from news_nlp.taxonomy import CATEGORY_GROUP_FLOOR, CATEGORY_SLUGS, OTHER_LABEL


def _leaves(top: int, score: float, second: float = 0.0) -> tuple[float, ...]:
    out = [0.0] * len(CATEGORY_SLUGS)
    out[top] = score
    out[(top + 1) % len(out)] = second
    return tuple(out)


def _article(aid: int, top: int, score: float, second: float = 0.1) -> ConfirmedArticle:
    leaves = _leaves(top, score, second)
    return ConfirmedArticle(aid, apply_rule(0.9, leaves), 0.9, leaves)


def test_baseline_reproduces_the_shipped_rule_and_the_floor() -> None:
    assert apply_rule(0.9, _leaves(2, 0.7)) == CATEGORY_SLUGS[2]
    assert apply_rule(0.9, _leaves(2, 0.59)) == OTHER_LABEL
    # the level-1 floor wins over any leaf score; short-circuit rows carry none
    assert apply_rule(CATEGORY_GROUP_FLOOR - 0.01, _leaves(2, 0.99)) == OTHER_LABEL


def test_threshold_and_margin_rules() -> None:
    leaves = _leaves(0, 0.62, second=0.35)
    assert apply_rule(0.9, leaves, Rule(0.60, 0.0)) == CATEGORY_SLUGS[0]
    assert apply_rule(0.9, leaves, Rule(0.65, 0.0)) == OTHER_LABEL
    # best 0.62 leads the runner-up 0.35 by 0.27: kept at a 0.20 margin, dropped at 0.30
    assert apply_rule(0.9, leaves, Rule(0.55, 0.20)) == CATEGORY_SLUGS[0]
    assert apply_rule(0.9, leaves, Rule(0.55, 0.30)) == OTHER_LABEL
    assert apply_rule(0.9, _leaves(0, 0.52, second=0.1), Rule(0.50, 0.20)) == CATEGORY_SLUGS[0]


def test_cells_and_bands() -> None:
    assert band_of(0.39) == 0 and band_of(0.4) == 1 and band_of(0.6) == 3 and band_of(0.8) == 4
    assert cell_of(0.9, _leaves(3, 0.55)) == f"{CATEGORY_SLUGS[3]}|2"
    assert cell_of(0.1, ()) == "short_circuit"
    assert report_half_size(f"{CATEGORY_SLUGS[0]}|1") == 45
    assert report_half_size(f"{CATEGORY_SLUGS[0]}|4") == 30
    assert report_half_size("short_circuit") == 30


def test_relabel_counts_by_direction() -> None:
    articles = [
        _article(1, 0, 0.9),  # stays a slug under 0.7; goes to other under 0.95
        _article(2, 0, 0.65),  # slug -> other under 0.7
        _article(3, 1, 0.55),  # other -> slug under 0.5
        _article(4, 1, 0.30),  # other under every rule
    ]
    by_rule = {r.rule.name: r for r in relabel_table(articles, [Rule(0.5), Rule(0.7), Rule(0.95)])}
    assert by_rule["t0.50+m0.00"].from_other == 1 and by_rule["t0.50+m0.00"].to_other == 0
    assert by_rule["t0.70+m0.00"].to_other == 1 and by_rule["t0.70+m0.00"].changed == 1
    assert by_rule["t0.95+m0.00"].to_other == 2
    assert by_rule["t0.70+m0.00"].by_cell == {f"{CATEGORY_SLUGS[0]}|3": 1}


def test_the_baseline_relabels_nothing() -> None:
    articles = [_article(i, i % 9, 0.3 + 0.07 * i) for i in range(10)]
    assert relabel_table(articles, [BASELINE])[0].changed == 0


def _judged(aid: int, label: str, ideal: str, top: float, bucket: str = "representative") -> tuple:
    leaves = _leaves(0, top)
    dist = dict(zip(CATEGORY_SLUGS, leaves, strict=True))
    item = EvalItem(
        article_id=aid,
        bucket=bucket,
        stratum_population=100,
        title="",
        body_text="",
        prediction={"label": label, "score": top, "distribution": dist},
    )
    return item, CategoryVerdict(agrees=label == ideal, ideal_slug=ideal)


def test_retag_only_tightens_and_refuses_a_loosening_rule() -> None:
    item = _judged(1, CATEGORY_SLUGS[0], CATEGORY_SLUGS[0], 0.65)[0]
    assert retag(item, Rule(0.7)).prediction["label"] == OTHER_LABEL
    assert retag(item, Rule(0.6)).prediction["label"] == CATEGORY_SLUGS[0]
    stored_other = _judged(2, OTHER_LABEL, OTHER_LABEL, 0.55)[0]
    assert retag(stored_other, Rule(0.7)).prediction["label"] == OTHER_LABEL
    with pytest.raises(ValueError, match="away from 'other'"):
        retag(item, Rule(0.5))


def test_verdict_table_scores_the_baseline_like_aggregate_category() -> None:
    pairs = [
        _judged(1, CATEGORY_SLUGS[0], CATEGORY_SLUGS[0], 0.9),
        _judged(2, CATEGORY_SLUGS[0], OTHER_LABEL, 0.65),
        _judged(3, OTHER_LABEL, OTHER_LABEL, 0.3),
        _judged(4, OTHER_LABEL, CATEGORY_SLUGS[0], 0.55),
    ]
    table = verdict_table(pairs, [BASELINE, Rule(0.7), Rule(0.5)])
    assert set(table) == {BASELINE.name, "t0.70+m0.00"}  # the loosening rule is skipped
    # sending the wrong 0.65 label to `other` raises `other` recall and lowers its precision
    # no further than the one extra correct `other` allows
    assert table["t0.70+m0.00"]["other_rate_model"] > table[BASELINE.name]["other_rate_model"]
    assert table["t0.70+m0.00"]["precision_other"] > table[BASELINE.name]["precision_other"]
    assert 0.0 <= table[BASELINE.name]["mean_slug_precision"] <= 1.0


def test_truth_model_reads_agreement_and_other_precision_from_stored_verdicts() -> None:
    pairs = (
        [_judged(i, CATEGORY_SLUGS[0], CATEGORY_SLUGS[0], 0.9) for i in range(1, 9)]
        + [_judged(i, CATEGORY_SLUGS[0], OTHER_LABEL, 0.9) for i in range(9, 11)]
        + [_judged(i, OTHER_LABEL, OTHER_LABEL, 0.3) for i in range(11, 13)]
        + [_judged(i, OTHER_LABEL, CATEGORY_SLUGS[0], 0.55) for i in range(13, 15)]
    )
    truth = fit_truth_model(pairs, slug_share=0.3, min_judged=10)
    assert truth.agree[f"{CATEGORY_SLUGS[0]}|4"] == pytest.approx(0.8)
    assert truth.other_given_wrong == 1.0
    assert truth.other_agreement == pytest.approx(0.5)
    # a >= 0.6 cell with too few judged items falls back to nothing measured: its band rate
    assert truth.agree[f"{CATEGORY_SLUGS[1]}|4"] == pytest.approx(0.8)
    assert truth.probabilities(f"{CATEGORY_SLUGS[0]}|4") == pytest.approx((0.8, 0.2))
    # the slug share ramps with the band and is zero below 0.4 and in the short-circuit cell
    assert truth.probabilities(f"{CATEGORY_SLUGS[0]}|2")[0] == pytest.approx(0.3)
    assert truth.probabilities(f"{CATEGORY_SLUGS[0]}|1")[0] == pytest.approx(0.15)
    assert truth.probabilities(f"{CATEGORY_SLUGS[0]}|0")[0] == 0.0
    assert truth.probabilities("short_circuit")[0] == 0.0


def _population() -> list[ConfirmedArticle]:
    """Four cells: 200 confident (>= 0.8), 100 in 0.6-0.8, 200 in 0.5-0.6, 100 short-circuit."""
    articles = [_article(i, 0, 0.9) for i in range(200)]
    articles += [_article(200 + i, 0, 0.7) for i in range(100)]
    articles += [_article(300 + i, 0, 0.55) for i in range(200)]
    articles += [ConfirmedArticle(500 + i, OTHER_LABEL, 0.1, (0.0,) * 9) for i in range(100)]
    return articles


_TRUTH = TruthModel(
    agree={f"{s}|{b}": 0.9 for s in CATEGORY_SLUGS for b in (3, 4)},
    other_given_wrong=0.5,
    other_agreement=0.3,
    slug_share=0.0,
)


def test_implied_gain_is_zero_for_the_baseline_and_grows_with_the_slug_share() -> None:
    articles = _population()
    assert implied_gain(frame_for(articles, BASELINE), _TRUTH) == 0.0
    loosen = frame_for(articles, Rule(0.5))
    low = implied_gain(loosen, TruthModel(**{**_TRUTH.__dict__, "slug_share": 0.2}))
    high = implied_gain(loosen, TruthModel(**{**_TRUTH.__dict__, "slug_share": 0.8}))
    assert 0.0 < low < high
    # tightening sends items that are mostly right to `other`: precision falls
    tighten = frame_for(articles, Rule(0.8))
    assert implied_gain(tighten, _TRUTH) < 0.0


def test_slug_share_for_gain_inverts_the_implied_gain() -> None:
    frame = frame_for(_population(), Rule(0.5))
    share = slug_share_for_gain(frame, _TRUTH, 0.05)
    assert share is not None
    got = implied_gain(frame, TruthModel(**{**_TRUTH.__dict__, "slug_share": share}))
    assert got == pytest.approx(0.05, abs=1e-3)
    assert slug_share_for_gain(frame, _TRUTH, 0.9) is None  # not reachable


def test_power_is_deterministic_and_detects_a_large_gain_but_not_none() -> None:
    frame = frame_for(_population(), Rule(0.5))
    big = TruthModel(**{**_TRUTH.__dict__, "slug_share": 1.0})
    a = simulate_power(frame, big, draws=20, boot_reps=100, seed=3)
    b = simulate_power(frame, big, draws=20, boot_reps=100, seed=3)
    assert a == b
    assert a.implied_gain > 0.05 and a.power > 0.8
    none = simulate_power(frame, _TRUTH, draws=20, boot_reps=100, seed=3)
    assert none.implied_gain == pytest.approx(0.0, abs=1e-9)
    assert none.power < 0.3


def test_candidate_rules_are_the_grid_and_include_the_shipped_rule() -> None:
    rules = candidate_rules()
    assert BASELINE in rules
    assert len(rules) == len(set(rules))
    assert np.isclose(min(r.threshold for r in rules), 0.5)
