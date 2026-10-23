"""Offline simulation of category decision rules (TASKS.md T-121(a)): hermetic, no judge
call, no model."""

import importlib.util
import random
import sqlite3
from pathlib import Path

import pytest

from news_nlp.eval.calibration import (
    BASELINE,
    POWER_TARGET,
    RAMPS,
    RECALL_TOLERANCE,
    ConfirmedArticle,
    Rule,
    TruthModel,
    apply_rule,
    band_of,
    candidate_rules,
    cell_of,
    classify_power,
    fit_truth_model,
    frame_for,
    implied_changes,
    implied_gain,
    needs_refinement,
    null_critical,
    null_truth,
    paired_rule_interval,
    relabel_table,
    report_half_size,
    retag,
    simulate_power,
    slug_share_for_gain,
    slug_share_for_recall,
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


def _judged(
    aid: int, label: str, ideal: str, top: float, bucket: str = "representative", pop: int = 100
) -> tuple:
    leaves = _leaves(0, top)
    dist = dict(zip(CATEGORY_SLUGS, leaves, strict=True))
    item = EvalItem(
        article_id=aid,
        bucket=bucket,
        stratum_population=pop,
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
        + [_judged(i, CATEGORY_SLUGS[0], CATEGORY_SLUGS[0], 0.7) for i in range(15, 17)]
        + [_judged(i, OTHER_LABEL, OTHER_LABEL, 0.3) for i in range(11, 13)]
        + [_judged(i, OTHER_LABEL, CATEGORY_SLUGS[0], 0.55) for i in range(13, 15)]
    )
    truth = fit_truth_model(pairs, slug_share=0.3, min_judged=10)
    assert truth.agree[f"{CATEGORY_SLUGS[0]}|4"] == pytest.approx(0.8)
    assert truth.other_given_wrong == 1.0
    assert truth.other_agreement == pytest.approx(0.5)
    # a >= 0.6 cell with fewer than min_judged items uses its band's pooled rate, and is listed
    assert truth.agree[f"{CATEGORY_SLUGS[1]}|4"] == pytest.approx(0.8)
    assert f"{CATEGORY_SLUGS[1]}|4" in truth.fallback_cells
    assert f"{CATEGORY_SLUGS[0]}|4" not in truth.fallback_cells
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
    assert min(r.threshold for r in rules) == 0.5


def test_a_confident_band_with_no_judged_item_is_an_error_not_a_zero() -> None:
    only_band_four = [_judged(i, CATEGORY_SLUGS[0], CATEGORY_SLUGS[0], 0.9) for i in range(1, 4)]
    with pytest.raises(ValueError, match=r"0\.6-0\.8"):
        fit_truth_model(only_band_four, slug_share=0.3)


def test_cell_agreement_is_weighted_by_the_stratum_weight() -> None:
    # two buckets, same cell: one wrong item in a bucket standing for 900, nine right ones
    # in a bucket of ten items standing for 100 (weights 900 and 10 each): the weighted rate
    # is 90/990, not the unweighted 0.9.
    heavy = _judged(1, CATEGORY_SLUGS[0], OTHER_LABEL, 0.9, bucket="heavy", pop=900)
    light = [
        _judged(i, CATEGORY_SLUGS[0], CATEGORY_SLUGS[0], 0.9, bucket="light", pop=100)
        for i in range(2, 11)
    ]
    band_three = _judged(50, CATEGORY_SLUGS[0], CATEGORY_SLUGS[0], 0.7, bucket="light", pop=100)
    truth = fit_truth_model([heavy, *light, band_three], slug_share=0.0, min_judged=10)
    assert truth.agree[f"{CATEGORY_SLUGS[0]}|4"] == pytest.approx(90 / 990)


def test_a_flat_slug_share_leaves_no_gain_and_a_graded_one_does() -> None:
    frame = frame_for(_population(), Rule(0.5))
    flat = TruthModel(**{**_TRUTH.__dict__, "slug_share": 0.5, "ramp": RAMPS["flat"]})
    graded = TruthModel(**{**_TRUTH.__dict__, "slug_share": 0.5, "ramp": RAMPS["graded"]})
    band = TruthModel(**{**_TRUTH.__dict__, "slug_share": 0.5, "ramp": RAMPS["band"]})
    assert implied_gain(frame, graded) > 0.0
    assert implied_gain(frame, band) > 0.0
    # the population here has no 0.4-0.5 or < 0.4 articles, so `band` equals `graded`
    # at the 0.5-0.6 band and `flat` differs only by the short-circuit stratum
    assert implied_gain(frame, flat) < implied_gain(frame, graded)


def test_a_tightening_rule_only_ever_gives_the_baseline_label_or_other() -> None:
    rng = random.Random(1)  # noqa: S311 -- a test fixture, not cryptography
    for _ in range(300):
        leaves = tuple(rng.random() * 0.3 for _ in CATEGORY_SLUGS)
        top = rng.randrange(len(CATEGORY_SLUGS))
        leaves = tuple(rng.uniform(0.3, 1.0) if i == top else x for i, x in enumerate(leaves))
        rule = Rule(rng.choice([0.6, 0.65, 0.8, 0.95]), rng.choice([0.0, 0.1, 0.3]))
        assert rule.tightens
        assert apply_rule(0.9, leaves, rule) in (apply_rule(0.9, leaves), OTHER_LABEL)
    assert not Rule(0.55, 0.3).tightens


def test_paired_interval_is_deterministic_and_matches_the_point_difference() -> None:
    pairs = [_judged(i, CATEGORY_SLUGS[0], CATEGORY_SLUGS[0], 0.9) for i in range(1, 21)]
    pairs += [_judged(i, CATEGORY_SLUGS[0], OTHER_LABEL, 0.65) for i in range(21, 31)]
    pairs += [_judged(i, OTHER_LABEL, OTHER_LABEL, 0.3) for i in range(31, 36)]
    pairs += [_judged(i, OTHER_LABEL, CATEGORY_SLUGS[0], 0.3) for i in range(36, 41)]
    a = paired_rule_interval(pairs, Rule(0.7), "precision_other", reps=100, seed=2)
    assert a == paired_rule_interval(pairs, Rule(0.7), "precision_other", reps=100, seed=2)
    # the ten wrong 0.65 labels go to `other`, where the judge says `other`: precision
    # rises from 5/10 to 15/20
    assert a.difference > 0 and a.low <= a.difference <= a.high and a.used == a.reps
    assert paired_rule_interval(pairs, BASELINE, "precision_other", reps=50).difference == 0.0
    with pytest.raises(ValueError, match="away from 'other'"):
        paired_rule_interval(pairs, Rule(0.5), "precision_other", reps=5)


def _script() -> object:
    path = Path(__file__).resolve().parents[2] / "scripts" / "simulate_category_rules_2026_10_23.py"
    spec = importlib.util.spec_from_file_location("simulate_category_rules", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_loads_the_confirmed_articles_with_leaf_scores_in_slug_order(
    tmp_path: Path,
) -> None:
    script = _script()
    source, results = tmp_path / "urls.db", tmp_path / "nlp.db"
    with sqlite3.connect(source) as c:
        c.execute("CREATE TABLE discovered_urls (id INTEGER PRIMARY KEY, match_class TEXT)")
        c.executemany(
            "INSERT INTO discovered_urls VALUES (?, ?)",
            [(1, "accept"), (2, "reject"), (3, "accept")],
        )
    leaf_cols = ", ".join(f"{s} REAL" for s in CATEGORY_SLUGS)
    with sqlite3.connect(results) as c:
        c.execute(
            "CREATE TABLE article_category (article_id INTEGER, label TEXT, group_score REAL, "
            + leaf_cols
            + ")"
        )
        marks = ", ".join("?" for _ in CATEGORY_SLUGS)
        for aid, label, group in [
            (1, CATEGORY_SLUGS[8], 0.9),
            (2, OTHER_LABEL, 0.9),
            (3, OTHER_LABEL, 0.1),
        ]:
            scores = [0.01 * (i + 1) for i in range(9)]
            scores[8] = 0.8 if aid == 1 else 0.3
            c.execute(
                f"INSERT INTO article_category VALUES (?, ?, ?, {marks})",  # noqa: S608
                (aid, label, group, *scores),
            )
    conn = script.connect(source, results)  # type: ignore[attr-defined]
    articles = script.load_confirmed(conn)  # type: ignore[attr-defined]
    assert [a.article_id for a in articles] == [1, 3]  # the rejected article is left out
    first = articles[0]
    assert first.stored_label == CATEGORY_SLUGS[8] and first.leaf_scores[8] == 0.8
    assert first.leaf_scores[0] == pytest.approx(0.01)  # column order is CATEGORY_SLUGS order
    assert apply_rule(first.group_score, first.leaf_scores) == first.stored_label
    # the connection is read-only
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM discovered_urls")


def test_power_row_reports_its_spread_and_monte_carlo_error() -> None:
    frame = frame_for(_population(), Rule(0.5))
    big = TruthModel(**{**_TRUTH.__dict__, "slug_share": 1.0})
    row = simulate_power(frame, big, draws=30, boot_reps=60, seed=5)
    assert row.defined == 30
    assert row.power_se == pytest.approx((row.power * (1 - row.power) / 30) ** 0.5)
    # the sampled paired difference centres on the implied gain and its lower bound sits below it
    assert row.mean_diff == pytest.approx(row.implied_gain, abs=0.03)
    assert row.sd_diff >= 0.0 and row.mean_lower < row.mean_diff
    # a draw count with no defined draw has no error to report
    assert row._replace(defined=0).power_se == 0.0


def test_partition_and_calibration_share_the_empty_denominator_rule() -> None:
    from news_nlp.eval.partition import is_defined  # noqa: PLC0415

    pairs = [_judged(1, OTHER_LABEL, OTHER_LABEL, 0.3)]
    assert is_defined("precision_other", pairs, {}, "all")
    assert not is_defined(f"precision_{CATEGORY_SLUGS[0]}", pairs, {}, "all")


def test_power_is_classified_by_two_standard_errors_from_the_bar() -> None:
    assert classify_power(0.95, 0.01) == "detectable"
    assert classify_power(0.79, 0.013) == "borderline"  # the T-121(a) borderline row
    assert classify_power(0.89, 0.010) == "detectable"
    assert classify_power(0.40, 0.035) == "not detectable"
    # the boundaries: exactly two standard errors clear, or short, count as clear
    assert classify_power(POWER_TARGET + 0.04, 0.02) == "detectable"
    assert classify_power(POWER_TARGET - 0.04, 0.02) == "borderline"
    assert classify_power(POWER_TARGET - 0.0401, 0.02) == "not detectable"


def test_only_a_power_near_the_bar_is_rerun() -> None:
    assert needs_refinement(0.79) and needs_refinement(0.71) and needs_refinement(0.89)
    assert not needs_refinement(0.69) and not needs_refinement(0.91) and not needs_refinement(0.05)


def test_the_null_truth_gives_every_rule_a_gain_of_zero() -> None:
    frame = frame_for(_population(), Rule(0.5))
    for share in (0.0, 0.5, 1.0):
        assert implied_gain(
            frame, null_truth(TruthModel(**{**_TRUTH.__dict__, "slug_share": share}))
        ) == pytest.approx(0.0)
    assert implied_gain(frame_for(_population(), Rule(0.8)), null_truth(_TRUTH)) == pytest.approx(
        0.0
    )


def test_a_null_critical_value_holds_the_false_positive_rate_on_fresh_draws() -> None:
    frame = frame_for(_population(), Rule(0.5))
    critical = null_critical(frame, _TRUTH, draws=300, boot_reps=100, seed=7)
    assert critical >= 0.0  # a conservative interval is never made more liberal
    # evaluated on draws the critical value was not fitted on (another seed, more draws):
    # nominal is 0.025, and 400 draws put five standard deviations at about 0.065
    fresh = simulate_power(
        frame, null_truth(_TRUTH), draws=400, boot_reps=100, seed=99, critical=critical
    )
    assert fresh.power <= 0.065
    assert fresh.critical == critical
    # under a real gain the calibrated power can only be at or below the nominal one
    big = TruthModel(**{**_TRUTH.__dict__, "slug_share": 1.0})
    nominal = simulate_power(frame, big, draws=40, boot_reps=100, seed=7)
    calibrated = simulate_power(frame, big, draws=40, boot_reps=100, seed=7, critical=critical)
    assert calibrated.power <= nominal.power


def test_a_loosening_rule_costs_other_recall_and_gains_accuracy() -> None:
    frame = frame_for(_population(), Rule(0.5))
    half = implied_changes(frame, TruthModel(**{**_TRUTH.__dict__, "slug_share": 0.5}))
    full = implied_changes(frame, TruthModel(**{**_TRUTH.__dict__, "slug_share": 1.0}))
    # the 200 articles at 0.5-0.6 stop being `other`; the more of them were really the slug,
    # the less `other` recall is lost and the more accuracy and precision are gained
    assert half.recall < full.recall <= 0.0  # at a share of 1 none of them was `other`
    assert half.precision < full.precision and half.accuracy < full.accuracy
    # a tightening rule does the opposite: more `other`, so more recall, less precision
    tight = implied_changes(frame_for(_population(), Rule(0.8)), _TRUTH)
    assert tight.recall > 0.0 and tight.precision < 0.0 and tight.accuracy < 0.0
    assert implied_changes(frame_for(_population(), BASELINE), _TRUTH) == (0.0, 0.0, 0.0)


def test_the_share_that_keeps_other_recall_within_tolerance_is_found_by_bisection() -> None:
    frame = frame_for(_population(), Rule(0.5))
    share = slug_share_for_recall(frame, _TRUTH)
    assert share is not None and 0.0 < share < 1.0
    at = implied_changes(frame, TruthModel(**{**_TRUTH.__dict__, "slug_share": share}))
    assert at.recall == pytest.approx(RECALL_TOLERANCE, abs=1e-3)
    # a bound that no share can reach, and one every share already meets
    assert slug_share_for_recall(frame, _TRUTH, bound=0.5) is None
    assert slug_share_for_recall(frame, _TRUTH, bound=-5.0) == 0.0


def test_the_joint_power_is_zero_when_the_recall_loss_breaks_the_tolerance() -> None:
    frame = frame_for(_population(), Rule(0.5))
    low = TruthModel(**{**_TRUTH.__dict__, "slug_share": 0.5})
    row = simulate_power(frame, low, draws=30, boot_reps=100, seed=2)
    assert row.recall_diff < RECALL_TOLERANCE and row.joint_power == 0.0
    # the precision test alone can still pass: that is the manager's point
    big = TruthModel(**{**_TRUTH.__dict__, "slug_share": 1.0})
    best = simulate_power(frame, big, draws=30, boot_reps=100, seed=2)
    assert best.joint_power <= best.power


def test_the_script_rejects_an_unknown_ramp_or_section_before_opening_a_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script()
    base = ["prog", "--source-db", "/nonexistent/urls.db", "--results-db", "/nonexistent/nlp.db"]
    for bad in (["--ramps", "graded,nope"], ["--sections", "power,oops"]):
        monkeypatch.setattr("sys.argv", [*base, *bad])
        with pytest.raises(SystemExit) as exc:
            script.main()  # type: ignore[attr-defined]
        assert exc.value.code == 2
