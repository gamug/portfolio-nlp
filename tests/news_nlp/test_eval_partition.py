"""Re-reading a stored category run per partition (TASKS.md T-133; the metric set of
SPEC.md FR-016 per partition): hermetic, no judge call."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

import news_nlp as db
from news_nlp.eval.metrics import aggregate_category
from news_nlp.eval.partition import (
    bootstrap_interval,
    domain_pairs,
    judge_label_counts,
    judged_counts,
    load_category_pairs,
    metrics_by_partition,
)
from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.verdicts import CategoryVerdict


def _pair(
    aid: int, bucket: str, pop: int, label: str, ideal: str
) -> tuple[EvalItem, CategoryVerdict]:
    item = EvalItem(
        article_id=aid,
        bucket=bucket,
        stratum_population=pop,
        title="",
        body_text="",
        prediction={"label": label, "distribution": {}},
    )
    return item, CategoryVerdict(agrees=label == ideal, ideal_slug=ideal)


def test_a_partition_that_is_the_whole_sample_reproduces_the_full_metrics() -> None:
    pairs = [
        _pair(1, "representative", 1000, "other", "other"),
        _pair(2, "representative", 1000, "other", "legal_regulatory"),
        _pair(3, "representative", 1000, "legal_regulatory", "legal_regulatory"),
        _pair(4, "representative", 1000, "earnings_performance", "other"),
    ]
    everyone = {i: "accept" for i in range(1, 5)}
    out = metrics_by_partition(pairs, everyone)
    for key in ("accuracy_vs_judge", "macro_f1", "precision_other", "recall_other"):
        assert out["accept"][key] == pytest.approx(out["all"][key])


def test_weights_stay_at_n_over_n_when_a_stratum_is_split() -> None:
    # stratum of 1000 articles sampled 4 times: 2 accept, 2 reject. Each item stands for
    # 250 articles in either view; the accept partition's population becomes 500.
    pairs = [_pair(i, "representative", 1000, "other", "other") for i in range(1, 5)]
    part = domain_pairs(pairs, lambda aid: aid <= 2)
    assert {it.stratum_population for it, _ in part} == {500}
    assert len(part) == 2
    # estimated total of the partition = (500 / 2) * 2 = 500 = 1000 * 2/4
    assert aggregate_category([it for it, _ in part], [v for _, v in part])["n"] == 2.0


def test_parse_failures_are_dropped_before_counting_the_stratum() -> None:
    ok = _pair(1, "representative", 900, "other", "other")
    bad = (
        _pair(2, "representative", 900, "other", "other")[0],
        CategoryVerdict(agrees=False, ideal_slug="other", parse_failed=True),
    )
    part = domain_pairs([ok, bad], lambda aid: True)
    assert [it.stratum_population for it, _ in part] == [900]


def test_low_conf_items_never_move_the_population_estimate() -> None:
    # low_conf is a diagnostic-only bucket: it is in the partition's items but the HT
    # estimator excludes it, exactly as in a live run.
    pairs = [
        _pair(1, "low_conf", 1000, "other", "legal_regulatory"),
        _pair(2, "representative", 1000, "other", "other"),
        _pair(3, "representative", 1000, "other", "other"),
    ]
    out = metrics_by_partition(pairs, {1: "accept", 2: "accept", 3: "reject"})
    assert out["accept"]["accuracy_vs_judge"] == pytest.approx(1.0)


def test_missing_partition_entry_is_none_and_counts_are_per_bucket() -> None:
    pairs = [
        _pair(1, "representative", 10, "other", "other"),
        _pair(2, "low_conf", 10, "other", "other"),
    ]
    assert judged_counts(pairs, {1: "accept"}) == {
        "(none)": {"low_conf": 1},
        "accept": {"representative": 1},
    }


def test_load_category_pairs_reads_a_stored_run(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "r.db")
    db.init_schema(conn)
    conn.execute(
        "INSERT INTO eval_run (id, stage, started_at, sample_size, low_conf_n, random_n, seed, "
        "judge_model, judge_url, code_version, metrics_json, status, strata_json, experiment) "
        "VALUES (7, 'category', 't', 1, 0, 1, 1, 'j', 'u', 'c', '{}', 'ok', ?, 'e')",
        (json.dumps({"representative": {"population": 123, "n": 1}}),),
    )
    conn.execute(
        "INSERT INTO eval_inference (id, run_id, article_id, task, experiment, bucket, prediction_json, created_at) "
        "VALUES (1, 7, 42, 'category', 'e', 'representative', ?, 't')",
        (json.dumps({"label": "other", "distribution": {}}),),
    )
    conn.execute(
        "INSERT INTO eval_verdict (inference_id, run_id, article_id, task, experiment, verdict_json, correct, judged_at) "
        "VALUES (1, 7, 42, 'category', 'e', ?, 1, 't')",
        (json.dumps({"agrees": True, "ideal_slug": "other"}),),
    )
    (pair,) = load_category_pairs(conn, 7)
    assert pair[0].article_id == 42 and pair[0].stratum_population == 123
    assert pair[1].ideal_slug == "other"
    conn.close()


def test_a_single_item_in_a_large_stratum_keeps_a_positive_weight() -> None:
    # 1 of 1000 sampled items is in the partition of a 10-article stratum:
    # round(10 * 1 / 1000) is 0, which would zero the item; it is clamped to 1.
    pairs = [_pair(i, "representative", 10, "other", "other") for i in range(1, 1001)]
    part = domain_pairs(pairs, lambda aid: aid == 1)
    assert [it.stratum_population for it, _ in part] == [1]


def test_judge_label_counts_match_the_estimator_inputs() -> None:
    bad = (
        _pair(3, "representative", 10, "other", "other")[0],
        CategoryVerdict(agrees=False, ideal_slug="legal_regulatory", parse_failed=True),
    )
    pairs = [
        _pair(1, "representative", 10, "other", "other"),
        _pair(2, "low_conf", 10, "other", "legal_regulatory"),
        bad,
        _pair(4, "representative", 10, "other", "legal_regulatory"),
    ]
    got = judge_label_counts(pairs, {1: "accept", 2: "accept", 3: "accept"})
    assert got == {"(none)": {"legal_regulatory": 1}, "accept": {"other": 1}}


def test_bootstrap_interval_brackets_the_estimate_and_is_deterministic() -> None:
    pairs = [
        _pair(i, "representative", 1000, "other", "other" if i % 3 else "legal_regulatory")
        for i in range(1, 61)
    ]
    part = {i: ("accept" if i <= 30 else "reject") for i in range(1, 61)}
    est, low, high, used, reps = bootstrap_interval(
        pairs, part, "precision_other", partition="accept", reps=80
    )
    assert (used, reps) == (80, 80)
    assert low <= est <= high
    assert bootstrap_interval(pairs, part, "precision_other", partition="accept", reps=80) == (
        est,
        low,
        high,
        used,
        reps,
    )
    diff, dlow, dhigh, _, _ = bootstrap_interval(
        pairs, part, "precision_other", minus=("accept", "reject"), reps=80
    )
    assert dlow <= diff <= dhigh


def test_load_category_pairs_names_a_missing_bucket_and_run(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "r.db")
    db.init_schema(conn)
    with pytest.raises(ValueError, match="not found"):
        load_category_pairs(conn, 99)
    conn.execute(
        "INSERT INTO eval_run (id, stage, started_at, sample_size, low_conf_n, random_n, seed, "
        "judge_model, judge_url, code_version, metrics_json, status, strata_json, experiment) "
        "VALUES (8, 'category', 't', 1, 0, 1, 1, 'j', 'u', 'c', '{}', 'ok', '{}', 'e')"
    )
    conn.execute(
        "INSERT INTO eval_inference (id, run_id, article_id, task, experiment, bucket, prediction_json, created_at) "
        "VALUES (1, 8, 42, 'category', 'e', 'representative', '{}', 't')"
    )
    conn.execute(
        "INSERT INTO eval_verdict (inference_id, run_id, article_id, task, experiment, verdict_json, correct, judged_at) "
        "VALUES (1, 8, 42, 'category', 'e', '{\"agrees\": true, \"ideal_slug\": \"other\"}', 1, 't')"
    )
    with pytest.raises(ValueError, match="missing from strata_json"):
        load_category_pairs(conn, 8)
    conn.close()


def test_the_measurement_script_maps_articles_to_their_upstream_class(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "measure_corrected_corpus_2026_10_22.py"
    )
    spec = importlib.util.spec_from_file_location("measure_script", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    urls = sqlite3.connect(tmp_path / "urls.db")
    urls.execute("CREATE TABLE discovered_urls (id INTEGER PRIMARY KEY, match_class TEXT)")
    urls.executemany(
        "INSERT INTO discovered_urls VALUES (?, ?)", [(1, "accept"), (2, "reject"), (3, None)]
    )
    assert module.load_partition_of(urls, [1, 2, 3, 4]) == {1: "accept", 2: "reject", 3: "(none)"}
    urls.close()


def test_bootstrap_skips_replicates_with_an_empty_denominator() -> None:
    # Only article 1 is predicted `other` in `accept`; a replicate that redraws it away
    # (or never draws it) has no predicted `other` there: precision is undefined, not 0.
    pairs = [_pair(1, "representative", 100, "other", "other")] + [
        _pair(i, "representative", 100, "legal_regulatory", "legal_regulatory")
        for i in range(2, 12)
    ]
    part = dict.fromkeys(range(1, 12), "accept")
    out = bootstrap_interval(pairs, part, "precision_other", partition="accept", reps=200)
    assert out.used < out.reps
    assert out.low == out.high == 1.0  # every usable replicate has precision 1
    none = bootstrap_interval(pairs[1:], part, "precision_other", partition="accept", reps=20)
    assert none.used == 0
