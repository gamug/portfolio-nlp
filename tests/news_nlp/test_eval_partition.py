"""Re-reading a stored category run per partition (T-133): hermetic, no judge call."""

import json
from pathlib import Path

import pytest

import news_nlp as db
from news_nlp.eval.metrics import aggregate_category
from news_nlp.eval.partition import (
    domain_pairs,
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
