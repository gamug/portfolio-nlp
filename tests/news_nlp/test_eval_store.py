"""news_nlp.eval.store round-trip + news_nlp.latest_eval_runs."""

from __future__ import annotations

import news_nlp as db_module
from news_nlp.eval.store import (
    create_eval_run,
    find_verdict_json,
    finish_eval_run,
    record_inference,
    record_verdict,
)


def test_eval_tables_exist_after_init_schema(conn: db_module.NewsNlpDatabase) -> None:
    cols = set(conn.table_columns("eval_run"))
    assert {"stage", "metrics_json", "strata_json", "mlflow_run_id", "status"} <= cols
    # eval_judgement is legacy (superseded by eval_inference/eval_verdict,
    # TASKS.md T-090) but its DDL/historical rows stay untouched.
    assert {"run_id", "bucket", "verdict_json", "correct"} <= set(
        conn.table_columns("eval_judgement")
    )
    assert {"run_id", "article_id", "task", "experiment", "bucket", "prediction_json"} <= set(
        conn.table_columns("eval_inference")
    )
    assert {
        "inference_id",
        "run_id",
        "article_id",
        "task",
        "experiment",
        "verdict_json",
        "correct",
    } <= set(conn.table_columns("eval_verdict"))


def test_run_inference_and_verdict_round_trip(conn: db_module.NewsNlpDatabase) -> None:
    run_id = create_eval_run(
        conn,
        stage="sentiment",
        sample_size=2,
        low_conf_n=1,
        random_n=1,
        seed=42,
        judge_model="deepseek-chat",
        judge_url="https://example.test",
        code_version="abc1234",
        strata_json='{"low_conf": {"population": 459112, "n": 1}, '
        '"representative": {"population": 245508, "n": 1}}',
    )
    conn.commit()
    assert isinstance(run_id, int)

    inference_id_1 = record_inference(
        conn,
        run_id,
        article_id=1,
        task="sentiment",
        experiment="base",
        bucket="low_conf",
        prediction={"label": "positive", "score": 0.31},
    )
    record_verdict(
        conn,
        inference_id_1,
        run_id,
        article_id=1,
        task="sentiment",
        experiment="base",
        verdict={"agrees": False, "ideal_label": "negative", "severity": 2},
        correct=False,
        severity=2,
        rationale="opposite call",
    )
    inference_id_2 = record_inference(
        conn,
        run_id,
        article_id=2,
        task="sentiment",
        experiment="base",
        bucket="representative",
        prediction={"label": "neutral", "score": 0.8},
    )
    record_verdict(
        conn,
        inference_id_2,
        run_id,
        article_id=2,
        task="sentiment",
        experiment="base",
        verdict={"agrees": True, "ideal_label": "neutral", "severity": 0},
        correct=True,
        severity=0,
        rationale="ok",
    )
    finish_eval_run(
        conn,
        run_id,
        metrics={"macro_f1_vs_judge": 0.5, "n": 2.0},
        mlflow_run_id="mlf123",
        status="ok",
    )
    conn.commit()

    row = conn.execute(
        "SELECT status, metrics_json, strata_json, mlflow_run_id, finished_at "
        "FROM eval_run WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert row["status"] == "ok"
    assert row["mlflow_run_id"] == "mlf123"
    assert row["finished_at"] is not None
    assert '"macro_f1_vs_judge": 0.5' in row["metrics_json"]
    assert '"representative": {"population": 245508, "n": 1}' in row["strata_json"]

    inferences = conn.execute(
        "SELECT article_id, task, experiment, bucket FROM eval_inference "
        "WHERE run_id = ? ORDER BY article_id",
        (run_id,),
    ).fetchall()
    assert [tuple(r) for r in inferences] == [
        (1, "sentiment", "base", "low_conf"),
        (2, "sentiment", "base", "representative"),
    ]

    verdicts = conn.execute(
        "SELECT article_id, task, experiment, correct, severity FROM eval_verdict "
        "WHERE run_id = ? ORDER BY article_id",
        (run_id,),
    ).fetchall()
    assert [tuple(r) for r in verdicts] == [
        (1, "sentiment", "base", 0, 2),
        (2, "sentiment", "base", 1, 0),
    ]


def test_latest_eval_runs_picks_newest_per_stage(conn: db_module.NewsNlpDatabase) -> None:
    common = {
        "sample_size": 1,
        "low_conf_n": 1,
        "random_n": 0,
        "seed": None,
        "judge_model": "m",
        "judge_url": "u",
        "code_version": "v",
    }
    old = create_eval_run(conn, stage="category", **common)
    finish_eval_run(conn, old, metrics={"accuracy_vs_judge": 0.6}, mlflow_run_id="old", status="ok")
    new = create_eval_run(conn, stage="category", **common)
    finish_eval_run(conn, new, metrics={"accuracy_vs_judge": 0.7}, mlflow_run_id="new", status="ok")
    create_eval_run(conn, stage="ner", **common)
    conn.commit()

    latest = db_module.latest_eval_runs(conn)
    by_stage = {r["stage"]: r for r in latest}
    assert set(by_stage) == {"category", "ner"}
    assert by_stage["category"]["mlflow_run_id"] == "new"
    assert by_stage["category"]["metrics"]["accuracy_vs_judge"] == 0.7
    assert by_stage["ner"]["status"] == "running"


def test_find_verdict_json_returns_the_matching_key_only(
    conn: db_module.NewsNlpDatabase,
) -> None:
    run_id = create_eval_run(
        conn,
        stage="sentiment",
        sample_size=1,
        low_conf_n=0,
        random_n=1,
        seed=None,
        judge_model="m",
        judge_url="u",
        code_version="v",
    )
    conn.commit()
    inference_id = record_inference(
        conn,
        run_id,
        article_id=1,
        task="sentiment",
        experiment="base",
        bucket="representative",
        prediction={"label": "positive", "score": 0.9},
    )
    record_verdict(
        conn,
        inference_id,
        run_id,
        article_id=1,
        task="sentiment",
        experiment="base",
        verdict={"agrees": True, "ideal_label": "positive", "severity": 0},
        correct=True,
        severity=0,
        rationale="ok",
    )
    conn.commit()

    found = find_verdict_json(conn, article_id=1, task="sentiment", experiment="base")
    assert found is not None
    assert '"ideal_label": "positive"' in found

    # A different article_id, task, or experiment must not match.
    assert find_verdict_json(conn, article_id=2, task="sentiment", experiment="base") is None
    assert find_verdict_json(conn, article_id=1, task="category", experiment="base") is None
    assert find_verdict_json(conn, article_id=1, task="sentiment", experiment="v2") is None
