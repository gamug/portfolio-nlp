"""news_nlp.eval.store round-trip + news_nlp.latest_eval_runs."""

from __future__ import annotations

import news_nlp as db_module
from news_nlp.eval.store import create_eval_run, finish_eval_run, record_judgement


def test_eval_tables_exist_after_init_schema(conn: db_module.NewsNlpDatabase) -> None:
    cols = set(conn.table_columns("eval_run"))
    assert {"stage", "metrics_json", "strata_json", "mlflow_run_id", "status"} <= cols
    assert {"run_id", "bucket", "verdict_json", "correct"} <= set(
        conn.table_columns("eval_judgement")
    )


def test_run_and_judgement_round_trip(conn: db_module.NewsNlpDatabase) -> None:
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

    record_judgement(
        conn,
        run_id,
        article_id=1,
        bucket="low_conf",
        prediction={"label": "positive", "score": 0.31},
        verdict={"agrees": False, "ideal_label": "negative", "severity": 2},
        correct=False,
        severity=2,
        rationale="opposite call",
    )
    record_judgement(
        conn,
        run_id,
        article_id=2,
        bucket="representative",
        prediction={"label": "neutral", "score": 0.8},
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

    judgements = conn.execute(
        "SELECT article_id, bucket, correct, severity FROM eval_judgement "
        "WHERE run_id = ? ORDER BY article_id",
        (run_id,),
    ).fetchall()
    assert [tuple(r) for r in judgements] == [(1, "low_conf", 0, 2), (2, "representative", 1, 0)]


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
