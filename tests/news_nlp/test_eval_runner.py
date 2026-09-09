"""news_nlp.eval.runner.run_eval end-to-end with a stubbed judge + tmp MLflow store."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("strands")

import news_nlp as db_module
from news_nlp.eval import runner
from news_nlp.eval.config import EvalSettings
from news_nlp.eval.tracking import log_to_mlflow
from news_nlp.eval.verdicts import (
    CategoryVerdict,
    NerVerdict,
    SentimentVerdict,
    SummaryVerdict,
)

_STUB_JUDGES = {
    "sentiment": lambda _a, it: SentimentVerdict(
        agrees=True, ideal_label="positive", severity=0, rationale="stub"
    ),
    "category": lambda _a, it: CategoryVerdict(
        agrees=True, ideal_slug=str(it.prediction["label"]), rationale="stub"
    ),
    "ner": lambda _a, _it: NerVerdict(wrong=[], missed=[], rationale="stub"),
    "c_summary": lambda _a, _it: SummaryVerdict(
        faithfulness=4, coverage=4, conciseness=4, hallucinations=[], rationale="stub"
    ),
}


@pytest.fixture
def stub_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "build_model", lambda _s: object())
    monkeypatch.setattr(runner, "build_judge_agent", lambda _m, _p: object())
    monkeypatch.setattr(runner, "JUDGES", _STUB_JUDGES)


def _settings(tmp_path: Path) -> EvalSettings:
    return EvalSettings(
        llm_api_key="k",
        llm_model="stub-model",
        llm_url="https://stub.test",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        sample_size=6,
        low_conf_frac=0.5,
        seed=1,
        max_workers=2,
    )


@pytest.mark.usefixtures("stub_judge")
def test_run_eval_writes_db_rows_and_mlflow_runs(
    tmp_path: Path, eval_store_paths: tuple[Path, Path]
) -> None:
    source, results = eval_store_paths
    settings = _settings(tmp_path)

    out = runner.run_eval(
        ["sentiment", "category"],
        settings=settings,
        source_db=str(source),
        results_db=str(results),
    )

    assert set(out) == {"sentiment", "category"}
    for r in out.values():
        assert r["n_judged"] == 6
        assert isinstance(r["eval_run_id"], int)
        assert r["mlflow_run_id"]
        assert r["headline_metric"] in r["metrics"]
        assert r["regressed"] is False
    assert out["sentiment"]["headline_metric"] == "recall_negative"

    check = db_module.connect(results)
    try:
        runs = check.execute("SELECT stage, status FROM eval_run ORDER BY stage").fetchall()
        assert [tuple(x) for x in runs] == [("category", "ok"), ("sentiment", "ok")]
        (n_j,) = check.execute("SELECT COUNT(*) FROM eval_judgement").fetchone()
        assert n_j == 12
    finally:
        check.close()

    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    assert len(mlflow.search_runs(experiment_names=["news_nlp_eval/sentiment"])) == 1


@pytest.mark.usefixtures("stub_judge")
def test_run_eval_exits_nonzero_on_regression(
    tmp_path: Path, eval_store_paths: tuple[Path, Path]
) -> None:
    source, results = eval_store_paths
    settings = _settings(tmp_path)
    # A strong prior recall_negative so the stub's 0.0 (it only ever returns
    # ideal_label="positive", so there's no true negative for it to recall)
    # looks like a drop.
    log_to_mlflow(
        stage="sentiment",
        params={"stage": "sentiment"},
        metrics={"recall_negative": 0.90},
        judgements=[],
        system_prompt="p",
        tracking_uri=settings.mlflow_tracking_uri,
    )
    time.sleep(0.05)

    with pytest.raises(SystemExit) as exc:
        runner.run_eval(
            ["sentiment"],
            settings=settings,
            source_db=str(source),
            results_db=str(results),
            check_regression=True,
            regression_tolerance=0.05,
        )
    assert exc.value.code == 1
