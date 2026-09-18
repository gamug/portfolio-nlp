"""news_nlp.eval.runner.run_eval end-to-end with a stubbed judge + tmp MLflow store."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("mlflow")
pytest.importorskip("strands")

import news_nlp as db_module
import sentiment_stage
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
        (n_inf,) = check.execute("SELECT COUNT(*) FROM eval_inference").fetchone()
        assert n_inf == 12
        (n_verdict,) = check.execute("SELECT COUNT(*) FROM eval_verdict").fetchone()
        assert n_verdict == 12
        # No candidate_model/run_name given -> defaults to "base" (TASKS.md T-090).
        experiments = {
            r[0] for r in check.execute("SELECT DISTINCT experiment FROM eval_inference")
        }
        assert experiments == {"base"}
        # TASKS.md T-092: sentiment/category both get a confusion-matrix
        # row -- the stub judge always agrees with the model's own stored
        # label, and eval_store_paths seeds every article with the same
        # sentiment_label ("positive")/category_label ("earnings_performance"),
        # so each stage collapses to exactly one (true, predicted) cell.
        conf_rows = check.execute(
            "SELECT task, true_label, predicted_label, count FROM eval_confusion ORDER BY task"
        ).fetchall()
        assert [tuple(r) for r in conf_rows] == [
            ("category", "earnings_performance", "earnings_performance", 6),
            ("sentiment", "positive", "positive", 6),
        ]
    finally:
        check.close()

    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    assert len(mlflow.search_runs(experiment_names=["news_nlp_eval/sentiment"])) == 1


@pytest.mark.usefixtures("stub_judge")
def test_run_eval_writes_no_confusion_rows_for_ner_and_c_summary(
    tmp_path: Path, eval_store_paths: tuple[Path, Path]
) -> None:
    """TASKS.md T-092: ner/c_summary have no discrete predicted/ideal label
    shape (NerVerdict is error-only, SummaryVerdict is 1-5 scales), so
    confusion_pairs returns None for them -- confirm no eval_confusion rows
    land for either, even though eval_inference/eval_verdict do."""
    source, results = eval_store_paths
    settings = _settings(tmp_path)

    runner.run_eval(
        ["ner", "c_summary"], settings=settings, source_db=str(source), results_db=str(results)
    )

    check = db_module.connect(results)
    try:
        (n_inf,) = check.execute("SELECT COUNT(*) FROM eval_inference").fetchone()
        assert n_inf == 12  # both stages did get judged/recorded
        (n_conf,) = check.execute("SELECT COUNT(*) FROM eval_confusion").fetchone()
        assert n_conf == 0
    finally:
        check.close()


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


# --- judge-table reuse (TASKS.md T-091, SPEC.md FR-015) ---------------------


def _counting_sentiment_judge(counter: list[int]) -> Any:
    def judge(_agent: Any, _item: Any) -> SentimentVerdict:
        counter[0] += 1
        return SentimentVerdict(agrees=True, ideal_label="positive", severity=0, rationale="stub")

    return judge


def test_run_eval_reuses_verdicts_for_an_unchanged_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_store_paths: tuple[Path, Path]
) -> None:
    source, results = eval_store_paths
    monkeypatch.setattr(runner, "build_model", lambda _s: object())
    monkeypatch.setattr(runner, "build_judge_agent", lambda _m, _p: object())
    call_count = [0]
    monkeypatch.setattr(runner, "JUDGES", {"sentiment": _counting_sentiment_judge(call_count)})
    settings = _settings(tmp_path)

    runner.run_eval(
        ["sentiment"], settings=settings, source_db=str(source), results_db=str(results)
    )
    assert call_count[0] == settings.sample_size  # first run: every item judged fresh

    runner.run_eval(
        ["sentiment"], settings=settings, source_db=str(source), results_db=str(results)
    )
    # Same experiment ("base", both unset run_name/candidate_model), same
    # deterministic seeded sample -> every key already has a verdict, so
    # the judge is never called again.
    assert call_count[0] == settings.sample_size

    check = db_module.connect(results)
    try:
        (n_verdicts,) = check.execute("SELECT COUNT(*) FROM eval_verdict").fetchone()
    finally:
        check.close()
    # Both runs still each record a full eval_verdict row per item (history
    # preserved) -- reuse skips the LLM call, not the row.
    assert n_verdicts == settings.sample_size * 2


def test_run_eval_judges_fresh_for_a_different_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_store_paths: tuple[Path, Path]
) -> None:
    source, results = eval_store_paths
    monkeypatch.setattr(runner, "build_model", lambda _s: object())
    monkeypatch.setattr(runner, "build_judge_agent", lambda _m, _p: object())
    call_count = [0]
    monkeypatch.setattr(runner, "JUDGES", {"sentiment": _counting_sentiment_judge(call_count)})

    settings_a = _settings(tmp_path).model_copy(update={"run_name": "exp-a"})
    runner.run_eval(
        ["sentiment"], settings=settings_a, source_db=str(source), results_db=str(results)
    )
    assert call_count[0] == settings_a.sample_size

    settings_b = _settings(tmp_path).model_copy(update={"run_name": "exp-b"})
    runner.run_eval(
        ["sentiment"], settings=settings_b, source_db=str(source), results_db=str(results)
    )
    # A different experiment label for the same articles always judges fresh.
    assert call_count[0] == settings_a.sample_size + settings_b.sample_size


# --- candidate-model wiring (TASKS.md T-089, SPEC.md FR-013) ----------------


class _FakeCandidateEncoding(dict):
    def __init__(self) -> None:
        super().__init__(
            {
                "input_ids": torch.tensor([[0]], dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
            }
        )

    def to(self, device: Any) -> _FakeCandidateEncoding:
        return self


class _FakeCandidateTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return ["x"] * 50

    def __call__(self, text: str, **kwargs: Any) -> _FakeCandidateEncoding:
        return _FakeCandidateEncoding()


class _FakeCandidateModel:
    def __init__(self) -> None:
        self.config = type(
            "Config", (), {"id2label": {0: "positive", 1: "negative", 2: "neutral"}}
        )()

    def to(self, device: Any) -> _FakeCandidateModel:
        return self

    def eval(self) -> _FakeCandidateModel:
        return self

    def __call__(self, **kwargs: Any) -> Any:
        # Always confidently "negative" -- distinct from every fixture-seeded
        # row (all "positive", per conftest.write_stage_predictions' default)
        # so a judged prediction's label alone proves which source it came from.
        logits = torch.tensor([[-8.0, 8.0, -8.0]])
        return type("Output", (), {"logits": logits})()


@pytest.mark.usefixtures("stub_judge")
def test_run_eval_with_candidate_model_scores_from_scratch_not_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_store_paths: tuple[Path, Path]
) -> None:
    source, results = eval_store_paths
    monkeypatch.setattr(
        sentiment_stage.AutoTokenizer,
        "from_pretrained",
        lambda *_a, **_k: _FakeCandidateTokenizer(),
    )
    monkeypatch.setattr(
        sentiment_stage.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_a, **_k: _FakeCandidateModel(),
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "candidate_model": "candidate/model",
            "candidate_revision": "candidate-rev",
            "candidate_prescore_size": 10,
        }
    )

    out = runner.run_eval(
        ["sentiment"], settings=settings, source_db=str(source), results_db=str(results)
    )

    assert out["sentiment"]["n_judged"] == 6
    check = db_module.connect(results)
    try:
        rows = check.execute(
            "SELECT prediction_json, experiment FROM eval_inference WHERE run_id IN "
            "(SELECT id FROM eval_run WHERE stage = 'sentiment')"
        ).fetchall()
    finally:
        check.close()
    assert len(rows) == 6
    import json  # noqa: PLC0415

    # Every judged row's stored prediction came from the candidate model
    # (always "negative"), never from eval_store_paths' fixture data
    # (always "positive") -- proves sampling drew from the scratch
    # connection, not the production one.
    assert all(json.loads(r[0])["label"] == "negative" for r in rows)
    # experiment resolves to candidate_model when set (TASKS.md T-090).
    assert all(r[1] == "candidate/model" for r in rows)

    # Production article_sentiment (seeded by eval_store_paths) is untouched.
    (prod_score,) = (
        db_module.connect(results)
        .execute("SELECT score FROM article_sentiment WHERE article_id = 1")
        .fetchone()
    )
    assert prod_score == pytest.approx(0.315)


def test_run_eval_rejects_candidate_model_with_more_than_one_stage(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"candidate_model": "candidate/model", "candidate_revision": "rev"}
    )
    with pytest.raises(ValueError, match="requires exactly one stage"):
        runner.run_eval(["sentiment", "category"], settings=settings, source_db="x", results_db="y")


def test_run_eval_rejects_candidate_model_without_revision(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(update={"candidate_model": "candidate/model"})
    with pytest.raises(ValueError, match="candidate_revision is required"):
        runner.run_eval(["sentiment"], settings=settings, source_db="x", results_db="y")
