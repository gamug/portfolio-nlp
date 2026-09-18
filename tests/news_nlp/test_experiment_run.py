"""`experiment.run_experiment` end to end, hermetic: stubbed judge (same
pattern as `test_eval_runner.py`), a fake trainer (no real GPU work), and a
fake local-checkpoint model/tokenizer (same pattern as
`test_eval_candidate.py`'s own candidate-scoring tests). PLAN.md Work item
11 steps 1/3, TASKS.md T-099 (its own T-102-scoped test, shipped here
following T-096/T-098's precedent of shipping tests with the code).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("mlflow")
pytest.importorskip("strands")

import experiment
import sentiment_stage
import train_sentiment
from experiment import EvalSpec, ExperimentSpec, PretrainSpec, run_experiment
from fti import TrainedArtifact
from news_nlp.eval import runner
from news_nlp.eval.verdicts import CategoryVerdict, SentimentVerdict

_STUB_JUDGES = {
    "sentiment": lambda _a, it: SentimentVerdict(
        agrees=True, ideal_label="positive", severity=0, rationale="stub"
    ),
    "category": lambda _a, it: CategoryVerdict(
        agrees=True, ideal_slug=str(it.prediction["label"]), rationale="stub"
    ),
}


class FakeTokenizer:
    """Minimal stand-in: `.encode()` reports a fixed per-sentence token
    count (`chunk_text` only needs a count); `__call__` always returns the
    same one-token encoding regardless of input."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return ["x"] * 50

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return _FakeEncoding()


class _FakeEncoding(dict):
    def __init__(self) -> None:
        super().__init__(
            {
                "input_ids": torch.tensor([[0]], dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
            }
        )

    def to(self, device: Any) -> _FakeEncoding:
        return self


class FakeModel:
    """Always confidently positive, regardless of input -- this test only
    checks that run_experiment's own plumbing (train -> candidate-score ->
    write result) works, not prediction content."""

    def __init__(self) -> None:
        self.config = type(
            "Config", (), {"id2label": {0: "positive", 1: "negative", 2: "neutral"}}
        )()

    def to(self, device: Any) -> FakeModel:
        return self

    def eval(self) -> FakeModel:
        return self

    def __call__(self, **kwargs: Any) -> Any:
        logits = torch.tensor([[8.0, -8.0, -8.0]])
        return type("Output", (), {"logits": logits})()


@pytest.fixture
def stub_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "build_model", lambda _s: object())
    monkeypatch.setattr(runner, "build_judge_agent", lambda _m, _p: object())
    monkeypatch.setattr(runner, "JUDGES", _STUB_JUDGES)


@pytest.fixture
def stub_llm_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`EvalSettings.load()` (called inside `run_experiment` itself, not
    overridable via its own signature) needs these three -- set directly
    rather than stubbing `EvalSettings.load`, so the real `.load()` code
    path is exercised too."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "stub-model")
    monkeypatch.setenv("LLM_URL", "https://stub.test")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))


@pytest.fixture
def results_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """`experiment.RESULTS_DIR` is a module-level constant (a real,
    git-tracked path for a real run) -- redirected here so this test never
    writes into the actual repo working tree."""
    path = tmp_path / "experiments" / "results"
    monkeypatch.setattr(experiment, "RESULTS_DIR", path)
    return path


@pytest.mark.usefixtures("stub_judge", "stub_llm_env")
def test_run_experiment_eval_only_writes_result_file(
    tmp_path: Path, eval_store_paths: tuple[Path, Path], results_dir: Path
) -> None:
    source, results = eval_store_paths
    spec = ExperimentSpec(
        name="sentiment_eval_only_test",
        stage="sentiment",
        eval=EvalSpec(sample_size=6, low_conf_frac=0.5, seed=1, max_workers=2),
    )

    result = run_experiment(spec, source_db=str(source), results_db=str(results))

    assert result.train_output_dir is None
    assert result.train_metrics is None
    assert result.spec == spec
    assert set(result.eval_results) == {"sentiment"}
    assert result.eval_results["sentiment"]["n_judged"] == 6

    result_path = results_dir / "sentiment_eval_only_test.result.json"
    assert result_path == Path(result.result_path)
    on_disk = json.loads(result_path.read_text())
    assert on_disk["spec"]["name"] == "sentiment_eval_only_test"
    assert on_disk["eval_results"]["sentiment"]["n_judged"] == 6


@pytest.mark.usefixtures("stub_judge", "stub_llm_env")
def test_run_experiment_pretrain_trains_then_scores_the_fresh_checkpoint(
    tmp_path: Path,
    eval_store_paths: tuple[Path, Path],
    results_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, results = eval_store_paths
    checkpoint_dir = str(tmp_path / "fake-checkpoint")
    captured_configs: list[Any] = []

    def fake_train(self: Any, config: Any) -> TrainedArtifact:
        captured_configs.append(config)
        return TrainedArtifact(output_dir=checkpoint_dir, metrics={"eval_f1": 0.9})

    monkeypatch.setattr(train_sentiment.SentimentTrainer, "train", fake_train)
    monkeypatch.setattr(
        sentiment_stage.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeTokenizer()
    )
    monkeypatch.setattr(
        sentiment_stage.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_a, **_k: FakeModel(),
    )

    spec = ExperimentSpec(
        name="sentiment_pretrain_test",
        stage="sentiment",
        pretrain=PretrainSpec(
            enabled=True, base_model="ProsusAI/finbert", hyperparameters={"weighted": True}
        ),
        eval=EvalSpec(sample_size=5, low_conf_frac=0.5, seed=1, max_workers=2),
    )

    result = run_experiment(spec, source_db=str(source), results_db=str(results))

    assert len(captured_configs) == 1
    assert captured_configs[0].base_model == "ProsusAI/finbert"
    assert captured_configs[0].weighted is True
    assert result.train_output_dir == checkpoint_dir
    assert result.train_metrics == {"eval_f1": 0.9}
    # spec is echoed back unchanged -- candidate_model/candidate_revision
    # were never user-supplied (ExperimentSpec's own validator forbids it
    # when pretrain.enabled), and run_experiment's auto-resolution is an
    # internal detail, fully reconstructable from train_output_dir alone.
    assert result.spec.eval.candidate_model is None
    assert result.eval_results["sentiment"]["n_judged"] == 5

    result_path = results_dir / "sentiment_pretrain_test.result.json"
    assert result_path.exists()
