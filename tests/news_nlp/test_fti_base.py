"""Structural tests for the FTI base classes (`src/fti.py`) in isolation --
no real stage, no real HF model, no real DB connection. TASKS.md T-082 is
scoped to designing/writing these base classes only; migrating a real stage
onto them (T-083-T-086) gets its own, separately-tested follow-up work."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from fti import Feature, FeatureBatch, Inference, NoOpTrainer, TrainConfig, TrainedArtifact, Trainer


class _EchoFeature(Feature[str, str]):
    def extract_one(self, tokenizer: Any, row: str) -> str:
        return row.upper()


def test_feature_extract_one_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        Feature().extract_one(None, "x")


def test_feature_extract_batch_default_loops_extract_one_in_order() -> None:
    batch = _EchoFeature().extract_batch(None, ["a", "b"])
    assert batch == FeatureBatch(items=["A", "B"])


def test_trainer_base_train_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        Trainer().train(TrainConfig())


def test_no_op_trainer_returns_no_output_dir() -> None:
    artifact = NoOpTrainer().train(TrainConfig())
    assert artifact == TrainedArtifact(output_dir=None, metrics=None)


class _FakeConn:
    """Minimal stand-in for `NewsNlpDatabase` -- `run()` calls `conn.commit()`
    once per batch after `write_predictions`, so any test that reaches a
    real batch needs a `conn` object with that method, even though the fake
    `write_predictions` above never actually writes anywhere."""

    def commit(self) -> None:
        pass


class _FakeInference(Inference[str, str]):
    MODEL_NAME = "fake/model"
    MODEL_REVISIONS: ClassVar[dict[str, str]] = {"fake/model": "abc123"}
    STAGE_NAME = "fake"

    def fetch_pending(self, conn: Any, limit: int | None, sample_seed: int | None) -> list[str]:
        return self._rows

    def load_model(self) -> None:
        self.model = "loaded-model"
        self.tokenizer = "loaded-tokenizer"

    def predict_batch(self, features: FeatureBatch[str]) -> list[Any]:
        return list(features.items)

    def write_predictions(self, conn: Any, rows: Any, predictions: list[Any]) -> None:
        self.written = predictions


def test_inference_run_loads_and_frees_model_and_reports_progress() -> None:
    inf = _FakeInference(_EchoFeature())
    inf._rows = ["a"]
    calls: list[tuple[str, int, int]] = []

    inf.run(conn=_FakeConn(), on_progress=lambda *a: calls.append(a))

    assert inf.written == ["A"]
    assert inf.model is None  # freed after run()
    assert inf.tokenizer is None
    assert calls == [("fake", 0, 1), ("fake", 1, 1)]


def test_inference_run_skips_load_model_when_nothing_pending() -> None:
    inf = _FakeInference(_EchoFeature())
    inf._rows = []

    def fail_if_called() -> None:
        raise AssertionError("model should not be loaded when there is nothing to process")

    inf.load_model = fail_if_called  # type: ignore[method-assign]
    calls: list[tuple[str, int, int]] = []

    inf.run(conn=None, on_progress=lambda *a: calls.append(a))

    assert calls == [("fake", 0, 0)]


def test_inference_run_frees_model_even_if_predict_batch_raises() -> None:
    class _BoomInference(_FakeInference):
        def predict_batch(self, features: FeatureBatch[str]) -> list[Any]:
            raise RuntimeError("boom")

    inf = _BoomInference(_EchoFeature())
    inf._rows = ["a"]

    with pytest.raises(RuntimeError, match="boom"):
        inf.run(conn=None)

    assert inf.model is None  # still freed, per run()'s try/finally
    assert inf.tokenizer is None
