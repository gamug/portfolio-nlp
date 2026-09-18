"""news_nlp.eval.candidate.candidate_scored_connection + run_eval's
candidate-model wiring (TASKS.md T-089, SPEC.md FR-013)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

import news_nlp as db
import sentiment_stage
from news_nlp.eval import candidate
from news_nlp.eval.candidate import candidate_scored_connection


class FakeCandidateTokenizer:
    """Minimal stand-in: `.encode()` reports a fixed per-sentence token
    count (chunk_text only needs a count); `__call__` always returns the
    same one-token encoding regardless of input, since this test doesn't
    care which chunk produced which score -- only that scoring happened
    and landed in the right database."""

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


class FakeCandidateModel:
    """Always confidently positive, regardless of input -- this test only
    checks routing/plumbing (which DB the prediction lands in), not
    prediction content."""

    def __init__(self) -> None:
        self.config = type(
            "Config", (), {"id2label": {0: "positive", 1: "negative", 2: "neutral"}}
        )()

    def to(self, device: Any) -> FakeCandidateModel:
        return self

    def eval(self) -> FakeCandidateModel:
        return self

    def __call__(self, **kwargs: Any) -> Any:
        logits = torch.tensor([[8.0, -8.0, -8.0]])
        return type("Output", (), {"logits": logits})()


def _patch_sentiment_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sentiment_stage.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeCandidateTokenizer()
    )
    monkeypatch.setattr(
        sentiment_stage.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_a, **_k: FakeCandidateModel(),
    )


def test_candidate_scored_connection_writes_only_to_the_scratch_file(
    monkeypatch: pytest.MonkeyPatch, eval_store_paths: tuple[Path, Path]
) -> None:
    source, production_results = eval_store_paths
    _patch_sentiment_model(monkeypatch)

    captured_paths: list[str] = []
    real_mkstemp = candidate.tempfile.mkstemp

    def spy_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        fd, path = real_mkstemp(*args, **kwargs)
        captured_paths.append(path)
        return fd, path

    monkeypatch.setattr(candidate.tempfile, "mkstemp", spy_mkstemp)

    with candidate_scored_connection(
        str(source),
        "sentiment",
        model_name="candidate/model",
        revision="candidate-rev",
        limit=5,
        sample_seed=1,
    ) as conn:
        rows = conn.execute("SELECT article_id, label FROM article_sentiment").fetchall()
        assert len(rows) == 5
        assert all(r["label"] == "positive" for r in rows)

    # The scratch file is cleaned up once the context exits.
    assert len(captured_paths) == 1
    assert not os.path.exists(captured_paths[0])

    # Production results DB (a completely separate fixture path) is untouched --
    # eval_store_paths already seeded ids 1..20 with sentiment predictions; a
    # candidate-scoring pass against a *different* file must not add or change
    # any of them.
    prod_conn = db.connect(production_results)
    try:
        (n,) = prod_conn.execute("SELECT COUNT(*) FROM article_sentiment").fetchone()
        assert n == 20
        # eval_store_paths seeds article 1's score as 0.30 + 1*0.015 = 0.315
        # (a low-confidence row, distinct from FakeCandidateModel's score) --
        # unchanged proves the candidate run never wrote into this file.
        (score,) = prod_conn.execute(
            "SELECT score FROM article_sentiment WHERE article_id = 1"
        ).fetchone()
        assert score == pytest.approx(0.315)
    finally:
        prod_conn.close()


def test_candidate_scored_connection_rejects_unknown_stage(
    eval_store_paths: tuple[Path, Path],
) -> None:
    source, _results = eval_store_paths
    with (
        pytest.raises(ValueError, match="stage must be one of"),
        candidate_scored_connection(
            str(source),
            "sector_summary",
            model_name="x",
            revision="unused",
            limit=5,
            sample_seed=None,
        ),
    ):
        pass
