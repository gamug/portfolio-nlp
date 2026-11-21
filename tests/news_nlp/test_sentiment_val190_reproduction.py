"""The step's sentiment reading reproduces the stored labels of the 190 held-out pieces: the pinned pair model (its local folder,
`models/sentiment_pair_all/final`, identity 685a5f64f427a4ff) read with the step's own code and the stack of `config/sentiment_stack.json`,
against the `predicted` column of the version-2 gold. It needs the model folder, which is git-ignored, so it is skipped without it (CI).
Known mismatch, kept on purpose: `sn289407#CAT` (the stack gives negative 0.545 against positive 0.453; the stored label is positive; the
cause is not diagnosed). The folder's identity is checked against `config/sentiment_step.json` first, as the step does, and the model runs on
CPU (the device of the recorded run; a GPU may round differently)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
GOLD = ROOT / "gold" / "sentiment_val190_2026_11_20_v2.csv"
TEXTS = ROOT / "gold" / "texts" / "sentiment_val190_texts.csv"
MODEL = ROOT / "models" / "sentiment_pair_all" / "final"


@pytest.mark.skipif(
    not MODEL.is_dir(), reason="the pinned pair model folder is not here (git-ignored, local only)"
)
def test_the_step_reading_reproduces_the_stored_labels() -> None:
    import news_sentiment_stage as step  # noqa: PLC0415 -- loads torch

    with GOLD.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with TEXTS.open(encoding="utf-8-sig", newline="") as fh:
        texts = {r["key"]: r for r in csv.DictReader(fh)}
    import torch  # noqa: PLC0415 -- loads torch

    pinned = step.pin()["identity"]
    scorer = step.SentimentScorer(folder=MODEL, device=torch.device("cpu"))
    assert scorer.identity() == pinned, (scorer.identity(), pinned)
    probs = scorer.probabilities(
        [(texts[r["key"]]["text"], texts[r["key"]]["company"]) for r in rows]
    )
    coefficients = step.stack_coefficients()
    mismatch = []
    for r, pr in zip(rows, probs, strict=True):
        assert pr is not None, r["key"]
        mean, _chunks = pr
        stacked = step.stacked(coefficients, mean, float(r["relevance_score"]))
        label = max(step.CLASSES, key=lambda c: stacked[c])
        if label != r["predicted"]:
            mismatch.append(r["key"])
    assert mismatch == ["sn289407#CAT"]
