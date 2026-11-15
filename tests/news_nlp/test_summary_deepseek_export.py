"""``export`` of the DeepSeek figures (T-206): what it says when the scratch is not ready, and what it writes when it is. No text in the output."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _rep() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rep2", ROOT / "scripts" / "summary_deepseek_report_2026_11_15.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scratch(tmp_path: Path, mod: ModuleType, *, with_d: bool = True, judged: bool = True) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    sample = [
        {"article_id": i, "tier": 1, "language": "en", "body": "b" * 100, "title": "t"}
        for i in (1, 2)
    ]
    arms = [
        {
            "article_id": i,
            "arms": {"B": f"b{i}", **({"D": f"d{i}"} if with_d else {})},
            "arm_chunks": {},
        }
        for i in (1, 2)
    ]
    (results / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    (results / "arms.json").write_text(json.dumps(arms), encoding="utf-8")
    (results / "deepseek.json").write_text(
        json.dumps(
            [
                {"article_id": i, "summary": f"d{i}", "in": 10, "out": 3, "model": "m1"}
                for i in (1, 2)
            ]
        ),
        encoding="utf-8",
    )
    verdict = {"faithfulness": 5, "coverage": 4, "conciseness": 5, "hallucinations": []}
    lines = [
        json.dumps({"article_id": i, "sha": mod.sha(s), "verdict": verdict})
        for i in (1, 2)
        for s in (f"b{i}", f"d{i}")
    ]
    (results / "verdicts.jsonl").write_text(
        "\n".join(lines[: (4 if judged else 2)]), encoding="utf-8"
    )
    return results


def test_export_says_which_step_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _rep()
    monkeypatch.setattr(mod, "OUT", tmp_path / "out")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(SystemExit, match=r"no arm D.*generate.*attach"):
        mod.export(_scratch(tmp_path / "a", mod, with_d=False))
    with pytest.raises(SystemExit, match=r"no verdict.*judge"):
        mod.export(_scratch(tmp_path / "b", mod, judged=False))


def test_export_writes_scores_and_the_model_and_no_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _rep()
    out = tmp_path / "out"
    monkeypatch.setattr(mod, "OUT", out)
    (tmp_path / "c").mkdir()
    mod.export(_scratch(tmp_path / "c", mod))
    with (out / "deepseek_vs_current.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["D_model"] == "m1" and rows[0]["D_tokens_in"] == "10"
    text = (out / "deepseek_vs_current.csv").read_text(encoding="utf-8")
    assert (
        "d1" not in text.replace("D_", "") and "b1" not in text
    )  # the summaries are not in the file
