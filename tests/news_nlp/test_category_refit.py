"""The refit and the scoring of the kept weights (T-201, T-202): what can be tested without a GPU. The refit refuses data that is not the pinned data and a recipe
that is not the pin's, and a score refuses a folder that is not a refit folder or whose weights were replaced."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import category_multilabel as ml

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads((ROOT / "config" / "category_model.json").read_text(encoding="utf-8"))


def test_the_pinned_data_files_match_the_pin() -> None:
    assert ml.data_hash_problems(PIN, ROOT) == []


def test_an_edited_or_missing_data_file_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "gold.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    pin: dict[str, object] = {
        "data": {"sheet": {"file": "gold.csv", "sha256": ml.sha256_file(f)}, "note": {"rows": 3}}
    }
    assert ml.data_hash_problems(pin, tmp_path) == []
    f.write_text("a,b\n1,3\n", encoding="utf-8")
    assert ml.data_hash_problems(pin, tmp_path) == ["gold.csv: the sha256 is not the pinned one"]
    f.unlink()
    assert ml.data_hash_problems(pin, tmp_path) == ["gold.csv: missing"]


def test_the_hyperparameters_come_from_the_pin() -> None:
    hyper = ml.recipe_hyperparameters(PIN)
    assert hyper == {"epochs": 3, "lr": 2e-5, "neg": 4, "bs": 4, "accum": 4, "max_len": 512}
    with pytest.raises(ValueError, match="negatives"):
        ml.recipe_hyperparameters(
            {
                "recipe": {
                    k: 1
                    for k in (
                        "epochs",
                        "learning_rate",
                        "batch_size",
                        "gradient_accumulation",
                        "max_length",
                    )
                }
            }
        )


def test_a_folder_without_a_record_or_with_other_weights_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a refit folder"):
        ml.check_refit_folder(tmp_path)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    record = {"model_safetensors_sha256": hashlib.sha256(b"weights").hexdigest(), "seed": 100}
    (tmp_path / "refit_record.json").write_text(json.dumps(record), encoding="utf-8")
    assert ml.check_refit_folder(tmp_path)["seed"] == 100
    weights.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="not the file the record names"):
        ml.check_refit_folder(tmp_path)


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_score_script_refuses_a_folder_that_is_not_a_refit_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _script("category_score_2026_11_15")
    rows = tmp_path / "rows.csv"
    with rows.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["article_id", "company", "text"], ["1", "Co", "x"]])
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--rows", str(rows), "--out", str(tmp_path / "o.csv"), "--folder", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="refused"):
        mod.main()
    monkeypatch.setattr(sys, "argv", ["x", "--heldout", "--folder", str(tmp_path)])
    with pytest.raises(
        SystemExit, match="refused"
    ):  # no refit folder: refused before anything of the held-out is read
        mod.main()


def test_the_refit_refuses_flags_that_are_not_the_pins() -> None:
    """The defaults of the command line are the pin's values (so a refit with no flag is the pin's recipe), and a flag that differs is found."""
    mod = _script("category_single_cv_2026_11_15")
    hyper = ml.recipe_hyperparameters(PIN)
    args = mod.build_parser().parse_args(["refit"])
    assert ml.refit_flag_problems(hyper, vars(args)) == {}
    changed = mod.build_parser().parse_args(["refit", "--lr", "3e-5", "--epochs", "4"])
    assert ml.refit_flag_problems(hyper, vars(changed)) == {"epochs": 4, "lr": 3e-5}


def test_the_body_overlap_table_is_the_one_of_the_committed_predictions() -> None:
    """``body_overlap.md`` quotes the accuracy with and without the rows of the body shared across folds; both are recomputable from the committed predictions."""
    folder = ROOT / "docs" / "category_results_2026_11_15"
    text = (folder / "body_overlap.md").read_text(encoding="utf-8")
    with (folder / "single_predictions.csv").open(encoding="utf-8", newline="") as f:
        pred = list(csv.DictReader(f))
    shared = {116568, 439320}  # the two articles of the one body, as the table names them
    assert "[116568, 439320]" in text
    rest = [p for p in pred if int(p["article_id"]) not in shared]
    hit = [p for p in pred if int(p["article_id"]) in shared]
    assert f"rows in them: {len(hit)} of {len(pred)}" in text
    assert (
        f"accuracy on all rows {sum(p['gold'] == p['top'] for p in pred) / len(pred):.4f}" in text
    )
    assert (
        f"without those rows ({len(rest)}) {sum(p['gold'] == p['top'] for p in rest) / len(rest):.4f}"
        in text
    )
