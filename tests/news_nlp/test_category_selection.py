"""The selected category model (T-187): ``config/category_model.json`` and the gold files it names stay the ones it says.

A change to a tag file or to the recipe has to change the pin, so the figures in ``docs/category_train.md`` are not left describing
other data. Reads files only: no model, no GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import category_multilabel as ml
from news_category_stage import LABELS as L14

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads((ROOT / "config" / "category_model.json").read_text(encoding="utf-8"))
L12 = list(ml.merge_labels(tuple(L14), ml.FOLD_INTO_OTHER))


def _rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_the_labels_are_the_fourteen_with_the_two_folded_into_other() -> None:
    assert PIN["labels"] == L12
    assert len(L12) == 12
    assert sorted(set(L14) - set(L12)) == sorted(PIN["folded_into_other"])


def test_the_start_model_is_the_pinned_zero_shot_model() -> None:
    pipeline = json.loads((ROOT / "config" / "pipeline_models.json").read_text(encoding="utf-8"))[
        "category"
    ]
    assert PIN["start_model"] == {"repo": pipeline["model"], "revision": pipeline["revision"]}


def test_every_file_of_the_pin_is_the_file_measured() -> None:
    for key, entry in PIN["data"].items():
        if isinstance(entry, dict):
            digest = hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest()
            assert digest == entry["sha256"], (
                f"{key}: {entry['file']} changed after the figures were taken"
            )


def test_the_sheet_has_one_valid_tag_per_row() -> None:
    rows = _rows(PIN["data"]["sheet_single_tags"]["file"])
    assert len(rows) == PIN["data"]["sheet_single_tags"]["rows"]
    assert {r["label"] for r in rows} <= set(L12)
    assert len({(r["article_id"], r["company"]) for r in rows}) == len(
        rows
    )  # one row per (article, company)


def test_the_older_rows_review_keeps_one_tag_per_pair_and_drops_the_second() -> None:
    entry = PIN["data"]["older_rows_reviewed"]
    rows = _rows(entry["file"])
    assert len(rows) == entry["rows"]
    kept = [r for r in rows if r["label"]]
    assert len(kept) == entry["kept"]
    assert {r["label"] for r in kept} <= set(L12)
    assert len({(r["article_id"], r["company"]) for r in kept}) == len(kept)
    for r in rows:
        assert r["why"], f"row {r['row']} has no reason"


def test_the_training_rows_are_the_sheet_plus_the_kept_older_rows() -> None:
    d = PIN["data"]
    assert d["training_rows"] == d["sheet_single_tags"]["rows"] + d["older_rows_reviewed"]["kept"]


def test_the_held_out_file_is_named_but_not_a_training_input() -> None:
    assert "held_out_not_read" in PIN["data"]
    training = {PIN["data"][k]["file"] for k in ("sheet_single_tags", "older_rows_reviewed")}
    assert PIN["data"]["held_out_not_read"]["file"] not in training


def test_the_pin_says_it_is_not_fitted_and_not_wired() -> None:
    assert PIN["status"] == "selected, not fitted, not wired"
    assert "NOT shown" in PIN["evidence"]["bar"]
