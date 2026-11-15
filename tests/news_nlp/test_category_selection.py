"""The selected category model (T-187): ``config/category_model.json`` and the gold files it names stay the ones it says.

A change to a tag file or to the recipe has to change the pin, so the figures in ``docs/category_training.md`` are not left describing
other data. Reads files only: no model, no GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import category_multilabel as ml
import news_category_stage as stage
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


def _scores(name: str) -> tuple[list[dict[str, str]], np.ndarray]:
    folder = ROOT / ml.COMMITTED_SCORES / name
    with (folder / "rows_single.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, np.load(folder / "oof_single.npz")["scores"]


def _accuracy_and_macro_f1(gold: list[str], top: list[str]) -> tuple[float, float]:
    per = ml.per_label([[g] for g in gold], [[t] for t in top], L12)
    f1 = []
    for label in L12:
        p, r = per[label]["precision"] or 0.0, per[label]["recall"] or 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean([g == t for g, t in zip(gold, top, strict=True)])), float(np.mean(f1))


def test_the_evidence_of_the_pin_is_what_the_committed_scores_give() -> None:
    rows, scores = _scores("category_single")
    assert len(rows) == PIN["data"]["training_rows"]
    top = [L12[j] for j in scores.argmax(1)]
    acc, f1 = _accuracy_and_macro_f1([r["gold"] for r in rows], top)
    assert round(acc, 3) == PIN["evidence"]["accuracy"]
    assert round(f1, 3) == PIN["evidence"]["macro_f1"]


def test_the_untouched_model_figures_are_on_the_rows_it_was_scored_on() -> None:
    rows, _ = _scores("category_single")
    folder = ROOT / ml.COMMITTED_SCORES / "category_multilabel"
    with (folder / "rows.csv").open(encoding="utf-8", newline="") as f:
        old_rows = list(csv.DictReader(f))
    base = ml.merge_scores(
        np.load(folder / "oof_base.npz")["scores"], tuple(L14), ml.FOLD_INTO_OTHER
    )
    index = {(r["article_id"], r["company"]): i for i, r in enumerate(old_rows)}
    pairs = [
        (i, index[(r["article_id"], r["company"])])
        for i, r in enumerate(rows)
        if (r["article_id"], r["company"]) in index
    ]
    claim = PIN["evidence"]["untouched_on_the_same_rows_and_tags"]
    assert len(pairs) == claim["rows"]
    left_out = {(r["article_id"], r["company"]) for r in rows} - set(index)
    assert (
        len(left_out) == len(rows) - claim["rows"]
    )  # exactly the rows the earlier cross-validation never saw
    acc, f1 = _accuracy_and_macro_f1(
        [rows[i]["gold"] for i, _ in pairs], [L12[int(base[j].argmax())] for _, j in pairs]
    )
    assert round(acc, 3) == claim["accuracy"]
    assert round(f1, 3) == claim["macro_f1"]


def test_the_older_rows_review_lines_up_with_the_rows_of_the_earlier_runs() -> None:
    folder = ROOT / ml.COMMITTED_SCORES / "category_multilabel"
    with (folder / "rows.csv").open(encoding="utf-8", newline="") as f:
        v2 = [r for r in csv.DictReader(f) if r["source"] == "v2"]
    review = _rows(PIN["data"]["older_rows_reviewed"]["file"])
    assert len(v2) == len(review)
    for i, (old, new) in enumerate(zip(v2, review, strict=True)):
        assert (old["article_id"], old["company"]) == (new["article_id"], new["company"]), (
            f"row {i}"
        )
        assert ml.FOLD_INTO_OTHER.get(old["gold"], old["gold"]) == new["label_old"], f"row {i}"


def test_every_change_of_the_sheet_has_its_review_comment() -> None:
    rows = _rows(PIN["data"]["sheet_single_tags"]["file"])
    assert {"label", "label_owner", "review"} <= set(rows[0])
    for r in rows:
        owner = r["label_owner"].strip()
        if r["label"] != ml.FOLD_INTO_OTHER.get(owner, owner):
            assert r["review"], f"review order {r['review_order']}: the tag changed with no comment"


def test_the_reading_of_the_pin_is_the_one_of_the_code() -> None:
    reading = {
        "definitions": stage.DEFINITIONS,
        "groups": {k: list(v) for k, v in stage.GROUPS.items()},
        "hypothesis": stage.HYPOTHESIS,
        "premise_chars": stage.PREMISE_CHARS,
    }
    digest = hashlib.sha256(
        json.dumps(reading, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert digest == PIN["recipe"]["reading_sha256"], (
        "a definition, a group, the hypothesis or the premise changed: it is another model"
    )


def test_the_refit_has_a_fixed_seed() -> None:
    """One refit, one seed: the weights that are scored on the held-out are the ones that get wired (T-201)."""
    assert PIN["recipe"]["refit_seed"] == 0
    assert PIN["recipe"]["cv_seed"] == 0


def test_the_review_audit_draws_the_changed_rows_blind_and_by_seed() -> None:
    """T-200: the second reader's sample comes from the 226 rows whose tag the review changed,"""
    import importlib.util  # noqa: PLC0415

    path = ROOT / "scripts" / "draw_category_review_audit_2026_11_15.py"
    spec = importlib.util.spec_from_file_location("draw_audit", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with (ROOT / "gold" / "category_single_gold.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    changed = mod.changed_rows(rows)
    assert len(changed) == 226
    assert all(mod.first_tag(r["label_owner"]) != r["label"] for r in changed)
