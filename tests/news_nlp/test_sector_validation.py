"""The sector step's two yes/no reviews, tracked in one file (owner, 2026-11-13): the headline figures of ``docs/sector_step.md`` come from it."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _rows() -> list[dict[str, str]]:
    with (ROOT / "gold" / "sector_validation.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_file_holds_both_reviews_with_one_answer_per_body_and_no_body_twice() -> None:
    rows = _rows()
    assert sum(r["review"] == "first" for r in rows) == 110
    assert sum(r["review"] == "fresh" for r in rows) == 57
    assert all(r["correct"] in ("yes", "no") for r in rows)
    ids = [r["article_id"] for r in rows]
    assert len(ids) == len(set(ids))  # the clean test shares no body with the first review


def test_every_body_has_the_text_the_owner_read() -> None:
    with (ROOT / "gold" / "texts" / "sector_validation_texts.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        texts = {r["article_id"]: r for r in csv.DictReader(fh)}
    assert {r["article_id"] for r in _rows()} == set(texts)
    assert all(t["text"].strip() and t["title"].strip() for t in texts.values())
    assert "spanish_text" not in next(iter(texts.values()))  # the translation aid is not tracked


def test_the_headline_figures_of_the_clean_test() -> None:
    fresh = [r for r in _rows() if r["review"] == "fresh"]
    assigned = [r for r in fresh if r["status"] == "assigned" and r["control"] == "no"]
    controls = [r for r in fresh if r["control"] == "yes"]
    dropped = [r for r in fresh if r["status"] == "out_of_scope"]
    assert (len(assigned), len(controls), len(dropped)) == (40, 5, 12)
    assert sum(r["correct"] == "yes" for r in assigned) == 35  # 87.5%
    assert (
        sum(r["correct"] == "yes" for r in controls) == 0
    )  # a reviewer that says yes too soon would fail here
    assert sum(r["correct"] == "yes" for r in dropped) == 7
    assert {r["step_revision"] for r in fresh} == {"sector-cf81d9b32565"}


def test_the_first_review_keeps_its_figures() -> None:
    first = [r for r in _rows() if r["review"] == "first"]
    real = [r for r in first if r["control"] == "no"]
    assert len(real) == 100 and sum(r["correct"] == "yes" for r in real) == 67
    assert sum(r["correct"] == "yes" for r in first if r["control"] == "yes") == 0
    sure = [r for r in real if r["sure"] == "yes"]
    assert len(sure) == 49 and sum(r["correct"] == "yes" for r in sure) == 43
