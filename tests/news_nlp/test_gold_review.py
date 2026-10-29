"""news_nlp.gold_review: the review file with the owner's labels is never overwritten."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from news_nlp import gold_review
from news_nlp.gold_review import TaggedFileError, tagged_rows, write_review_files

REVIEW = [
    {"review_order": 1, "article_id": 10, "title": "A", "human_label": "", "human_notes": ""},
    {"review_order": 2, "article_id": 11, "title": "B", "human_label": "", "human_notes": ""},
]
KEY = [
    {"review_order": 1, "article_id": 10, "llm_label": "other"},
    {"review_order": 2, "article_id": 11, "llm_label": "legal_regulatory"},
]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_a_fresh_export_writes_both_files(tmp_path: Path) -> None:
    write_review_files(tmp_path, REVIEW, KEY)
    assert [r["article_id"] for r in _read(tmp_path / "category_gold_review.csv")] == ["10", "11"]
    assert [r["article_id"] for r in _read(tmp_path / "category_gold_review_key.csv")] == [
        "10",
        "11",
    ]
    assert tagged_rows(tmp_path / "category_gold_review.csv") == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_an_untagged_review_file_can_be_rewritten(tmp_path: Path) -> None:
    write_review_files(tmp_path, REVIEW, KEY)
    write_review_files(tmp_path, REVIEW[:1], KEY[:1])
    assert len(_read(tmp_path / "category_gold_review.csv")) == 1


def test_a_review_file_with_labels_is_never_overwritten(tmp_path: Path) -> None:
    write_review_files(tmp_path, REVIEW, KEY)
    path = tmp_path / "category_gold_review.csv"
    rows = _read(path)
    rows[0]["human_label"] = "market_analyst_sentiment"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_bytes()
    key_before = (tmp_path / "category_gold_review_key.csv").read_bytes()
    assert tagged_rows(path) == 1
    with pytest.raises(TaggedFileError, match="1 human labels"):
        write_review_files(tmp_path, REVIEW, KEY)
    assert path.read_bytes() == before  # the owner's work is untouched
    assert (tmp_path / "category_gold_review_key.csv").read_bytes() == key_before


def test_a_failure_while_writing_leaves_neither_file_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_review_files(tmp_path, REVIEW, KEY)
    review_before = (tmp_path / "category_gold_review.csv").read_bytes()
    key_before = (tmp_path / "category_gold_review_key.csv").read_bytes()
    real = gold_review._write
    calls = {"n": 0}

    def flaky(path: Path, rows: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:  # the key file
            raise OSError("disk full")
        real(path, rows)  # type: ignore[arg-type]

    monkeypatch.setattr(gold_review, "_write", flaky)
    with pytest.raises(OSError, match="disk full"):
        write_review_files(tmp_path, REVIEW[:1], KEY[:1])
    assert (tmp_path / "category_gold_review.csv").read_bytes() == review_before
    assert (tmp_path / "category_gold_review_key.csv").read_bytes() == key_before
    assert not list(tmp_path.glob("*.tmp"))
