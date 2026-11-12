"""The sample of the cut the owner reads (T-183): bodies drawn from one named cut, one row per sub-news, the yes/no questions empty."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import draw_cut_sample_2026_11_12 as sample

from news_nlp import triage

REV = "units-aaaaaaaaaaaa"


def _body(n: int) -> str:
    return f"Body number {n}. " + "Company news sentence. " * 5


@pytest.fixture
def pair(tmp_path: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    results = sqlite3.connect(tmp_path / "nlp.db")
    results.execute(
        "CREATE TABLE sub_news (id INTEGER PRIMARY KEY, body_hash TEXT, article_id INTEGER, "
        "segment_index INTEGER, text TEXT, extractor_revision TEXT, processed_at TEXT)"
    )
    source = sqlite3.connect(tmp_path / "urls.db")
    source.execute(
        "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, body_text TEXT, company TEXT, "
        "ticker TEXT, pub_date TEXT, source_domain TEXT)"
    )
    sub_id = 0
    for n in range(1, 11):  # ten bodies, three pieces each, in the cut REV
        body = _body(n)
        source.execute(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
            (n, f"Title {n}", body, f"Company {n}", f"T{n}", "2026-01-01", "example.com"),
        )
        for seg in (2, 0, 1):  # inserted out of order: the sample puts them in segment order
            sub_id += 1
            results.execute(
                "INSERT INTO sub_news VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sub_id, triage.body_hash(body), n, seg, f"Piece {seg} of {n}.", REV, "t"),
            )
    results.execute(  # another cut of body 1: never read
        "INSERT INTO sub_news VALUES (999, ?, 1, 0, 'old cut', 'units4-x', 't')",
        (triage.body_hash(_body(1)),),
    )
    results.commit()
    source.commit()
    return results, source


def test_the_draw_is_seeded_and_reads_only_the_named_cut(
    pair: tuple[sqlite3.Connection, sqlite3.Connection],
) -> None:
    results, source = pair
    first = sample.draw(results, source, REV, bodies=4, seed=1)
    again = sample.draw(results, source, REV, bodies=4, seed=1)
    other = sample.draw(results, source, REV, bodies=4, seed=2)
    assert [b.article_id for b in first] == [b.article_id for b in again]
    assert [b.article_id for b in first] != [b.article_id for b in other]
    assert len(first) == 4 and len({b.article_id for b in first}) == 4
    for body in first:
        assert [p.segment_index for p in body.pieces] == [0, 1, 2]
        assert all(p.text.startswith("Piece") for p in body.pieces)  # the 'old cut' never appears
    assert sample.draw(results, source, "units-nothing", bodies=4, seed=1) == []


def test_a_body_whose_text_changed_since_the_cut_is_left_out_and_counted(
    pair: tuple[sqlite3.Connection, sqlite3.Connection],
) -> None:
    results, source = pair
    source.execute("UPDATE articles SET body_text = 'Rewritten.' WHERE id = 3")
    source.commit()
    drawn = sample.draw(results, source, REV, bodies=10, seed=1)
    assert 3 not in {b.article_id for b in drawn} and len(drawn) == 9


def test_more_bodies_than_the_cut_has_gives_all_of_them(
    pair: tuple[sqlite3.Connection, sqlite3.Connection],
) -> None:
    results, source = pair
    assert len(sample.draw(results, source, REV, bodies=500, seed=1)) == 10


def test_the_rows_have_one_row_per_sub_news_the_questions_empty_and_the_spanish_columns(
    pair: tuple[sqlite3.Connection, sqlite3.Connection],
) -> None:
    results, source = pair
    bodies = sample.draw(results, source, REV, bodies=2, seed=1)
    rows, ids = sample.build_rows(bodies, REV)
    assert len(rows) == 6 == len(ids)
    assert list(rows[0]) == list(sample.COLUMNS)
    assert [r["review_order"] for r in rows] == ["1", "2", "3", "4", "5", "6"]
    for row in rows:
        assert row["faithful"] == row["one_story"] == row["complete"] == row["notes"] == ""
        assert row["spanish_title"] == row["spanish_text"] == row["spanish_body"] == ""
        assert row["body"].startswith("Body number")
    # the ids file ties each row to the exact text the owner read, by hash, never by position
    assert ids[0]["text_hash"] == sample.text_hash(rows[0]["text"])
    assert {i["extractor_revision"] for i in ids} == {REV}
    assert [i["segment_index"] for i in ids[:3]] == ["0", "1", "2"]


def test_the_questions_are_yes_no_columns_named_in_one_place() -> None:
    assert sample.QUESTIONS == ("faithful", "one_story", "complete")
    assert set(sample.QUESTIONS) <= set(sample.COLUMNS)
    assert "yes" in sample.READING_RULE and "no" in sample.READING_RULE


def test_a_long_body_is_cut_for_reading_and_says_so() -> None:
    long = "x" * (sample.TEXT_CAP + 50)
    shown = sample.for_reading(long)
    assert len(shown) < len(long) and shown.endswith(sample.STAGE_TEXT)
    assert sample.for_reading("short") == "short"


def test_the_files_are_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / sample.TO_TAG_NAME).write_text("owner's marks")
    with pytest.raises(SystemExit):
        sample.refuse_to_overwrite(tmp_path)
    sample.refuse_to_overwrite(tmp_path / "empty-folder")  # nothing there: fine
