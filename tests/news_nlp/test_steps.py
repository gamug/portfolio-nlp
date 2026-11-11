"""Step 1 storage and run tracking (T-162/T-163): a rerun adds nothing, --limit bounds the work, failures are retried."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import news_nlp as db_module
from news_nlp import steps
from scope_stage import combine, fragment, revision_of
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source

REV = "scope-nli-test"


def _source_with(path: Path, rows: list[tuple[int, str, str]]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(ARTICLES_SCHEMA)
    stamp_source(conn)
    for i, title, body in rows:
        seed_article(conn, id=i, title=title, body_text=body)
    conn.commit()
    conn.close()


@pytest.fixture
def pair(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    source = tmp_path / "src.db"
    _source_with(
        source,
        [
            (1, "A", "Body one. " * 5),
            (2, "A again", "Body one.  " * 5),  # same body, spacing differs: one unique body
            (3, "B", "Body two. " * 5),
            (4, "C", "Body three. " * 5),
        ],
    )
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    yield conn
    db_module.detach_source(conn)
    conn.close()


def _do(conn: db_module.NewsNlpDatabase, body: steps.Body, scope: str = "single") -> None:
    steps.record_scope(conn, body, REV, scope, {"topic:company": 1.0}, "title+800")


def test_a_body_under_two_tickers_is_one_item(pair: db_module.NewsNlpDatabase) -> None:
    got = list(steps.pending_bodies(pair, steps.STEP_SCOPE, REV))
    assert [b.article_id for b in got] == [1, 3, 4]


def test_limit_touches_at_most_n(pair: db_module.NewsNlpDatabase) -> None:
    assert len(list(steps.pending_bodies(pair, steps.STEP_SCOPE, REV, limit=2))) == 2


def test_a_rerun_adds_and_changes_nothing(pair: db_module.NewsNlpDatabase) -> None:
    for body in steps.pending_bodies(pair, steps.STEP_SCOPE, REV):
        _do(pair, body)
    before = pair.execute("SELECT * FROM body_scope ORDER BY 1").fetchall()
    state = pair.execute("SELECT * FROM pipeline_step_state ORDER BY 2").fetchall()
    assert list(steps.pending_bodies(pair, steps.STEP_SCOPE, REV)) == []
    assert (
        len(list(steps.pending_bodies(pair, steps.STEP_SCOPE, "other"))) == 3
    )  # a new revision starts over
    assert pair.execute("SELECT * FROM body_scope ORDER BY 1").fetchall() == before
    assert pair.execute("SELECT * FROM pipeline_step_state ORDER BY 2").fetchall() == state


def test_a_failed_body_is_retried_and_the_done_ones_are_not(
    pair: db_module.NewsNlpDatabase,
) -> None:
    first, second, third = list(steps.pending_bodies(pair, steps.STEP_SCOPE, REV))
    _do(pair, first)
    steps.record_failure(pair, second, steps.STEP_SCOPE, REV, "boom")
    again = list(steps.pending_bodies(pair, steps.STEP_SCOPE, REV))
    assert [b.body_hash for b in again] == [second.body_hash, third.body_hash]
    _do(pair, second, "multi")
    row = pair.execute(
        "SELECT status, attempts, error FROM pipeline_step_state WHERE item_id = ?",
        (second.body_hash,),
    ).fetchone()
    assert tuple(row) == ("done", 2, None)


def test_status_counts_add_up(pair: db_module.NewsNlpDatabase) -> None:
    first, second, third = list(steps.pending_bodies(pair, steps.STEP_SCOPE, REV))
    _do(pair, first, "single")
    _do(pair, second, "out_of_scope")
    steps.record_failure(pair, third, steps.STEP_SCOPE, REV, "x")
    got = steps.status(pair, steps.STEP_SCOPE, REV)
    assert got["by_state"] == {"done": 2, "failed": 1}
    assert got["by_scope"] == {"single": 1, "out_of_scope": 1}


def test_combine_follows_the_two_stages() -> None:
    assert combine("sector", None) == "sector"
    assert combine("not_related", None) == "out_of_scope"
    assert combine("company", "multi") == "multi"
    with pytest.raises(ValueError, match="stage B"):
        combine("company", None)


def test_the_model_reads_the_title_and_the_first_800_characters() -> None:
    assert fragment("T", "x" * 900, 800) == "T. " + "x" * 800


def test_the_revision_is_the_hash_of_the_training_record(tmp_path: Path) -> None:
    (tmp_path / "training_record.json").write_text('{"a": 1}', encoding="utf-8")
    first = revision_of(tmp_path)
    (tmp_path / "training_record.json").write_text('{"a": 2}', encoding="utf-8")
    assert first.startswith("scope-nli-") and revision_of(tmp_path) != first
