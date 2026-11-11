"""Step 1.2, the cut (T-163): multi bodies become sub-news, a rerun adds nothing, a body with no company text is skipped."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

import news_nlp as db_module
from cut_stage import UnitsExtractor, companies_of, distinct, revision_of
from news_nlp import attribution_model as am
from news_nlp import steps
from news_nlp.triage import Company
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source

SCOPE_REV = "scope-nli-test"
CUT_REV = "units-test"
UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
]
STORY = (
    "Stocks making the biggest moves midday:\n"
    "Apple - Shares rose 3% after Apple raised its dividend.\n"
    "Microsoft - Shares fell 2% as cloud growth slowed in the quarter."
)


class QuietScorer:
    """Every pair scores low: only the pieces that name a company are given to it."""

    def score(self, pairs: Sequence[am.Pair]) -> list[float]:
        return [0.1 for _ in pairs]


@pytest.fixture
def pair(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    source = tmp_path / "src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    for i, title, text in [
        (1, "A", "First multi body. " * 6),
        (2, "B", "A single body. " * 6),
        (3, "C", "Second multi body. " * 6),
        (4, "C again", "Second  multi body. " * 6),  # same body as 3, spacing differs
        (5, "D", "Third multi body. " * 6),
        (6, "E", "A quarantined multi body. " * 6),
    ]:
        seed_article(
            raw, id=i, title=title, body_text=text, fetch_status="quarantined" if i == 6 else "ok"
        )
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    for article_id, scope in [
        (1, "multi"),
        (2, "single"),
        (3, "multi"),
        (5, "multi"),
        (6, "multi"),
    ]:
        body = next(
            b
            for b in steps.pending_bodies(conn, steps.STEP_SCOPE, SCOPE_REV)
            if b.article_id == article_id
        )
        steps.record_scope(conn, body, SCOPE_REV, scope, {}, "title+800")
    yield conn
    db_module.detach_source(conn)
    conn.close()


def test_only_multi_bodies_are_listed_once_each(pair: db_module.NewsNlpDatabase) -> None:
    got = list(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV))
    # the single body is left out, 3 and 4 are one body, and the quarantined body 6 is part of the corpus (D36)
    assert [b.article_id for b in got] == [1, 3, 5, 6]


def test_another_classifier_revision_has_no_multi_bodies(pair: db_module.NewsNlpDatabase) -> None:
    assert list(steps.pending_multi_bodies(pair, "scope-nli-other", CUT_REV)) == []


def test_limit_touches_at_most_n(pair: db_module.NewsNlpDatabase) -> None:
    assert len(list(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV, limit=2))) == 2


def test_the_cut_writes_sub_news_and_a_rerun_adds_nothing(pair: db_module.NewsNlpDatabase) -> None:
    for body in steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV):
        steps.record_cut(pair, body, CUT_REV, ["text for one", "text for two"])
    rows = pair.execute(
        "SELECT body_hash, article_id, segment_index, text FROM sub_news ORDER BY id"
    ).fetchall()
    assert len(rows) == 8
    assert [r[2] for r in rows[:2]] == [0, 1]
    assert {r[1] for r in rows} == {1, 3, 5, 6}  # the lowest article id of each body
    state = pair.execute(
        "SELECT * FROM pipeline_step_state WHERE step = 'cut' ORDER BY 2"
    ).fetchall()
    assert list(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV)) == []
    assert pair.execute("SELECT COUNT(*) FROM sub_news").fetchone()[0] == 8
    assert (
        pair.execute("SELECT * FROM pipeline_step_state WHERE step = 'cut' ORDER BY 2").fetchall()
        == state
    )
    assert (
        len(list(steps.pending_multi_bodies(pair, SCOPE_REV, "units-new"))) == 4
    )  # a new revision starts over


def test_writing_the_same_body_twice_keeps_one_set_of_rows(pair: db_module.NewsNlpDatabase) -> None:
    body = next(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV))
    steps.record_cut(pair, body, CUT_REV, ["a", "b"])
    steps.record_cut(pair, body, CUT_REV, ["a", "b"])
    assert pair.execute("SELECT COUNT(*) FROM sub_news").fetchone()[0] == 2


def test_the_lean_article_row_is_copied_for_the_foreign_key(
    pair: db_module.NewsNlpDatabase,
) -> None:
    body = next(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV))
    steps.record_cut(pair, body, CUT_REV, ["a"])
    assert (
        pair.execute("SELECT id FROM main.articles WHERE id = ?", (body.article_id,)).fetchone()
        is not None
    )


def test_a_body_with_no_company_text_is_skipped_and_not_retried(
    pair: db_module.NewsNlpDatabase,
) -> None:
    body = next(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV))
    steps.record_cut(pair, body, CUT_REV, [])
    assert pair.execute("SELECT COUNT(*) FROM sub_news").fetchone()[0] == 0
    row = pair.execute(
        "SELECT status, error FROM pipeline_step_state WHERE step = 'cut'"
    ).fetchone()
    assert tuple(row) == ("skipped", "no company text")
    assert body.body_hash not in {
        b.body_hash for b in steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV)
    }


def test_a_failed_body_is_retried_and_the_done_ones_are_not(
    pair: db_module.NewsNlpDatabase,
) -> None:
    first, second, third, fourth = list(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV))
    steps.record_cut(pair, first, CUT_REV, ["a"])
    steps.record_failure(pair, second, steps.STEP_CUT, CUT_REV, "boom")
    again = list(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV))
    assert [b.body_hash for b in again] == [second.body_hash, third.body_hash, fourth.body_hash]


def test_status_counts_the_sub_news_and_the_states(pair: db_module.NewsNlpDatabase) -> None:
    first, second, *_rest = list(steps.pending_multi_bodies(pair, SCOPE_REV, CUT_REV))
    steps.record_cut(pair, first, CUT_REV, ["a", "b"])
    steps.record_cut(pair, second, CUT_REV, [])
    got = steps.status(pair, steps.STEP_CUT, CUT_REV)
    assert got["by_state"] == {"done": 1, "skipped": 1}
    assert got["sub_news"] == {"rows": 2, "bodies": 1}


def test_the_latest_classifier_revision_is_the_last_one_written(
    pair: db_module.NewsNlpDatabase,
) -> None:
    assert steps.latest_classifier_revision(pair) == SCOPE_REV


def test_the_extractor_gives_each_company_its_text() -> None:
    extractor = UnitsExtractor(QuietScorer(), UNIVERSE, "local:test")
    texts = extractor.cut(STORY)
    assert len(texts) == 2
    assert "Apple" in texts[0] and "Microsoft" not in texts[0]
    assert "Microsoft" in texts[1] and "Apple" not in texts[1]


def test_the_extractor_gives_nothing_for_a_body_that_names_no_company() -> None:
    assert (
        UnitsExtractor(QuietScorer(), UNIVERSE, "local:test").cut(
            "Markets were quiet today. Nothing moved."
        )
        == []
    )


def test_repeated_texts_are_one_sub_news() -> None:
    assert distinct(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_the_revision_follows_the_model_and_the_names_the_model_is_asked_about() -> None:
    names = {c.ticker: c.name for c in UNIVERSE}
    base = revision_of("local:one", names)
    assert base == revision_of("local:one", dict(names))
    assert base.startswith("units-")
    assert revision_of("local:two", names) != base
    assert revision_of("local:one", {**names, "ORCL": "Oracle Corporation"}) != base


def test_the_universe_is_read_from_the_source(pair: db_module.NewsNlpDatabase) -> None:
    assert all(isinstance(c, Company) for c in companies_of(pair))
