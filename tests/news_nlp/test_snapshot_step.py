"""Step 8, the snapshot (T-195, D42): one row per (text, main company) with the crawl date, title, text, category and sentiment. What is written, what waits and why,
what the revision depends on, what the run refuses. No model, no network."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import news_nlp as db_module
import news_snapshot_stage as snap
from news_nlp import steps, triage
from news_nlp.steps import CategoryScore, CompanyScore, SentimentScore
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source

SUM, COMP, CUT, CAT, SENT = "summary-t", "companies-t", "units5-t", "category-t", "sentiment-t"
SECTOR = "sector-t"
ROOT = Path(__file__).resolve().parents[2]
BODY_A = "Apple reported results. " * 20
BODY_B = "Microsoft sold a unit. " * 20
HASH_A, HASH_B = triage.body_hash(BODY_A), triage.body_hash(BODY_B)


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "news_nlp_steps_cli_snapshot", ROOT / "cli" / "news_nlp_steps.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pair(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    """A SOURCE store with the body A crawled twice (ids 1 and 2, the later first in time) and the body B once, attached to a RESULTS store."""
    source = tmp_path / "src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    seed_article(raw, id=1, title="First A", body_text=BODY_A, fetched_at="2022-01-10T12:00:00Z")
    seed_article(raw, id=2, title="Second A", body_text=BODY_A, fetched_at="2022-01-02T09:00:00Z")
    seed_article(raw, id=3, title="Only B", body_text=BODY_B, fetched_at="2022-01-05T10:00:00Z")
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    yield conn
    db_module.detach_source(conn)


def _category(label: str = "earnings_performance") -> CategoryScore:
    return CategoryScore(label, 0.9, "financial", 0.8, {label: 0.9})


def _sentiment(label: str = "positive") -> SentimentScore:
    return SentimentScore(label, 0.8, 0.8, 0.1, 0.1)


def _seed(conn: db_module.NewsNlpDatabase) -> None:
    """Body A (AAPL, MSFT main), body B (MSFT main; its sentiment is missing), a text with no category yet (body C, AAPL main)."""
    for item_id, text in (
        (HASH_A, "Apple summary"),
        (HASH_B, "Microsoft summary"),
        ("c", "c text"),
    ):
        steps.record_summary(conn, steps.ITEM_BODY, item_id, SUM, text, "summary", 1)
    steps.record_companies(
        conn,
        steps.ITEM_BODY,
        HASH_A,
        COMP,
        [CompanyScore("AAPL", 2, 0.9), CompanyScore("MSFT", 1, 0.8)],
        0.5,
    )
    steps.record_companies(conn, steps.ITEM_BODY, HASH_B, COMP, [CompanyScore("MSFT", 1, 0.8)], 0.5)
    steps.record_companies(conn, steps.ITEM_BODY, "c", COMP, [CompanyScore("AAPL", 1, 0.8)], 0.5)
    conn.execute("INSERT INTO main.articles (id) VALUES (1)")  # the foreign key of sub_news
    conn.execute(
        "INSERT INTO sub_news (body_hash, article_id, segment_index, text, extractor_revision, processed_at) "
        "VALUES (?, 1, 0, 's', ?, 't')",
        (HASH_A, CUT),
    )
    conn.commit()
    steps.record_category(
        conn,
        steps.ITEM_BODY,
        HASH_A,
        CAT,
        {"AAPL": _category(), "MSFT": _category("legal_regulatory")},
        "m",
    )
    steps.record_category(conn, steps.ITEM_BODY, HASH_B, CAT, {"MSFT": _category()}, "m")
    steps.record_sentiment(
        conn,
        steps.ITEM_BODY,
        HASH_A,
        SENT,
        {"AAPL": _sentiment(), "MSFT": _sentiment("negative")},
        "m",
    )


def _revision() -> str:
    return snap.revision(SUM, COMP, CUT, CAT, SENT)


def _run(conn: db_module.NewsNlpDatabase, limit: int | None = None) -> dict[str, int]:
    return snap.run(conn, _revision(), SUM, COMP, CAT, SENT, limit)


def _rows(conn: db_module.NewsNlpDatabase) -> list[Any]:
    return conn.execute(
        "SELECT item_id, ticker, fetched_at, title, text, category, sentiment, sector, companies_mentioned FROM news_snapshot "
        "ORDER BY item_id, ticker"
    ).fetchall()


def test_only_complete_pairs_are_written_and_the_others_wait_with_a_reason(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    result = _run(pair)
    assert result["written"] == 2  # the two pairs of body A; B has no sentiment, C no category
    assert result["waiting"] == 2
    status = snap.status(
        pair,
        _revision(),
        summary_revision=SUM,
        companies_revision=COMP,
        category_revision=CAT,
        sentiment_revision=SENT,
    )
    assert status["waiting_pairs"] == {snap.WAIT_NO_SENTIMENT: 1, snap.WAIT_NO_CATEGORY: 1}
    assert status["rows"] == {"company": 2, "sector_texts": 0}


def test_a_row_carries_the_text_the_earliest_crawl_and_the_first_articles_title(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    _run(pair)
    rows = {(r[0], r[1]): r for r in _rows(pair)}
    aapl = rows[(HASH_A, "AAPL")]
    assert (
        aapl["fetched_at"] == "2022-01-02T09:00:00Z"
    )  # the earliest of the two articles of the body, not the lowest id
    assert aapl["title"] == "First A"  # the title of the lowest id
    assert (aapl["text"], aapl["category"], aapl["sentiment"]) == (
        "Apple summary",
        "earnings_performance",
        "positive",
    )
    assert rows[(HASH_A, "MSFT")]["category"] == "legal_regulatory"
    assert rows[(HASH_A, "MSFT")]["sentiment"] == "negative"
    assert aapl["companies_mentioned"] == "[]" and aapl["sector"] == ""


def test_a_rerun_adds_nothing_and_a_late_sentiment_adds_only_its_pair(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    _run(pair)
    before = _rows(pair)
    assert _run(pair)["written"] == 0
    assert _rows(pair) == before
    steps.record_sentiment(
        pair, steps.ITEM_BODY, HASH_B, SENT, {"MSFT": _sentiment("neutral")}, "m"
    )
    assert _run(pair)["written"] == 1  # the sibling pairs of body A are not touched
    assert len(_rows(pair)) == 3


def test_limit_stops_the_run_and_the_rest_is_done_by_the_next(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    assert _run(pair, limit=1)["written"] == 1
    assert _run(pair)["written"] == 1
    assert len(_rows(pair)) == 2


def test_a_sub_news_takes_the_title_and_date_of_its_article(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    sub_id = str(pair.execute("SELECT id FROM sub_news").fetchone()[0])
    steps.record_summary(pair, steps.ITEM_SUB_NEWS, sub_id, SUM, "sub text", "original", 0)
    steps.record_companies(
        pair, steps.ITEM_SUB_NEWS, sub_id, COMP, [CompanyScore("AAPL", 1, 0.9)], 0.5
    )
    steps.record_category(pair, steps.ITEM_SUB_NEWS, sub_id, CAT, {"AAPL": _category("labor")}, "m")
    steps.record_sentiment(pair, steps.ITEM_SUB_NEWS, sub_id, SENT, {"AAPL": _sentiment()}, "m")
    _run(pair)
    row = pair.execute(
        "SELECT title, fetched_at, text FROM news_snapshot WHERE item_kind = 'sub_news'"
    ).fetchone()
    assert (row["title"], row["fetched_at"], row["text"]) == (
        "First A",
        "2022-01-02T09:00:00Z",
        "sub text",
    )


def test_a_pair_whose_article_is_gone_is_counted_not_written(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    steps.record_summary(pair, steps.ITEM_BODY, "gone", SUM, "t", "summary", 1)
    steps.record_companies(pair, steps.ITEM_BODY, "gone", COMP, [CompanyScore("AAPL", 1, 0.9)], 0.5)
    steps.record_category(pair, steps.ITEM_BODY, "gone", CAT, {"AAPL": _category()}, "m")
    steps.record_sentiment(pair, steps.ITEM_BODY, "gone", SENT, {"AAPL": _sentiment()}, "m")
    result = _run(pair)
    assert result["no_article"] == 1 and result["written"] == 2


def test_the_sector_texts_have_no_category_no_sentiment_and_the_two_digit_sector(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    steps.record_summary(pair, steps.ITEM_BODY, HASH_B, "other-summary", "x", "summary", 1)
    pair.execute(
        "INSERT INTO body_sector (body_hash, classifier_revision, status, sector, candidate, margin, processed_at) "
        "VALUES (?, ?, 'assigned', '45', '45', 0.5, 't')",
        (HASH_B, SECTOR),
    )
    pair.commit()
    revision = snap.sector_revision_of(_revision(), SECTOR)
    assert snap.run_sectors(pair, revision, SUM, SECTOR, None) == {
        "written": 1,
        "no_article": 0,
        "waiting": 0,
    }
    row = pair.execute(
        "SELECT * FROM news_snapshot WHERE model_revision = ?", (revision,)
    ).fetchone()
    assert (
        row["ticker"],
        row["sector"],
        row["category"],
        row["sentiment"],
        row["sentiment_score"],
    ) == ("", "45", "", "", None)
    assert snap.run_sectors(pair, revision, SUM, SECTOR, None)["written"] == 0
    assert revision != _revision()


def test_the_revision_follows_each_input_but_not_the_sector_revision() -> None:
    base = _revision()
    assert base == snap.revision(SUM, COMP, CUT, CAT, SENT) and base.startswith("snapshot-")
    for changed in (
        snap.revision("s2", COMP, CUT, CAT, SENT),
        snap.revision(SUM, "c2", CUT, CAT, SENT),
        snap.revision(SUM, COMP, "u2", CAT, SENT),
        snap.revision(SUM, COMP, CUT, "k2", SENT),
        snap.revision(SUM, COMP, CUT, CAT, "e2"),
    ):
        assert changed != base
    assert snap.sector_revision_of(base, "a") != snap.sector_revision_of(base, "b")


def test_the_run_is_refused_for_a_revision_with_no_rows_and_for_rows_of_other_companies(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    assert snap.check_inputs(pair, SUM, COMP, CUT, CAT, SENT, None) == []
    assert snap.check_inputs(pair, "nope", COMP, CUT, CAT, SENT, None) == [
        "summary revision nope: no rows"
    ]
    assert "sector revision x: no rows" in snap.check_inputs(pair, SUM, COMP, CUT, CAT, SENT, "x")
    steps.record_sentiment(
        pair, steps.ITEM_BODY, HASH_B, SENT, {"AAPL": _sentiment()}, "m"
    )  # not a main pair of B
    problems = snap.check_inputs(pair, SUM, COMP, CUT, CAT, SENT, None)
    assert len(problems) == 1 and "not main pairs" in problems[0]


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "summary_revision": SUM,
        "companies_revision": COMP,
        "cut_revision": CUT,
        "category_revision": CAT,
        "sentiment_revision": SENT,
        "sector_revision": None,
        "status": False,
        "limit": None,
    }
    return argparse.Namespace(**{**base, **kw})


def test_the_command_writes_the_snapshots_and_prints_the_counts(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(pair)
    cli = _load_cli()
    cli.main_snapshot(_args(), pair)
    out = capsys.readouterr().out
    assert json.loads(out.split("\n}\n")[0] + "\n}")["written"] == 2
    cli.main_snapshot(_args(status=True), pair)
    assert '"company": 2' in capsys.readouterr().out


def test_the_command_refuses_inputs_that_do_not_agree(pair: db_module.NewsNlpDatabase) -> None:
    _seed(pair)
    with pytest.raises(SystemExit, match="refused: summary revision zz: no rows"):
        _load_cli().main_snapshot(_args(summary_revision="zz"), pair)


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--step",
            "snapshot",
            "--summary-revision",
            "s",
            "--companies-revision",
            "c",
        ],  # no cut, category, sentiment
        [
            "--step",
            "snapshot",
            "--cut-revision",
            "u",
            "--summary-revision",
            "s",
            "--companies-revision",
            "c",
            "--category-revision",
            "k",
        ],
        [
            "--step",
            "sentiment",
            "--category-revision",
            "k",
            "--summary-revision",
            "s",
            "--companies-revision",
            "c",
        ],
        ["--step", "snapshot", "--status", "--model-dir", "x"],
    ],
)
def test_the_options_of_the_snapshot_step_are_required_and_refused_elsewhere(
    argv: list[str],
) -> None:
    cli = _load_cli()
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        cli.check_options(parser, parser.parse_args(argv))


def test_status_alone_needs_no_revision() -> None:
    cli = _load_cli()
    parser = cli.build_parser()
    cli.check_options(parser, parser.parse_args(["--step", "snapshot", "--status"]))
