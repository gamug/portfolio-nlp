"""Fixture-DB coverage for scripts/migrate_from_urls_db.py."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "migrate_from_urls_db",
    Path(__file__).resolve().parents[2] / "scripts" / "migrate_from_urls_db.py",
)
migrate_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate_mod)

FULL_ARTICLES_DDL = """
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    ticker TEXT, company TEXT, gics_sector TEXT, gics_sub_industry TEXT,
    title TEXT, author TEXT, pub_date TEXT, body_text TEXT, word_count INTEGER,
    language TEXT, source_domain TEXT, extraction_method TEXT,
    fetch_status TEXT, http_status_code INTEGER, fetched_at TEXT,
    FOREIGN KEY (id) REFERENCES discovered_urls (id)
)
"""

RESULT_DDL = {
    "article_sentiment": """CREATE TABLE article_sentiment (
        article_id INTEGER PRIMARY KEY REFERENCES articles(id), label TEXT NOT NULL,
        score REAL NOT NULL, positive REAL NOT NULL, negative REAL NOT NULL,
        neutral REAL NOT NULL, model_name TEXT NOT NULL, processed_at TEXT NOT NULL)""",
    "article_entities": """CREATE TABLE article_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER NOT NULL REFERENCES articles(id),
        entity_type TEXT NOT NULL, text TEXT NOT NULL, start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL, score REAL, model_name TEXT NOT NULL,
        processed_at TEXT NOT NULL)""",
    "article_category": """CREATE TABLE article_category (
        article_id INTEGER PRIMARY KEY REFERENCES articles(id), label TEXT NOT NULL,
        score REAL NOT NULL, earnings_performance REAL NOT NULL,
        mergers_acquisitions REAL NOT NULL, leadership_governance REAL NOT NULL,
        legal_regulatory REAL NOT NULL, product_innovation REAL NOT NULL,
        capital_shareholder_returns REAL NOT NULL, labor_human_capital REAL NOT NULL,
        market_analyst_sentiment REAL NOT NULL, partnerships_business_dev REAL NOT NULL,
        model_name TEXT NOT NULL, processed_at TEXT NOT NULL)""",
    "article_summary": """CREATE TABLE article_summary (
        article_id INTEGER PRIMARY KEY REFERENCES articles(id),
        summary_text TEXT NOT NULL, num_chunks INTEGER NOT NULL,
        model_name TEXT NOT NULL, processed_at TEXT NOT NULL)""",
    "sector_summary": """CREATE TABLE sector_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT, gics_sector TEXT NOT NULL,
        gics_sub_industry TEXT NOT NULL, week_start TEXT NOT NULL, week_end TEXT NOT NULL,
        summary_text TEXT NOT NULL, num_articles INTEGER NOT NULL,
        num_companies INTEGER NOT NULL, model_name TEXT NOT NULL,
        processed_at TEXT NOT NULL, format_version INTEGER NOT NULL DEFAULT 0,
        facts_json TEXT NOT NULL DEFAULT '{}', intro_text TEXT NOT NULL DEFAULT '')""",
}


def _build_source(path: Path) -> None:
    c = sqlite3.connect(path)
    c.executescript("CREATE TABLE discovered_urls (id INTEGER PRIMARY KEY, url TEXT);")
    c.executescript(FULL_ARTICLES_DDL)
    for ddl in RESULT_DDL.values():
        c.executescript(ddl)
    c.executemany(
        "INSERT INTO discovered_urls VALUES (?, ?)", [(i, f"http://x/{i}") for i in (1, 2, 3)]
    )
    # articles 1,2,3 exist; result rows will also reference a missing id 99
    for i in (1, 2, 3):
        c.execute(
            "INSERT INTO articles (id, ticker, title, body_text, http_status_code) "
            "VALUES (?, ?, ?, ?, 200)",
            (i, "MMM", f"t{i}", "X" * 5000),
        )
    for i in (1, 2):
        c.execute(
            "INSERT INTO article_sentiment VALUES (?,?,?,?,?,?,?,?)",
            (i, "positive", 0.9, 0.9, 0.05, 0.05, "finbert", "2024-01-01"),
        )
        c.execute(
            "INSERT INTO article_category VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                i,
                "earnings_performance",
                0.8,
                0.8,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                "deberta",
                "2024-01-01",
            ),
        )
        c.execute(
            "INSERT INTO article_summary VALUES (?,?,?,?,?)",
            (i, "sum", 1, "distilbart", "2024-01-01"),
        )
    c.executemany(
        "INSERT INTO article_entities "
        "(article_id, entity_type, text, start_char, end_char, score, model_name, processed_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, "ORG", "3M", 0, 2, 0.99, "sec-bert", "2024-01-01"),
            (2, "ORG", "3M", 0, 2, 0.99, "sec-bert", "2024-01-01"),
            (99, "ORG", "Ghost", 0, 5, 0.99, "sec-bert", "2024-01-01"),
        ],  # orphan
    )
    c.execute(
        "INSERT INTO article_sentiment VALUES (?,?,?,?,?,?,?,?)",
        (99, "neutral", 0.5, 0.3, 0.2, 0.5, "finbert", "2024-01-01"),
    )  # orphan
    c.execute(
        "INSERT INTO sector_summary "
        "(gics_sector, gics_sub_industry, week_start, week_end, summary_text, "
        "num_articles, num_companies, model_name, processed_at) "
        "VALUES ('Industrials','X','2024-01-01','2024-01-07','s',2,1,'m','2024-01-01')"
    )
    c.commit()
    c.close()


def test_migrate_copies_results_and_lean_articles(tmp_path: Path) -> None:
    src, dest = tmp_path / "urls.db", tmp_path / "nlp.db"
    _build_source(src)

    counts = migrate_mod.migrate(src, dest)

    d = sqlite3.connect(dest)
    cols = {r[1] for r in d.execute("PRAGMA table_info(articles)")}
    assert "body_text" not in cols
    assert cols == {
        "id",
        "ticker",
        "company",
        "gics_sector",
        "gics_sub_industry",
        "title",
        "author",
        "pub_date",
        "word_count",
        "language",
        "source_domain",
        "extraction_method",
        "fetch_status",
        "http_status_code",
        "fetched_at",
    }
    # only articles referenced by a *kept* result row: ids 1 and 2
    assert {r[0] for r in d.execute("SELECT id FROM articles")} == {1, 2}
    # orphan (id 99) result rows skipped
    assert d.execute("SELECT COUNT(*) FROM article_sentiment").fetchone()[0] == 2
    assert d.execute("SELECT COUNT(*) FROM article_entities").fetchone()[0] == 2
    assert d.execute("SELECT COUNT(*) FROM sector_summary").fetchone()[0] == 1
    # values landed in the right columns -- dest `sector_summary` column order
    # (SCHEMA + ALTER TABLE ADD COLUMN) differs from src, so a positional
    # `SELECT *` copy would silently shuffle these
    ss = d.execute(
        "SELECT model_name, summary_text, processed_at, format_version, facts_json "
        "FROM sector_summary"
    ).fetchone()
    assert ss == ("m", "s", "2024-01-01", 0, "{}")
    assert d.execute("PRAGMA foreign_key_check").fetchall() == []
    # the entities index is back
    idx = {r[1] for r in d.execute("PRAGMA index_list(article_entities)")}
    assert "idx_article_entities_article_id" in idx
    d.close()

    assert counts["article_sentiment"] == 2
    assert counts["article_entities"] == 2
    assert counts["articles"] == 2


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    src, dest = tmp_path / "urls.db", tmp_path / "nlp.db"
    _build_source(src)
    migrate_mod.migrate(src, dest)
    second = migrate_mod.migrate(src, dest)
    assert all(v == 0 for v in second.values())
    d = sqlite3.connect(dest)
    assert d.execute("SELECT COUNT(*) FROM article_sentiment").fetchone()[0] == 2
    d.close()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    src, dest = tmp_path / "urls.db", tmp_path / "nlp.db"
    _build_source(src)
    counts = migrate_mod.migrate(src, dest, dry_run=True)
    assert counts["article_sentiment"] == 2
    assert (
        not dest.exists()
        or sqlite3.connect(dest)
        .execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        .fetchone()[0]
        == 0
    )


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        migrate_mod.migrate(tmp_path / "nope.db", tmp_path / "nlp.db")
