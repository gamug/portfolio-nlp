import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import news_nlp as db_module

ARTICLES_SCHEMA = """
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    ticker TEXT,
    company TEXT,
    gics_sector TEXT,
    gics_sub_industry TEXT,
    title TEXT,
    author TEXT,
    pub_date TEXT,
    fetched_at TEXT,
    body_text TEXT,
    word_count INTEGER,
    source_domain TEXT,
    fetch_status TEXT,
    http_status_code INTEGER
)
"""

# The RESULTS store's `articles` subset: every column above except `body_text`
# (the heavy crawled text stays in the SOURCE store). See docs/db-topology.md.
LEAN_ARTICLES_SCHEMA = ARTICLES_SCHEMA.replace("    body_text TEXT,\n", "")


def seed_article(
    conn: sqlite3.Connection,
    id: int,
    company: str = "3M",
    ticker: str = "MMM",
    title: str = "Test Title",
    pub_date: str | None = "2023-01-15T00:00:00Z",
    body_text: str = "Body text.",
    word_count: int = 2,
    source_domain: str = "example.com",
    fetch_status: str = "ok",
    gics_sector: str = "Industrials",
    gics_sub_industry: str = "Industrial Conglomerates",
    fetched_at: str = "2023-01-15T00:00:00Z",
    http_status_code: int = 200,
) -> None:
    conn.execute(
        """INSERT INTO articles
           (id, ticker, company, gics_sector, gics_sub_industry, title, author, pub_date,
            fetched_at, body_text, word_count, source_domain, fetch_status, http_status_code)
           VALUES (?, ?, ?, ?, ?, ?, 'Author', ?, ?, ?, ?, ?, ?, ?)""",
        (
            id,
            ticker,
            company,
            gics_sector,
            gics_sub_industry,
            title,
            pub_date,
            fetched_at,
            body_text,
            word_count,
            source_domain,
            fetch_status,
            http_status_code,
        ),
    )


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.executescript(ARTICLES_SCHEMA)
    conn.commit()
    conn.close()

    conn = db_module.connect(path)
    db_module.init_schema(conn)
    conn.close()
    return path


@pytest.fixture
def conn(test_db_path: Path) -> Iterator[sqlite3.Connection]:
    c = db_module.connect(test_db_path)
    yield c
    c.close()


@pytest.fixture
def client(test_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", str(test_db_path))
    # Deliberately deferred: importing apps.news_nlp_api pulls in pipeline,
    # which imports torch/transformers -- fine for the tests that use this fixture,
    # but every other test in the suite would otherwise pay that import cost too.
    from apps.news_nlp_api import app, pipeline_status  # noqa: PLC0415

    pipeline_status.reset()
    yield TestClient(app)
    pipeline_status.reset()


# --- two-tier DB contract (SOURCE + RESULTS) --------------------------------


@pytest.fixture
def source_db_path(tmp_path: Path) -> Path:
    """A read-only-style SOURCE database: `articles` (incl. `body_text`), three
    seeded rows, no result tables. See docs/db-topology.md."""
    path = tmp_path / "source.db"
    conn = sqlite3.connect(path)
    conn.executescript(ARTICLES_SCHEMA)
    for i, (ticker, sector) in enumerate(
        [("MMM", "Industrials"), ("AAPL", "Information Technology"), ("XOM", "Energy")], start=1
    ):
        seed_article(
            conn,
            id=i,
            ticker=ticker,
            company=f"Company {ticker}",
            gics_sector=sector,
            body_text=f"Body text for article {i}. " * 20,
        )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def results_db_path(tmp_path: Path) -> Path:
    """A RESULTS store: lean `articles` (no `body_text`) + the result tables,
    all empty."""
    path = tmp_path / "results.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEAN_ARTICLES_SCHEMA)
    conn.commit()
    conn.close()

    conn = db_module.connect(path)
    db_module.init_schema(conn)
    conn.close()
    return path


@pytest.fixture
def two_tier_conn(source_db_path: Path, results_db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source_db_path)
    yield conn
    db_module.detach_source(conn)
    conn.close()


@pytest.fixture
def two_tier_client(
    source_db_path: Path, results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(source_db_path))
    from apps.news_nlp_api import app, pipeline_status  # noqa: PLC0415

    pipeline_status.reset()
    yield TestClient(app)
    pipeline_status.reset()
