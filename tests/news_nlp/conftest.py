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


# --- news_nlp.eval helpers ------------------------------------------------------


def build_eval_source(path: Path, n: int) -> None:
    """A SOURCE db with *n* articles (id 1..n), each with distinct body_text."""
    conn = sqlite3.connect(path)
    conn.executescript(ARTICLES_SCHEMA)
    for i in range(1, n + 1):
        seed_article(
            conn,
            id=i,
            ticker=f"T{i:02d}",
            company=f"Company {i:02d}",
            title=f"Headline for article {i}",
            body_text=(
                f"Article {i} body. Acme Corp reported quarterly results in New York; "
                f"CEO Jane Doe commented on demand. " * 4
            ),
        )
    conn.commit()
    conn.close()


#: In the ``eval_store_paths`` / ``eval_conn`` fixtures, article ids 1..8 are
#: seeded low-confidence, 9..20 high-confidence.
EVAL_N = 20
EVAL_LOW_IDS = frozenset(range(1, 9))


@pytest.fixture
def eval_store_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Seed a two-tier (SOURCE, RESULTS) pair with ``EVAL_N`` articles and one
    prediction per per-article stage, then return the two paths (no open
    connection). ids in ``EVAL_LOW_IDS`` get low-confidence scores."""
    source = tmp_path / "eval_src.db"
    results = tmp_path / "eval_res.db"
    build_eval_source(source, EVAL_N)

    raw = sqlite3.connect(results)
    raw.executescript(LEAN_ARTICLES_SCHEMA)
    raw.commit()
    raw.close()
    seed_conn = db_module.connect(results)
    db_module.init_schema(seed_conn)
    seed_conn.close()

    conn = db_module.connect_pipeline(results_db=results, source_db=source)
    try:
        for i in range(1, EVAL_N + 1):
            if i in EVAL_LOW_IDS:
                write_stage_predictions(
                    conn,
                    i,
                    sentiment_score=0.30 + i * 0.015,
                    category_label="earnings_performance",
                    category_score=0.33 + i * 0.01,
                    ner_score=0.40 + i * 0.02,
                )
            else:
                write_stage_predictions(
                    conn, i, sentiment_score=0.95, category_score=0.88, ner_score=0.97
                )
    finally:
        db_module.detach_source(conn)
        conn.close()
    return source, results


@pytest.fixture
def eval_conn(
    eval_store_paths: tuple[Path, Path],
) -> Iterator[db_module.NewsNlpDatabase]:
    source, results = eval_store_paths
    conn = db_module.connect_pipeline(results_db=results, source_db=source)
    yield conn
    db_module.detach_source(conn)
    conn.close()


def write_stage_predictions(
    conn: sqlite3.Connection,
    article_id: int,
    *,
    sentiment_score: float = 0.92,
    sentiment_label: str = "positive",
    category_label: str = "earnings_performance",
    category_score: float = 0.81,
    ner_score: float = 0.95,
    with_summary: bool = True,
) -> None:
    """Write one row per per-article result table for *article_id* (via the real
    ``news_nlp.queries`` writers, so the lean ``articles`` row is copied too)."""
    dist = {slug: 0.02 for slug in db_module.CATEGORY_SLUGS}
    if category_label in dist:
        dist[category_label] = category_score
    db_module.write_sentiment(
        conn,
        article_id,
        label=sentiment_label,
        score=sentiment_score,
        positive=sentiment_score if sentiment_label == "positive" else 0.1,
        negative=sentiment_score if sentiment_label == "negative" else 0.1,
        neutral=sentiment_score if sentiment_label == "neutral" else 0.1,
        model_name="test-finbert",
    )
    db_module.write_category(
        conn,
        article_id,
        label=category_label,
        score=category_score,
        scores=dist,
        model_name="test-deberta",
    )
    db_module.write_entities(
        conn,
        article_id,
        [
            {
                "entity_type": "ORG",
                "text": "Acme Corp",
                "start_char": 0,
                "end_char": 9,
                "score": ner_score,
            }
        ],
        model_name="test-secbert",
    )
    if with_summary:
        db_module.write_company_summary(
            conn, article_id, f"Summary of article {article_id}.", 1, "test-distilbart"
        )
    conn.commit()


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
