"""Two-tier DB behaviours that run through `pipeline.run_pipeline` / the FastAPI
app. The pure connection/attach/query cases live in portfolio-common's
`tests/test_news_nlp_db.py` now that the DB layer moved there.
"""

import sqlite3
from pathlib import Path

import pytest
from conftest import LEAN_ARTICLES_SCHEMA
from fastapi.testclient import TestClient

import db
import pipeline


def test_run_pipeline_raises_without_source(
    results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.delenv("SOURCE_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="SOURCE_DATABASE_URL"):
        pipeline.run_pipeline()


def test_run_pipeline_raises_when_source_lacks_body_text(
    results_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lean_source = tmp_path / "lean_source.db"
    setup = sqlite3.connect(lean_source)
    setup.executescript(LEAN_ARTICLES_SCHEMA)
    setup.commit()
    setup.close()

    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(lean_source))
    with pytest.raises(RuntimeError, match="body_text"):
        pipeline.run_pipeline()


def test_serving_endpoints_work_without_a_source(
    results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = db.connect(results_db_path)
    seed.execute(
        "INSERT INTO articles (id, ticker, company, title, pub_date, gics_sector) "
        "VALUES (1, 'MMM', '3M', 'A title', '2023-01-01', 'Industrials')"
    )
    db.write_sentiment(seed, 1, "positive", 0.9, 0.9, 0.05, 0.05, "fake-model")
    seed.commit()
    seed.close()

    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.delenv("SOURCE_DATABASE_URL", raising=False)
    from apps.news_nlp_api import app  # noqa: PLC0415

    api = TestClient(app)
    for path in ("/articles", "/articles/1", "/stats/sentiment", "/stats/categories"):
        assert api.get(path).status_code == 200, path
    assert api.get("/sectors/summary").status_code == 200
