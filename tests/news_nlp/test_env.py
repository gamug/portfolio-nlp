"""news_nlp.env path resolution: `$DATABASE_URL` (RESULTS store, packaged
default `data/nlp.db`) and `$SOURCE_DATABASE_URL` (SOURCE store, no default ->
`None` when unset). A relative value is left relative (resolved against CWD at
open time); an absolute value is used as-is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from news_nlp import env


def test_results_db_path_defaults_to_nlp_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = env.results_db_path()
    assert result == Path("data/nlp.db")
    assert "urls.db" not in str(result)


def test_relative_database_url_is_left_relative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "data/custom.db")
    result = env.results_db_path()
    assert not result.is_absolute()
    assert result == Path("data/custom.db")


def test_absolute_database_url_used_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    abs_path = str(Path("/tmp/x/nlp.db"))  # noqa: S108
    monkeypatch.setenv("DATABASE_URL", abs_path)
    assert env.results_db_path() == Path(abs_path)


def test_results_db_path_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "data/env.db")
    assert env.results_db_path("/tmp/explicit.db") == Path("/tmp/explicit.db")  # noqa: S108


def test_source_database_url_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOURCE_DATABASE_URL", raising=False)
    assert env.source_db_path() is None


def test_relative_source_database_url_is_left_relative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATABASE_URL", "data/urls.db")
    result = env.source_db_path()
    assert result is not None
    assert not result.is_absolute()
    assert result == Path("data/urls.db")


def test_absolute_source_database_url_used_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    abs_path = str(Path("/tmp/x/urls.db"))  # noqa: S108
    monkeypatch.setenv("SOURCE_DATABASE_URL", abs_path)
    assert env.source_db_path() == Path(abs_path)


def test_empty_string_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("SOURCE_DATABASE_URL", "")
    assert env.results_db_path() == Path("data/nlp.db")
    assert env.source_db_path() is None
