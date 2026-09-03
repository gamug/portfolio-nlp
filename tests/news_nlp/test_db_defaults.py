"""The unset-DATABASE_URL fallback must point at data/nlp.db (this repo's
dedicated database), not the legacy shared data/urls.db. SOURCE_DATABASE_URL
(the read-only crawl DB the text stages need) resolves the same way but has no
fallback -- unset -> None. See docs/db-topology.md."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest


def _reload_db_with_env(
    monkeypatch: pytest.MonkeyPatch, value: str | None, source_value: str | None = None
) -> ModuleType:
    for name, val in (("DATABASE_URL", value), ("SOURCE_DATABASE_URL", source_value)):
        if val is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, val)
    import db as db_module  # noqa: PLC0415

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return importlib.reload(db_module)


def test_default_db_path_is_nlp_db(monkeypatch: pytest.MonkeyPatch) -> None:
    db_module = _reload_db_with_env(monkeypatch, None)
    assert db_module.DB_PATH.name == "nlp.db"
    assert db_module.DB_PATH.parent.name == "data"
    assert "urls.db" not in str(db_module.DB_PATH)


def test_relative_database_url_resolved_against_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_module = _reload_db_with_env(monkeypatch, "data/custom.db")
    assert db_module.DB_PATH.is_absolute()
    assert db_module.DB_PATH.parts[-2:] == ("data", "custom.db")


def test_absolute_database_url_used_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    abs_path = (
        str(Path("D:/thesis/data/nlp.db")) if Path("D:/").exists() else "/tmp/x/nlp.db"  # noqa: S108
    )
    db_module = _reload_db_with_env(monkeypatch, abs_path)
    assert str(db_module.DB_PATH) == str(Path(abs_path))
    # restore module state for other tests
    _reload_db_with_env(monkeypatch, None)


def test_source_database_url_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    db_module = _reload_db_with_env(monkeypatch, None, source_value=None)
    assert db_module.SOURCE_DB_PATH is None
    _reload_db_with_env(monkeypatch, None)


def test_relative_source_database_url_resolved_against_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_module = _reload_db_with_env(monkeypatch, None, source_value="data/urls.db")
    assert db_module.SOURCE_DB_PATH is not None
    assert db_module.SOURCE_DB_PATH.is_absolute()
    assert db_module.SOURCE_DB_PATH.parts[-2:] == ("data", "urls.db")
    _reload_db_with_env(monkeypatch, None)


def test_absolute_source_database_url_used_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    abs_path = (
        str(Path("D:/thesis/data/urls.db")) if Path("D:/").exists() else "/tmp/x/urls.db"  # noqa: S108
    )
    db_module = _reload_db_with_env(monkeypatch, None, source_value=abs_path)
    assert str(db_module.SOURCE_DB_PATH) == str(Path(abs_path))
    _reload_db_with_env(monkeypatch, None)
