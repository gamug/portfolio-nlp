"""The SOURCE schema guard (T-135): `PRAGMA user_version` minimum plus a column check,
read-only, never cached, warning once per process on a newer version.
"""

import hashlib
import importlib.util
import sqlite3
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest
from conftest import ARTICLES_SCHEMA, stamp_source
from fastapi.testclient import TestClient

import news_nlp as db
import news_nlp.db as db_mod


@pytest.fixture(autouse=True)
def _reset_warning_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db_mod, "_warned_newer_schema", False)


def _make(path: Path, version: int, schema: str = ARTICLES_SCHEMA) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(schema)
    stamp_source(conn, version)
    conn.commit()
    conn.close()
    return path


@pytest.mark.parametrize("version", [0, 2])
def test_versions_below_the_minimum_fail(tmp_path: Path, version: int) -> None:
    with pytest.raises(db.SourceSchemaError, match=rf"schema version {version};"):
        db.check_schema(_make(tmp_path / "s.db", version))


@pytest.mark.parametrize("version", [3, 4, 5])
def test_versions_at_or_above_the_minimum_pass(tmp_path: Path, version: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 3, 4 and 5 must not warn (the real file is at 5)
        assert db.check_schema(_make(tmp_path / "s.db", version)) == version


def test_version_zero_message_says_never_stamped(tmp_path: Path) -> None:
    with pytest.raises(db.SourceSchemaError, match="never stamped"):
        db.check_schema(_make(tmp_path / "s.db", 0))


def test_a_newer_version_warns_once_per_process(tmp_path: Path) -> None:
    path = _make(tmp_path / "s.db", 6)
    with pytest.warns(UserWarning, match="newer than the latest"):
        db.check_schema(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        db.check_schema(path)  # second open, same process: no second warning
    # a new process warns again
    code = (
        "import sys, warnings; sys.path.insert(0, 'src'); import news_nlp as db\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always'); db.check_schema(sys.argv[1])\n"
        "print(len(w))"
    )
    out = subprocess.run(  # noqa: S603 -- fixed interpreter and our own snippet
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "1"


def test_a_missing_column_fails_even_when_the_number_passes(tmp_path: Path) -> None:
    path = _make(tmp_path / "s.db", 3, ARTICLES_SCHEMA.replace("    body_text TEXT,\n", ""))
    with pytest.raises(db.SourceSchemaError, match=r"articles\.body_text"):
        db.check_schema(path)


def test_the_column_list_is_a_parameter(tmp_path: Path) -> None:
    path = _make(tmp_path / "s.db", 5)
    with pytest.raises(db.SourceSchemaError, match=r"articles\.prev_fetch_status"):
        db.check_schema(path, columns={"articles": ("id", "prev_fetch_status")})


def test_a_restamped_file_passes_both_times(tmp_path: Path) -> None:
    path = _make(tmp_path / "s.db", 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert db.check_schema(path) == 5
        c = sqlite3.connect(path)
        stamp_source(c, 3)  # the documented cache rollback: 5 -> 3
        c.commit()
        c.close()
        assert db.check_schema(path) == 3  # not cached from the first open


def test_the_helper_runs_on_a_bare_read_only_connection(tmp_path: Path) -> None:
    path = _make(tmp_path / "s.db", 4)
    raw = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert db.check_schema(raw, name=str(path)) == 4
    finally:
        raw.close()


def test_the_guard_writes_nothing(tmp_path: Path) -> None:
    path = _make(tmp_path / "s.db", 5)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db.check_schema(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_a_wal_file_with_a_live_writer_is_read_only_checkable(tmp_path: Path) -> None:
    path = _make(tmp_path / "s.db", 5)
    writer = sqlite3.connect(path)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("INSERT INTO articles (id, ticker) VALUES (1, 'MMM')")
    writer.commit()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert db.check_schema(path) == 5
    finally:
        writer.close()


# --- wiring: connect_pipeline, single-file runs, the API -----------------------


def test_connect_pipeline_rejects_an_old_source_and_closes(
    tmp_path: Path, results_db_path: Path
) -> None:
    old = _make(tmp_path / "old.db", 2)
    with pytest.raises(db.SourceSchemaError, match="schema version 2"):
        db.connect_pipeline(results_db=results_db_path, source_db=old)


def test_a_single_file_run_does_not_run_the_guard(source_db_path: Path) -> None:
    # SOURCE == RESULTS: nothing is attached, and RESULTS is never stamped.
    c = sqlite3.connect(source_db_path)
    stamp_source(c, 0)
    c.commit()
    c.close()
    conn = db.connect_pipeline(results_db=source_db_path, source_db=source_db_path)
    conn.close()


def test_the_api_answers_503_and_takes_no_run_slot(
    client: TestClient,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _make(tmp_path / "old.db", 2)
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(old))
    first = client.post("/pipeline/run", json={})
    assert first.status_code == 503
    assert "schema version 2" in first.json()["detail"]
    again = client.post("/pipeline/run", json={})
    assert again.status_code == 503  # not 409: the slot was never taken
    assert client.get("/pipeline/status").json()["status"] != "running"


def test_the_no_source_endpoints_are_unaffected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SOURCE_DATABASE_URL", raising=False)
    assert client.get("/health").status_code == 200
    assert client.get("/articles").status_code == 200


def test_the_api_swallows_an_unreadable_source_and_the_worker_reports_it(
    client: TestClient, tmp_path: Path, results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # check_source cannot open the file (sqlite3.Error): not a 503, not a 500; the
    # request is accepted and the worker's own attach reports the problem in the status.
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(tmp_path / "missing.db"))
    resp = client.post("/pipeline/run", json={})
    assert resp.status_code == 202
    status = {}
    for _ in range(300):
        status = client.get("/pipeline/status").json()
        if status["status"] != "running":
            break
        time.sleep(0.01)
    assert status["status"] == "error"
    assert status["error"]


def test_exit_on_schema_error_gives_a_one_line_exit() -> None:
    with pytest.raises(SystemExit) as info, db.exit_on_schema_error():
        raise db.SourceSchemaError("SOURCE database x is at schema version 2")
    assert str(info.value) == "error: SOURCE database x is at schema version 2"
    with pytest.raises(ValueError), db.exit_on_schema_error():  # other errors keep their traceback
        raise ValueError("boom")


def test_the_pipeline_cli_exits_with_the_message(
    tmp_path: Path, results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _make(tmp_path / "old.db", 2)
    path = Path(__file__).resolve().parents[2] / "cli" / "news_nlp_cli.py"
    spec = importlib.util.spec_from_file_location("news_nlp_cli_under_test", path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(
        sys, "argv", ["news_nlp_cli", "--source-db", str(old), "--results-db", str(results_db_path)]
    )
    with pytest.raises(SystemExit) as info:
        cli.main()
    assert "schema version 2" in str(info.value)
    assert info.value.code != 0
