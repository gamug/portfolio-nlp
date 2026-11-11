"""Step 1 storage and run tracking (T-162/T-163): a rerun adds nothing, --limit bounds the work, failures are retried."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import news_nlp as db_module
from news_nlp import steps
from scope_stage import ScopeResult, assemble, combine, fragment, revision_of
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source

REV = "scope-nli-test"


def _source_with(
    path: Path, rows: list[tuple[int, str, str]], quarantined: tuple[int, ...] = ()
) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(ARTICLES_SCHEMA)
    stamp_source(conn)
    for i, title, body in rows:
        seed_article(
            conn,
            id=i,
            title=title,
            body_text=body,
            fetch_status="quarantined" if i in quarantined else "ok",
        )
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


def test_a_quarantined_article_is_part_of_the_corpus(tmp_path: Path, results_db_path: Path) -> None:
    source = tmp_path / "q.db"
    _source_with(source, [(1, "A", "One. " * 5), (2, "B", "Two. " * 5)], quarantined=(2,))
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    try:
        assert [b.article_id for b in steps.pending_bodies(conn, steps.STEP_SCOPE, REV)] == [1, 2]
    finally:
        db_module.detach_source(conn)
        conn.close()


def test_missing_titles_are_counted_per_unique_body(tmp_path: Path, results_db_path: Path) -> None:
    source = tmp_path / "t.db"
    _source_with(source, [(1, "A", "One. " * 5), (2, "", "Two. " * 5), (3, "  ", "Three. " * 5)])
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    try:
        assert steps.missing_titles(conn) == 2
    finally:
        db_module.detach_source(conn)
        conn.close()


def test_the_revision_follows_the_weights(tmp_path: Path) -> None:
    (tmp_path / "training_record.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stage_a").mkdir()
    weights = tmp_path / "stage_a" / "model.safetensors"
    weights.write_bytes(b"one")
    first = revision_of(tmp_path)
    weights.write_bytes(b"two")  # same record, refitted weights
    assert revision_of(tmp_path) != first


RECORD = {
    "stage_a": {"topics": ["company", "sector", "not_related"]},
    "stage_b": {"classes": ["single", "multi"]},
}


def test_stage_b_rows_follow_the_company_bodies() -> None:
    """Stage B was asked only of the bodies stage A called company: its rows line up with those, not with the batch."""
    topics = ["not_related", "company", "sector", "company"]
    a = {
        "company": [0.0, 5.0, 0.0, 5.0],
        "sector": [0.0, 0.0, 5.0, 0.0],
        "not_related": [5.0, 0.0, 0.0, 0.0],
    }
    b = {"single": [3.0, -3.0], "multi": [-3.0, 3.0]}  # body 1 single, body 3 multi
    got = assemble(RECORD, a, b, topics, company=[1, 3])
    assert [r.scope for r in got] == ["out_of_scope", "single", "sector", "multi"]
    assert "count:single" not in got[0].scores and "count:single" in got[3].scores


class _Stub:
    """A classifier without a model: bodies whose text starts with 'bad' make the batch fail."""

    revision = "scope-nli-stub"
    chars = 800

    def classify(self, items: list[tuple[str, str]]) -> list[ScopeResult]:
        if any(text.startswith("bad") for _, text in items):
            raise RuntimeError("boom")
        return [ScopeResult("single", {"topic:company": 1.0}) for _ in items]


def _cli() -> object:
    spec = importlib.util.spec_from_file_location(
        "news_nlp_steps_cli", Path(__file__).resolve().parents[2] / "cli" / "news_nlp_steps.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_cli_redoes_a_failed_batch_body_by_body(tmp_path: Path, results_db_path: Path) -> None:
    source = tmp_path / "cli.db"
    _source_with(
        source,
        [(1, "A", "good one. " * 5), (2, "B", "bad two. " * 5), (3, "C", "good three. " * 5)],
    )
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    try:
        cli = _cli()
        counts = cli.run_scope(conn, _Stub(), limit=None)  # type: ignore[attr-defined]
        assert counts["done"] == 2 and counts["failed"] == 1 and counts["single"] == 2
        got = steps.status(conn, steps.STEP_SCOPE, "scope-nli-stub")
        assert got["by_state"] == {"done": 2, "failed": 1}
        again = cli.run_scope(conn, _Stub(), limit=None)  # type: ignore[attr-defined]
        assert again["done"] == 0 and again["failed"] == 1  # only the failed one is retried
        assert cli.run_scope(conn, _Stub(), limit=1)["failed"] <= 1  # type: ignore[attr-defined]
    finally:
        db_module.detach_source(conn)
        conn.close()


class _Broken:
    """A classifier whose every call fails, as with a lost GPU."""

    revision = "scope-nli-broken"
    chars = 800

    def classify(self, items: list[tuple[str, str]]) -> list[ScopeResult]:
        raise RuntimeError("CUDA error")


def test_the_run_stops_when_batch_after_batch_fails_completely(
    tmp_path: Path, results_db_path: Path
) -> None:
    source = tmp_path / "broken.db"
    _source_with(source, [(i, "T", f"Body number {i}. " * 5) for i in range(1, 80)])
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    try:
        cli = _cli()
        with pytest.raises(cli.RunAborted, match="in a row"):  # type: ignore[attr-defined]
            cli.run_scope(conn, _Broken(), limit=None)  # type: ignore[attr-defined]
        got = steps.status(conn, steps.STEP_SCOPE, "scope-nli-broken")
        assert got["by_state"] == {"failed": 3 * cli.BATCH}  # type: ignore[attr-defined]
        assert "CUDA error" in got["failed_examples"][0]["error"]  # type: ignore[index]
    finally:
        db_module.detach_source(conn)
        conn.close()


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["news_nlp_steps.py", *argv])
    _cli().main()  # type: ignore[attr-defined]


def test_status_needs_no_model(
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "s.db"
    _source_with(source, [(1, "A", "One. " * 5)])
    _run_main(
        monkeypatch,
        "--step", "scope", "--status",
        "--model-dir", str(tmp_path / "none"),
        "--results-db", str(results_db_path),
        "--source-db", str(source),
    )  # fmt: skip
    assert '"by_state": {}' in capsys.readouterr().out


def test_a_run_without_the_training_record_stops_with_a_message(
    tmp_path: Path, results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "s2.db"
    _source_with(source, [(1, "A", "One. " * 5)])
    with pytest.raises(SystemExit, match="train the models first"):
        _run_main(
            monkeypatch,
            "--step", "scope",
            "--model-dir", str(tmp_path / "none"),
            "--results-db", str(results_db_path),
            "--source-db", str(source),
        )  # fmt: skip


def test_a_limit_below_one_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as stop:
        _run_main(monkeypatch, "--step", "scope", "--limit", "0")
    assert stop.value.code == 2
