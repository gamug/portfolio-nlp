"""Step 1.2, the cut (T-163): multi bodies become sub-news, a rerun adds nothing, a body with no company text is skipped."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import cut_stage
import news_nlp as db_module
from cut_stage import UnitsExtractor, companies_of, distinct, revision_of
from news_nlp import attribution_model as am
from news_nlp import steps
from news_nlp.triage import Company
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source
from tests.news_nlp.test_steps import _cli

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
    assert base.startswith("units5-")  # the default reading since T-177 (owner, 2026-11-12)
    assert revision_of("local:two", names) != base
    assert revision_of("local:one", {**names, "ORCL": "Oracle Corporation"}) != base


def test_the_universe_is_read_from_the_source(pair: db_module.NewsNlpDatabase) -> None:
    assert all(isinstance(c, Company) for c in companies_of(pair))


class _Stub:
    """An extractor that cuts every body into two texts, or fails for the bodies it is told to."""

    revision = "units-stub"

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on

    def cut(self, body: str) -> list[str]:
        if self.fail_on is not None and (self.fail_on == "*" or self.fail_on in body):
            raise ValueError("bad piece")
        return ["one", "two"] if "multi" in body else []


def test_run_cut_counts_and_a_failed_body_does_not_stop_the_run(
    pair: db_module.NewsNlpDatabase,
) -> None:
    cli = _cli()
    counts = cli.run_cut(pair, _Stub(fail_on="Third"), SCOPE_REV, None)  # type: ignore[attr-defined]
    assert counts == {"done": 3, "no_company_text": 0, "failed": 1, "sub_news": 6}
    got = steps.status(pair, steps.STEP_CUT, "units-stub")
    assert got["by_state"] == {"done": 3, "failed": 1}
    # the stored reason has the format of step 1: Type: message
    assert got["failed_examples"][0]["error"] == "ValueError: bad piece"  # type: ignore[index]
    # a rerun retries the failed body only
    again = cli.run_cut(pair, _Stub(), SCOPE_REV, None)  # type: ignore[attr-defined]
    assert again["done"] == 1 and pair.execute("SELECT COUNT(*) FROM sub_news").fetchone()[0] == 8


def test_run_cut_stops_when_body_after_body_fails(tmp_path: Path, results_db_path: Path) -> None:
    cli = _cli()
    source = tmp_path / "many.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    for i in range(1, 30):
        seed_article(raw, id=i, body_text=f"Multi body number {i}. " * 5)
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    try:
        for body in steps.pending_bodies(conn, steps.STEP_SCOPE, SCOPE_REV):
            steps.record_scope(conn, body, SCOPE_REV, "multi", {}, "title+800")
        with pytest.raises(cli.RunAborted, match="in a row"):  # type: ignore[attr-defined]
            cli.run_cut(conn, _Stub(fail_on="*"), SCOPE_REV, None)  # type: ignore[attr-defined]
        got = steps.status(conn, steps.STEP_CUT, "units-stub")
        assert got["by_state"] == {"failed": cli.MAX_FAILED_BODIES}  # type: ignore[attr-defined]
    finally:
        db_module.detach_source(conn)
        conn.close()


def _argv(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["news_nlp_steps.py", *argv])
    _cli().main()  # type: ignore[attr-defined]


READINGS_ASKED: list[str] = []  # the reading each `--step cut` run asked the extractor for


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """``UnitsExtractor.load`` returns a stub and records the folder it was asked for."""
    asked: list[Path] = []
    READINGS_ASKED.clear()

    def load(model_dir: Path, companies: Sequence[Company], reading: str = "units") -> _Stub:
        asked.append(model_dir)
        READINGS_ASKED.append(reading)
        return _Stub()

    monkeypatch.setattr(cut_stage.UnitsExtractor, "load", staticmethod(load))
    # the stub folders hold no model files: the pin check (tested apart below) is told the folder is the pinned one
    monkeypatch.setattr(cut_stage, "folder_identity", lambda model_dir: "stub")
    monkeypatch.setattr(
        cut_stage, "is_pinned", lambda model_dir, pin_path=None, identity=None: True
    )
    monkeypatch.setattr(
        cut_stage, "pin_status", lambda model_dir, pin_path=None, identity=None: "pinned (stub)"
    )
    return asked


def _model(tmp_path: Path) -> list[str]:
    """``--model-dir`` of an existing folder (the stub extractor does not read it)."""
    folder = tmp_path / "model"
    folder.mkdir(exist_ok=True)
    return ["--model-dir", str(folder)]


def _paths(pair: db_module.NewsNlpDatabase, tmp_path: Path, results_db_path: Path) -> list[str]:
    return ["--results-db", str(results_db_path), "--source-db", str(tmp_path / "src.db")]


def test_the_cut_command_writes_sub_news_and_prints_the_counts(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stub_model: list[Path],
) -> None:
    _argv(
        monkeypatch, "--step", "cut", "--limit", "2", *_model(tmp_path),
        *_paths(pair, tmp_path, results_db_path),
    )  # fmt: skip
    out = capsys.readouterr().out
    assert f'"scope_revision": "{SCOPE_REV}"' in out and '"sub_news": 4' in out
    assert stub_model[0].name == "model"  # the folder given is the one loaded
    assert pair.execute("SELECT COUNT(*) FROM sub_news").fetchone()[0] == 4


def test_the_cut_command_reads_the_scope_revision_it_is_given(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stub_model: list[Path],
) -> None:
    _argv(
        monkeypatch, "--step", "cut", "--scope-revision", "scope-nli-other", *_model(tmp_path),
        *_paths(pair, tmp_path, results_db_path),
    )  # fmt: skip
    assert '"done": 0' in capsys.readouterr().out
    assert pair.execute("SELECT COUNT(*) FROM sub_news").fetchone()[0] == 0


def test_the_cut_command_stops_when_step_one_has_not_run(
    tmp_path: Path, results_db_path: Path, monkeypatch: pytest.MonkeyPatch, stub_model: list[Path]
) -> None:
    source = tmp_path / "e.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    raw.commit()
    raw.close()
    with pytest.raises(SystemExit, match="run --step scope first"):
        _argv(
            monkeypatch,
            "--step",
            "cut",
            "--results-db",
            str(results_db_path),
            "--source-db",
            str(source),
        )
    assert stub_model == []  # the model was not loaded


def test_the_cut_command_stops_without_the_articles_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_model: list[Path]
) -> None:
    source = tmp_path / "e2.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    raw.commit()
    raw.close()
    bare = (
        tmp_path / "bare.db"
    )  # init_schema runs, the crawler's lean articles table does not exist
    with pytest.raises(SystemExit, match="no articles table"):
        _argv(
            monkeypatch, "--step", "cut", "--scope-revision", "x",
            "--results-db", str(bare), "--source-db", str(source),
        )  # fmt: skip
    assert stub_model == []


def test_cut_status_needs_no_model_and_shows_the_latest_revision(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stub_model: list[Path],
) -> None:
    first, second, *_ = list(steps.pending_multi_bodies(pair, SCOPE_REV, "units-old"))
    steps.record_cut(pair, first, "units-old", ["a"])
    steps.record_cut(pair, second, "units-new", ["a", "b"])
    _argv(monkeypatch, "--step", "cut", "--status", *_paths(pair, tmp_path, results_db_path))
    out = capsys.readouterr().out
    assert '"revision": "units-new"' in out and '"rows": 2' in out  # old rows are not added in
    assert '"units-old"' in out  # but the revision is listed
    assert stub_model == []


def test_scope_only_flags_are_refused_for_the_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as stop:
        _argv(monkeypatch, "--step", "cut", "--titles")
    assert stop.value.code == 2


def test_the_pin_flags_are_refused_for_the_other_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    for step in ("summary", "companies", "sector", "category"):
        for flag in ("--fetch", "--allow-unpinned"):
            with pytest.raises(SystemExit) as stop:
                _argv(monkeypatch, "--step", step, flag)
            assert stop.value.code == 2


def _fake_model(folder: Path, weights: bytes = b"w") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(weights)
    (folder / "config.json").write_text("{}")


def _pin(tmp_path: Path, folder: Path) -> Path:
    pin = tmp_path / "pin.json"
    identity = am.directory_identity(folder)
    pin.write_text(
        json.dumps(
            {
                "model": "o/m",
                "revision": "r" * 40,
                "identity": identity,
                "files": ["model.safetensors", "config.json"],
            }
        )
    )
    return pin


def test_a_folder_is_pinned_when_its_identity_is_the_recorded_one(tmp_path: Path) -> None:
    _fake_model(tmp_path / "a")
    pin = _pin(tmp_path, tmp_path / "a")
    assert cut_stage.is_pinned(tmp_path / "a", pin) is True
    _fake_model(tmp_path / "b", b"other")
    assert cut_stage.is_pinned(tmp_path / "b", pin) is False
    assert "NOT the pinned" in cut_stage.pin_status(tmp_path / "b", pin)
    assert cut_stage.is_pinned(tmp_path / "a", tmp_path / "missing.json") is None


def test_fetch_pinned_moves_in_only_a_copy_with_the_pinned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_model(tmp_path / "source")
    pin = _pin(tmp_path, tmp_path / "source")
    seen: dict[str, Any] = {}

    def download(repo: str, revision: str, local_dir: str, allow_patterns: list[str]) -> str:
        seen.update(repo=repo, revision=revision, patterns=allow_patterns)
        shutil.copytree(tmp_path / "source", local_dir, dirs_exist_ok=True)
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "token")
    target = tmp_path / "models" / "attribution"
    assert cut_stage.fetch_pinned(target, pin) == am.directory_identity(tmp_path / "source")
    assert seen == {
        "repo": "o/m",
        "revision": "r" * 40,
        "patterns": ["model.safetensors", "config.json"],
    }
    assert cut_stage.is_pinned(target, pin) is True
    with pytest.raises(FileExistsError):  # a folder with files is never overwritten
        cut_stage.fetch_pinned(target, pin)


def test_fetch_pinned_deletes_a_copy_with_another_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_model(tmp_path / "source")
    pin = _pin(tmp_path, tmp_path / "source")
    _fake_model(tmp_path / "wrong", b"altered")

    def download(repo: str, revision: str, local_dir: str, allow_patterns: list[str]) -> str:
        shutil.copytree(tmp_path / "wrong", local_dir, dirs_exist_ok=True)
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "token")
    target = tmp_path / "models" / "attribution"
    with pytest.raises(RuntimeError, match="not the pinned"):
        cut_stage.fetch_pinned(target, pin)
    assert not target.exists()
    assert not (tmp_path / "models" / ".attribution.fetching").exists()


def test_fetch_pinned_says_what_is_missing_without_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_model(tmp_path / "source")
    pin = _pin(tmp_path, tmp_path / "source")
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: pytest.fail("no download")
    )
    with pytest.raises(PermissionError, match="HF_TOKEN"):
        cut_stage.fetch_pinned(tmp_path / "models" / "attribution", pin)


def test_a_failed_fetch_exits_with_one_line_not_a_stack(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model: list[Path],
) -> None:
    argv = ("--step", "cut", *_paths(pair, tmp_path, results_db_path), "--fetch")
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x")  # a full folder: never overwritten
    with pytest.raises(SystemExit, match=r"--fetch: .*not empty"):
        _argv(monkeypatch, *argv, "--model-dir", str(folder))

    def no_token(model_dir: Path, pin_path: Path | None = None) -> str:
        raise PermissionError("gamug/x is private and no Hub token is set: put HF_TOKEN in .env")

    monkeypatch.setattr(cut_stage, "fetch_pinned", no_token)
    with pytest.raises(SystemExit, match=r"--fetch: .*HF_TOKEN"):
        _argv(monkeypatch, *argv, "--model-dir", str(tmp_path / "empty"))
    assert stub_model == []


def test_the_pin_functions_do_not_hash_the_folder_again_when_given_its_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def identity_of(directory: Path) -> str:
        calls.append(directory)
        return "x"

    monkeypatch.setattr(cut_stage.am, "directory_identity", identity_of)
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({"identity": "x", "model": "o/m", "revision": "r" * 40}))
    identity = cut_stage.folder_identity(tmp_path)
    cut_stage.pin_status(tmp_path, pin, identity)
    assert cut_stage.is_pinned(tmp_path, pin, identity) is True
    assert len(calls) == 1  # one read of the weights, not three


def test_the_cut_refuses_an_unpinned_folder_unless_allowed(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model: list[Path],
) -> None:
    monkeypatch.setattr(
        cut_stage, "is_pinned", lambda model_dir, pin_path=None, identity=None: False
    )
    argv = ("--step", "cut", *_paths(pair, tmp_path, results_db_path), *_model(tmp_path))
    with pytest.raises(SystemExit, match="not the pinned attribution model"):
        _argv(monkeypatch, *argv)
    assert stub_model == []
    _argv(monkeypatch, *argv, "--allow-unpinned")
    assert len(stub_model) == 1


def test_the_default_model_folder_of_each_step() -> None:
    cli = _cli()
    assert cli.default_model_dir("cut").name == "attribution_2026_11_08"  # type: ignore[attr-defined]
    assert cli.default_model_dir("scope").name == "scope_final"  # type: ignore[attr-defined]


def test_the_quality_breakdown_counts_what_a_reader_would_worry_about() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import cut_quality_check_2026_11_11 as quality  # noqa: PLC0415 -- a script, not a package

    per_body = {"a": ["x" * 100, "y" * 200], "b": ["z" * 50], "c": ["w" * 900]}
    got = quality.summarise(per_body, {"a": 1000, "b": 60, "c": 1000}, skipped=1)
    assert got["bodies_done"] == 3 and got["skipped_share"] == 0.25
    assert got["sub_news_total"] == 4 and got["bodies_with_one_sub_news"] == 2
    assert got["sub_news_under_80_chars"] == 1  # the 50-character one
    assert got["bodies_not_cut_(longest>=80%_of_body)"] == 2  # b (50 of 60) and c (900 of 1000)
    assert quality.quantiles([])["max"] == 0.0


def test_a_model_folder_that_is_not_there_stops_with_a_message(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit, match="is not a folder"):
        _argv(
            monkeypatch, "--step", "cut", "--model-dir", str(tmp_path / "nothing"),
            *_paths(pair, tmp_path, results_db_path),
        )  # fmt: skip


def test_a_model_folder_without_weights_stops_with_a_message(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = tmp_path / "empty_model"
    empty.mkdir()
    with pytest.raises(SystemExit, match="not a trained model directory"):
        _argv(
            monkeypatch, "--step", "cut", "--model-dir", str(empty),
            *_paths(pair, tmp_path, results_db_path),
        )  # fmt: skip


def test_the_quality_script_says_so_when_there_is_no_cut_state(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import cut_quality_check_2026_11_11 as quality  # noqa: PLC0415 -- a script, not a package

    empty = sqlite3.connect(tmp_path / "e.db")
    empty.execute("CREATE TABLE pipeline_step_state (step, model_revision, finished_at)")
    with pytest.raises(SystemExit, match="no cut state"):
        quality.latest_cut_revision(empty)


def test_tiny_and_heading_only_pieces_are_counted_and_flagged() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import cut_quality_check_2026_11_11 as quality  # noqa: PLC0415 -- a script, not a package

    assert quality.is_tiny("Micron Technology —") and quality.is_tiny("Short.")
    assert not quality.is_tiny("Micron Technology — Shares ticked up 1% after an upgrade.")
    per_body = {
        "a": ["Micron Technology —", "Shares ticked up after an upgrade by the bank."],
        "b": ["x" * 100],
    }
    got = quality.summarise(per_body, {"a": 500, "b": 100}, skipped=0)
    assert got["heading_only_pieces"] == 1 and got["bodies_with_a_tiny_or_heading_piece"] == 1
    assert quality.flag_of(per_body["b"], 100) == "not_cut"
    assert quality.flag_of(per_body["a"], 500) == "tiny_piece"
    assert quality.flag_of(["x" * 200, "y" * 200], 1000) == ""


def test_the_determinism_check_counts_the_bodies_that_change() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import cut_determinism_check_2026_11_11 as det  # noqa: PLC0415 -- a script, not a package

    stored = {"a": ["x", "y"], "b": ["z"]}
    bodies = [("a", "text a"), ("b", "text b")]
    same = det.compare(lambda t: stored["a"] if t == "text a" else stored["b"], bodies, stored)
    assert same == {"bodies": 2, "differ_from_stored": 0, "differ_between_two_runs": 0}
    calls: dict[str, int] = {}

    def flips(
        text: str,
    ) -> list[str]:  # the second cut of body b differs, and both differ from the stored rows
        calls[text] = calls.get(text, 0) + 1
        return ["x", "y"] if text == "text a" else [f"z{calls[text]}"]

    got = det.compare(flips, bodies, stored)
    assert got == {"bodies": 2, "differ_from_stored": 1, "differ_between_two_runs": 1}


def test_fragments_and_several_headings_are_told_apart() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import cut_quality_check_2026_11_11 as quality  # noqa: PLC0415 -- a script, not a package

    assert quality.is_fragment("Marvell Technology rallied 12%,")
    assert not quality.is_fragment("Shares rose 3%.")
    two = "RTX — The defense name sank nearly 10%. Calix — The technology services company jumped."
    assert quality.holds_several_headings(two)
    assert not quality.holds_several_headings(
        "Micron Technology — Shares ticked up after an upgrade."
    )
    assert quality.flag_of([two], 100000) == "several_headings"


def test_the_cut_command_reads_with_units5_unless_it_is_told_otherwise(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    common = [*_model(tmp_path), *_paths(pair, tmp_path, results_db_path)]
    _argv(monkeypatch, "--step", "cut", "--limit", "1", *common)
    first = capsys.readouterr().err
    _argv(monkeypatch, "--step", "cut", "--limit", "1", "--reading", "units", *common)
    second = capsys.readouterr().err
    assert READINGS_ASKED == ["units5", "units"]
    # the run says first which reading it cuts with and the revision it writes (a cut once ran with the wrong reading unnoticed)
    assert (
        "cutting with the reading units5, revision units-stub" in first
    )  # the stub names its revision
    assert "cutting with the reading units, revision units-stub" in second


def test_the_reading_is_part_of_the_revision() -> None:
    names = {c.ticker: c.name for c in UNIVERSE}
    base = revision_of("local:one", names)
    assert base == revision_of("local:one", names, "units5") and base.startswith("units5-")
    other = revision_of("local:one", names, "units")
    assert other != base and other.startswith("units-")
    older = revision_of(
        "local:one", names, "units4"
    )  # still a reading of its own, no longer the default
    assert older.startswith("units4-") and older not in (base, other)
    with pytest.raises(ValueError, match="reading 'units2'"):
        UnitsExtractor(QuietScorer(), UNIVERSE, "local:test", "units2")  # type: ignore[arg-type]
    assert UnitsExtractor(QuietScorer(), UNIVERSE, "local:test", "units").revision != (
        UnitsExtractor(QuietScorer(), UNIVERSE, "local:test").revision
    )
    assert UnitsExtractor(QuietScorer(), UNIVERSE, "local:test").reading == "units5"


def test_the_comparison_of_two_readings_counts_faults_and_movements() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import compare_cut_readings_2026_11_11 as cmp  # noqa: PLC0415 -- a script, not a package

    two = "RTX — The defense name sank. Calix — The technology company jumped."
    a = {"b1": {"AAA": two, "BBB": "Alpha rose 3% on the day."}, "b2": {"CCC": "Gamma fell."}}
    b = {
        "b1": {"AAA": "RTX — The defense name sank.", "BBB": "Alpha rose 3% on the day."},
        "b2": {},
    }
    old = cmp.fault_counts({h: list(v.values()) for h, v in a.items()})
    new = cmp.fault_counts({h: list(v.values()) for h, v in b.items()})
    assert old["texts_with_several_headings"] == 1 and new["texts_with_several_headings"] == 0
    assert old["bodies_with_such_a_text"] == 1 and new["bodies_with_such_a_text"] == 0
    moved = cmp.movement(a, b)
    assert moved == {"lost": [("b2", "CCC")], "gained": [], "changed": [("b1", "AAA")]}
    assert cmp.compare(lambda t: {"X": t}, lambda t: {"Y": t}, [("h", "text")]) == (
        {"h": {"X": "text"}},
        {"h": {"Y": "text"}},
    )


def test_the_owner_marks_are_looked_up_by_text_and_the_outcomes_are_told_apart(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import units4_vs_owner_marks_2026_11_11 as marks_script  # noqa: PLC0415 -- a script, not a package

    (tmp_path / "faithfulness_ids.csv").write_text(
        "review_order,key,article_id,ticker,text_hash\n"
        f"1,1#AAA#1,1,AAA,{marks_script.text_hash('same text')}\n"
        f"2,1#BBB#1,1,BBB,{marks_script.text_hash('old text')}\n"
        f"3,1#CCC#1,1,CCC,{marks_script.text_hash('read already')}\n",
        encoding="utf-8",
    )
    (tmp_path / "faithfulness_to_tag.csv").write_text(
        "review_order,faithful\n1,yes\n2,no\n3,YES\n", encoding="utf-8"
    )
    marks = marks_script.read_marks(tmp_path)
    assert marks[("1", "AAA", marks_script.text_hash("same text"))] == "yes"
    assert (
        marks[("1", "CCC", marks_script.text_hash("read already"))] == "yes"
    )  # case is not a difference
    classify = marks_script.classify
    assert classify("same text", "same text", marks, "1", "AAA") == ("same", "yes", "yes")
    assert classify("old text", None, marks, "1", "BBB") == ("lost", "no", None)
    assert classify("old text", "read already", marks, "1", "CCC") == ("changed", None, "yes")
    assert classify("old text", "new text", marks, "1", "BBB") == (
        "changed",
        "no",
        None,
    )  # B unread
    assert classify(None, "new text", marks, "1", "BBB") == ("gained", None, None)


def test_the_interval_of_the_marks_redraws_bodies_and_pairs_the_two_readings() -> None:
    import units4_vs_owner_marks_2026_11_11 as marks_script  # noqa: PLC0415 -- a script, not a package

    a: dict[str, list[int]] = {}
    b: dict[str, list[int]] = {}
    for body in ("x", "y", "z"):
        marks_script.add_mark(a, body, "yes")
        marks_script.add_mark(b, body, "yes")
    marks_script.add_mark(a, "x", "no")
    marks_script.add_mark(a, "y", None)  # a text without a mark is not in the figure
    assert a == {"x": [1, 2], "y": [1, 1], "z": [1, 1]}
    same = marks_script.body_intervals(b, {k: list(v) for k, v in b.items()})
    assert same["B-A"] == (0.0, 0.0, 0.0)  # the same marks: no difference, no width
    out = marks_script.body_intervals(a, b)
    point, low, high = out["B-A"]
    assert low <= point <= high
    assert out["A"][0] == pytest.approx(3 / 4)
    assert out["B"][0] == 1.0


def test_the_help_of_the_steps_command_says_how_to_cut_again(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pending set comes from `pipeline_step_state`, so deleting `sub_news` is not enough (review of #169)."""
    monkeypatch.setattr(sys, "argv", ["news_nlp_steps.py", "--help"])
    with pytest.raises(SystemExit):
        _cli().main()  # type: ignore[attr-defined]
    out = capsys.readouterr().out
    assert "DELETE FROM pipeline_step_state WHERE step = 'cut'" in out
    assert "DELETE FROM sub_news WHERE extractor_revision" in out


def test_reading_is_refused_outside_the_cut(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--reading` is read only by the cut: for the scope step it must not pass silently (review of #169)."""
    monkeypatch.setattr(sys, "argv", ["news_nlp_steps.py", "--step", "scope", "--reading", "units"])
    with pytest.raises(SystemExit):
        _cli().main()  # type: ignore[attr-defined]
    assert "--reading only applies to --step cut" in capsys.readouterr().err


def test_units5_is_the_default_cut_reading_with_its_own_revision() -> None:
    assert "units5" in cut_stage.CUT_READINGS
    revisions = {r: revision_of("local:x", {"AAPL": "Apple"}, r) for r in cut_stage.CUT_READINGS}
    assert len(set(revisions.values())) == len(revisions)
    assert revisions["units5"].startswith("units5-")
    assert (
        cut_stage.DEFAULT_READING == "units5"
    )  # the default since the owner decided it, 2026-11-12 (T-177)


def test_the_extra_marks_are_counted_once_however_many_draw_folders_there_are(
    tmp_path: Path,
) -> None:
    """Review of #172 (R2): they were added inside the loop over the folders, so every extra mark counted once per folder."""
    import units4_vs_owner_marks_2026_11_11 as marks_script  # noqa: PLC0415 -- a script, not a package

    folders = []
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "faithfulness_ids.csv").write_text(
            f"review_order,article_id,ticker,text_hash\n1,{name},AAA,{marks_script.text_hash(name)}\n",
            encoding="utf-8",
        )
        (folder / "faithfulness_to_tag.csv").write_text(
            "review_order,faithful\n1,yes\n", encoding="utf-8"
        )
        folders.append(folder)
    extra = tmp_path / "extra.csv"
    extra.write_text("article_id,ticker,faithful,text\n9,ZZZ,no,a later text\n", encoding="utf-8")
    loaded = marks_script.load_marks(folders, extra)
    key = ("9", "ZZZ", marks_script.text_hash("a later text"))
    assert [name for name, _ in loaded] == ["a", "b"]
    assert sum(key in marks for _, marks in loaded) == 1  # once, not once per folder
    assert loaded[0][1][key] == "no"
    assert len(loaded[0][1]) == 2 and len(loaded[1][1]) == 1  # each folder keeps its own draw
    assert [len(m) for _, m in marks_script.load_marks(folders)] == [
        1,
        1,
    ]  # no extra: nothing added


def test_the_units5_sample_is_seeded_leaves_out_the_marked_texts_and_lists_what_is_read() -> None:
    import draw_units5_sample_2026_11_11 as draw_script  # noqa: PLC0415 -- a script, not a package

    def row(kind: str, ticker: str, text: str) -> dict[str, str]:
        return {
            "kind": kind,
            "body_hash": f"h{ticker}",
            "ticker": ticker,
            "text_units4": "",
            "text_units5": text,
        }

    rows = [row("gained", f"T{i}", f"text {i}") for i in range(10)]
    rows += [
        row("lost", "LOST", "gone"),
        row("changed", "T0", "text 0"),
    ]  # a lost one, and a repeat
    marked = {("T1", "text 1")}
    first = draw_script.draw(rows, marked, 4, 172)
    again = draw_script.draw(rows, marked, 4, 172)
    assert [r["ticker"] for r in first] == [
        r["ticker"] for r in again
    ]  # the same seed, the same sample
    assert len({(r["ticker"], r["text_units5"]) for r in first}) == 4  # distinct pairs
    assert all(r["kind"] != "lost" and r["ticker"] != "T1" for r in first)  # no lost, no marked
    assert [r["ticker"] for r in first] != [
        r["ticker"] for r in draw_script.draw(rows, marked, 4, 173)
    ]  # another seed, another draw
    listing = draw_script.status_list(rows, marked, first)
    reads = {r["ticker"]: r["read"] for r in listing}
    assert reads["T1"] == "marked" and "LOST" not in reads
    assert sum(r["read"] == "sample" for r in listing if r["kind"] == "gained") == 4
    assert set(reads.values()) == {"marked", "sample", "no"}
