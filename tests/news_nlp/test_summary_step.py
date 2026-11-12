"""Step 2, the summary (T-164, T-166; PLAN D39 (2), (3.3), D40): single bodies and sub-news over a size are summarised, the rest is
used as it is, a rerun adds nothing, a failed item is retried. The model is a stub: no network, no GPU."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import news_nlp as db_module
import news_summary_stage as stage
from news_nlp import steps
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source
from tests.news_nlp.test_steps import _cli


def _load_cli() -> Any:
    """The steps command as a module; ``_cli`` returns it as ``object``, which hides every attribute from the type checker."""
    return _cli()


SCOPE_REV = "scope-nli-test"
CUT_REV = "units4-test"
SUM_REV = "summary-test"
LONG = "A long story about one company. " * 70  # 2,240 characters
SHORT = "A short story about one company. " * 5  # 165 characters
EXACT = "x" * (
    2000 - len("Title-Exact single\nBody-")
)  # the whole input is 2,000 characters: the size itself, not over it


class StubSummarizer:
    """A summariser that never loads a model: it returns ``summary of <n> chars`` and remembers what it was given."""

    revision = SUM_REV
    min_chars = 2000
    model_name = "stub-model"
    model_revision = "0" * 40
    device = "cpu"

    def __init__(self, broken: Sequence[str] = (), empty: Sequence[str] = ()) -> None:
        self.calls: list[list[str]] = []
        self.broken = tuple(broken)
        self.empty = tuple(empty)

    def summarize(self, texts: Sequence[str]) -> list[tuple[str, int]]:
        self.calls.append(list(texts))
        if any(marker in t for t in texts for marker in self.broken):
            raise RuntimeError("CUDA out of memory")
        return [
            ("" if any(m in t for m in self.empty) else f"summary of {len(t)} chars", 2)
            for t in texts
        ]


@pytest.fixture
def pair(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    source = tmp_path / "src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    for i, title, text in [
        (1, "Long single", LONG),
        (2, "Short single", SHORT),
        (3, "Multi", "Multi body one. " * 40),
        (4, "Exact single", EXACT),
        (5, "Long single again", LONG + " Different ending."),
        (6, "Out of scope", "Not for the pipeline. " * 40),
    ]:
        seed_article(raw, id=i, title=title, body_text=text, fetch_status="ok")
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    for article_id, scope in [
        (1, "single"),
        (2, "single"),
        (3, "multi"),
        (4, "single"),
        (5, "single"),
        (6, "out_of_scope"),
    ]:
        body = next(
            b
            for b in steps.pending_bodies(conn, steps.STEP_SCOPE, SCOPE_REV)
            if b.article_id == article_id
        )
        steps.record_scope(conn, body, SCOPE_REV, scope, {}, "title+800")
    # the cut of the multi body: three sub-news, one over the size, at one revision; another revision has one more
    multi = next(steps.pending_multi_bodies(conn, SCOPE_REV, CUT_REV))
    steps.record_cut(conn, multi, CUT_REV, ["A short sub-news.", "B " * 1200, "Another short one."])
    steps.record_cut(conn, multi, "units-old", ["An old sub-news."])
    yield conn
    db_module.detach_source(conn)
    conn.close()


# --- the rule ---------------------------------------------------------------------------------------------------------


def test_only_a_text_over_the_size_is_summarised() -> None:
    module = stage
    assert module.MIN_CHARS == 2000  # the owner's size, 2026-11-12
    assert module.needs_summary("x" * 2001) is True
    assert module.needs_summary("x" * 2000) is False  # the size itself is used as it is
    assert module.needs_summary("x" * 10, min_chars=5) is True


def test_the_input_of_a_body_is_the_title_and_the_body_as_the_earlier_summary_run_wrote_it() -> (
    None
):
    assert stage.input_of_body("A title", "A body.") == "Title-A title\nBody-A body."


def test_the_revision_follows_the_model_the_size_and_the_code() -> None:
    module = stage
    base = module.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000)
    assert base.startswith("summary-") and len(base) == len("summary-") + 12
    assert base == module.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000)
    assert base != module.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 3000)
    assert base != module.revision_of("sshleifer/distilbart-cnn-12-6", "other", 2000)


# --- what is pending ----------------------------------------------------------------------------------------------------


def test_only_single_bodies_are_pending_once_each(pair: db_module.NewsNlpDatabase) -> None:
    got = list(steps.pending_single_bodies(pair, SCOPE_REV, SUM_REV))
    assert [b.article_id for b in got] == [
        1,
        2,
        4,
        5,
    ]  # not the multi body, not the out of scope one
    assert list(steps.pending_single_bodies(pair, "scope-nli-other", SUM_REV)) == []
    assert len(list(steps.pending_single_bodies(pair, SCOPE_REV, SUM_REV, limit=2))) == 2


def test_the_sub_news_of_one_cut_revision_are_pending_with_their_ids(
    pair: db_module.NewsNlpDatabase,
) -> None:
    got = list(steps.pending_sub_news(pair, CUT_REV, SUM_REV))
    assert [s.text[:8] for s in got] == ["A short ", "B B B B ", "Another "]
    assert all(isinstance(s.sub_news_id, int) for s in got)
    assert (
        len(list(steps.pending_sub_news(pair, "units-old", SUM_REV))) == 1
    )  # another revision, its own rows
    assert list(steps.pending_sub_news(pair, "units-none", SUM_REV)) == []
    assert len(list(steps.pending_sub_news(pair, CUT_REV, SUM_REV, limit=2))) == 2


def test_a_recorded_summary_is_not_pending_again_and_a_new_revision_starts_over(
    pair: db_module.NewsNlpDatabase,
) -> None:
    body = next(steps.pending_single_bodies(pair, SCOPE_REV, SUM_REV))
    steps.record_summary(pair, steps.ITEM_BODY, body.body_hash, SUM_REV, "a summary", "summary", 3)
    steps.record_summary(
        pair, steps.ITEM_BODY, body.body_hash, SUM_REV, "a summary", "summary", 3
    )  # a repeat
    rows = pair.execute(
        "SELECT item_kind, item_id, model_revision, summary_text, text_source, num_chunks FROM news_summary"
    ).fetchall()
    assert len(rows) == 1 and tuple(rows[0])[3:] == ("a summary", "summary", 3)
    assert body.body_hash not in {
        b.body_hash for b in steps.pending_single_bodies(pair, SCOPE_REV, SUM_REV)
    }
    assert body.body_hash in {
        b.body_hash for b in steps.pending_single_bodies(pair, SCOPE_REV, "summary-new")
    }
    state = pair.execute("SELECT status FROM pipeline_step_state WHERE step = 'summary'").fetchall()
    assert [r[0] for r in state] == ["done"]


# --- the run -------------------------------------------------------------------------------------------------------------


def _run(
    pair: db_module.NewsNlpDatabase,
    summarizer: StubSummarizer,
    items: tuple[str, ...] = ("single", "sub_news"),
    limit: int | None = None,
) -> dict[str, int]:
    cli = _load_cli()
    counts: dict[str, int] = cli.run_summary(pair, summarizer, SCOPE_REV, CUT_REV, items, limit)
    return counts


def test_a_long_item_is_summarised_and_a_short_one_is_used_as_it_is(
    pair: db_module.NewsNlpDatabase,
) -> None:
    stub = StubSummarizer()
    counts = _run(pair, stub)
    # 4 single bodies (2 long: 1 and 5; the short one and the exact-size one stay) + 3 sub-news (1 long)
    assert counts["summary"] == 3 and counts["original"] == 4 and counts["failed"] == 0
    rows = {
        (r[0], r[1]): tuple(r[2:])
        for r in pair.execute(
            "SELECT item_kind, item_id, summary_text, text_source, num_chunks FROM news_summary"
        )
    }
    assert len(rows) == 7
    originals = {k: v for k, v in rows.items() if v[1] == "original"}
    assert len(originals) == 4 and all(v[2] == 0 for v in originals.values())
    assert any(
        v[0] == "A short sub-news." for v in originals.values()
    )  # the text itself, stored as the item's text
    summaries = [v for v in rows.values() if v[1] == "summary"]
    assert len(summaries) == 3 and all(
        v[0].startswith("summary of ") and v[2] == 2 for v in summaries
    )
    assert {k[0] for k in rows} == {"body", "sub_news"}


def test_the_model_sees_only_the_long_items_and_in_the_input_format(
    pair: db_module.NewsNlpDatabase,
) -> None:
    stub = StubSummarizer()
    _run(pair, stub)
    seen = [t for call in stub.calls for t in call]
    assert len(seen) == 3
    assert sum(t.startswith("Title-") for t in seen) == 2  # a body: title and body
    assert any(t.startswith("B B B") for t in seen)  # a sub-news: its text alone, no title
    assert all(len(t) > 2000 for t in seen)


def test_a_rerun_adds_and_changes_nothing(pair: db_module.NewsNlpDatabase) -> None:
    _run(pair, StubSummarizer())
    before = (
        pair.execute("SELECT * FROM news_summary ORDER BY item_kind, item_id").fetchall(),
        pair.execute(
            "SELECT * FROM pipeline_step_state WHERE step = 'summary' ORDER BY item_kind, item_id"
        ).fetchall(),
    )
    again = StubSummarizer()
    counts = _run(pair, again)
    assert counts["summary"] == counts["original"] == counts["failed"] == 0 and again.calls == []
    after = (
        pair.execute("SELECT * FROM news_summary ORDER BY item_kind, item_id").fetchall(),
        pair.execute(
            "SELECT * FROM pipeline_step_state WHERE step = 'summary' ORDER BY item_kind, item_id"
        ).fetchall(),
    )
    assert before == after


def test_limit_touches_at_most_n_items_across_both_kinds(pair: db_module.NewsNlpDatabase) -> None:
    counts = _run(pair, StubSummarizer(), limit=5)
    assert counts["summary"] + counts["original"] == 5
    assert pair.execute("SELECT COUNT(*) FROM news_summary").fetchone()[0] == 5
    _run(pair, StubSummarizer(), limit=5)  # the next run takes the rest
    assert pair.execute("SELECT COUNT(*) FROM news_summary").fetchone()[0] == 7


def test_items_chooses_single_bodies_or_sub_news(pair: db_module.NewsNlpDatabase) -> None:
    _run(pair, StubSummarizer(), items=("sub_news",))
    assert {r[0] for r in pair.execute("SELECT item_kind FROM news_summary")} == {"sub_news"}
    _run(pair, StubSummarizer(), items=("single",))
    assert {r[0] for r in pair.execute("SELECT item_kind FROM news_summary")} == {
        "sub_news",
        "body",
    }


def test_a_failed_batch_is_redone_item_by_item_and_the_bad_item_is_retried_next_run(
    pair: db_module.NewsNlpDatabase,
) -> None:
    broken = StubSummarizer(broken=("Different ending",))  # the long single body 5 fails alone
    counts = _run(pair, broken, items=("single",))
    assert counts["failed"] == 1 and counts["summary"] == 1  # body 1 is summarised, body 5 failed
    failed = pair.execute(
        "SELECT item_kind, status, error FROM pipeline_step_state WHERE status = 'failed'"
    ).fetchall()
    assert (
        len(failed) == 1
        and failed[0][0] == "body"
        and "RuntimeError: CUDA out of memory" in failed[0][2]
    )
    fixed = StubSummarizer()
    counts = _run(pair, fixed, items=("single",))
    assert (
        counts["summary"] == 1 and counts["failed"] == 0 and len(fixed.calls) == 1
    )  # only the failed one is redone
    assert (
        pair.execute("SELECT COUNT(*) FROM pipeline_step_state WHERE status = 'failed'").fetchone()[
            0
        ]
        == 0
    )


def test_an_empty_summary_is_a_failure_not_a_stored_empty_text(
    pair: db_module.NewsNlpDatabase,
) -> None:
    counts = _run(pair, StubSummarizer(empty=("Different ending",)), items=("single",))
    assert counts["failed"] == 1
    assert (
        pair.execute("SELECT COUNT(*) FROM news_summary WHERE summary_text = ''").fetchone()[0] == 0
    )
    error = pair.execute(
        "SELECT error FROM pipeline_step_state WHERE status = 'failed'"
    ).fetchone()[0]
    assert "empty summary" in error


def test_the_run_stops_when_batch_after_batch_fails_completely(
    pair: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli, "SUMMARY_BATCH", 1
    )  # one text a batch: the three long items are three batches
    all_broken = StubSummarizer(broken=("A long story", "B B"))
    with pytest.raises(cli.RunAborted):
        cli.run_summary(pair, all_broken, SCOPE_REV, CUT_REV, ("single", "sub_news"), None)
    failed = pair.execute(
        "SELECT COUNT(*) FROM pipeline_step_state WHERE status = 'failed'"
    ).fetchone()[0]
    assert failed == cli.MAX_FAILED_BATCHES


# --- status and the command --------------------------------------------------------------------------------------------


def test_the_status_counts_the_summaries_by_source_and_kind(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _run(pair, StubSummarizer())
    report = steps.status(pair, steps.STEP_SUMMARY, SUM_REV)
    assert report["by_state"] == {"done": 7}
    assert report["news_summary"] == {"summary": 3, "original": 4}
    assert report["by_item_kind"] == {"body": 4, "sub_news": 3}


def test_the_command_status_needs_no_model(
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "status_src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    raw.commit()
    raw.close()
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv(
        "SOURCE_DATABASE_URL", str(source)
    )  # the connection wants a source store; the status reads none
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "news_nlp_steps.py",
            "--step",
            "summary",
            "--status",
            "--results-db",
            str(results_db_path),
        ],
    )
    _load_cli().main()
    report = json.loads(capsys.readouterr().out)
    assert report["step"] == "summary" and report["by_state"] == {}


def test_the_command_refuses_options_of_other_steps_and_a_bad_items_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for argv, message in [
        (["--step", "summary", "--reading", "units5"], "--reading only applies to --step cut"),
        (["--step", "summary", "--fetch"], "only apply to --step scope"),
        (
            ["--step", "scope", "--items", "single"],
            "--items and --cut-revision only apply to --step summary",
        ),
    ]:
        monkeypatch.setattr(sys, "argv", ["news_nlp_steps.py", *argv])
        with pytest.raises(SystemExit):
            _load_cli().main()
        assert message in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["news_nlp_steps.py", "--step", "summary", "--items", "all"])
    with pytest.raises(SystemExit):
        _load_cli().main()  # "all" is not a choice: single, sub_news or both by default
    assert "invalid choice" in capsys.readouterr().err


def test_the_wrapper_summarises_through_the_existing_chunk_and_reduce_and_loads_no_model_until_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = stage
    loaded: list[str] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name: str, revision: str) -> FakeTokenizer:
            loaded.append(f"tok {name}@{revision}")
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name: str, revision: str) -> FakeModel:
            loaded.append(f"model {name}@{revision}")
            return cls()

        def to(self, _device: object) -> FakeModel:
            return self

        def eval(self) -> FakeModel:
            return self

    monkeypatch.setattr(module, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(module, "AutoModelForSeq2SeqLM", FakeModel)
    monkeypatch.setattr(
        module,
        "hierarchical_summarize_batch",
        lambda texts, tokenizer, model, device, **kw: [(f"s{len(t)}", 1) for t in texts],
    )
    summarizer = module.Summarizer()
    assert loaded == []  # a summariser is cheap to build: --status and the tests need none
    assert summarizer.revision.startswith("summary-") and summarizer.min_chars == 2000
    assert summarizer.summarize(["a" * 3000, "b" * 2500]) == [("s3000", 1), ("s2500", 1)]
    assert loaded == [
        "tok sshleifer/distilbart-cnn-12-6@a4f8f3ea906ed274767e9906dbaede7531d660ff",
        "model sshleifer/distilbart-cnn-12-6@a4f8f3ea906ed274767e9906dbaede7531d660ff",
    ]
    summarizer.summarize(["c" * 3000])
    assert len(loaded) == 2  # loaded once


# --- review of #173 (R1, R2, R3, N1, N3, R5) -----------------------------------------------------------------------------


def test_the_revision_covers_the_source_of_the_stage_that_summarises() -> None:
    """R1 of #173: the docs say the revision hashes news_summary_stage.py; M3: and only the functions it runs of the shared v1 modules."""
    names = set(stage.hashed_sources())
    assert "news_summary_stage" in names
    assert {
        "summary_stage.hierarchical_summarize_batch",
        "chunking.chunk_text",
        "chunking.split_sentences",
    } <= names


def test_an_edit_of_shared_code_this_step_does_not_run_keeps_the_revision_and_one_it_runs_changes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3: summary_stage.py holds the v1 stage and chunking.py the NER merge; an edit there must not re-queue about 14 GPU hours."""
    base = stage.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000)
    monkeypatch.setattr(stage.chunking, "merge_char_spans", lambda spans: spans)  # NER only
    monkeypatch.setattr(
        stage.summary_stage.SummaryInference, "batch_size", lambda self: 99
    )  # the v1 stage only
    assert stage.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000) == base
    monkeypatch.setattr(
        stage.chunking, "chunk_text", lambda text, tokenizer, max_tokens=510: []
    )  # what the step runs
    assert stage.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000) != base


def _summary_argv(tmp_path: Path, results_db_path: Path, *extra: str) -> list[str]:
    return [
        "news_nlp_steps.py",
        "--step",
        "summary",
        "--results-db",
        str(results_db_path),
        "--source-db",
        str(tmp_path / "src.db"),
        *extra,
    ]


def test_sub_news_need_a_named_cut_revision_and_never_the_latest(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R2: `--cut-revision` defaulted to the latest cut, against the rule of docs/scope_training.md 8.1 ("never the latest")."""
    monkeypatch.setattr(stage, "Summarizer", StubSummarizer)
    cli = _load_cli()
    # both kinds, no revision named: refused, and the revisions the cut has are listed
    monkeypatch.setattr(sys, "argv", _summary_argv(tmp_path, results_db_path))
    with pytest.raises(SystemExit) as refused:
        cli.main()
    assert (
        "--cut-revision" in str(refused.value)
        and CUT_REV in str(refused.value)
        and "units-old" in str(refused.value)
    )
    assert pair.execute("SELECT COUNT(*) FROM news_summary").fetchone()[0] == 0  # nothing was done
    # a revision the cut never wrote is refused too
    monkeypatch.setattr(
        sys, "argv", _summary_argv(tmp_path, results_db_path, "--cut-revision", "units4-nope")
    )
    with pytest.raises(SystemExit) as unknown:
        cli.main()
    assert "units4-nope" in str(unknown.value)
    # the single bodies alone need no cut
    monkeypatch.setattr(sys, "argv", _summary_argv(tmp_path, results_db_path, "--items", "single"))
    cli.main()
    assert {r[0] for r in pair.execute("SELECT item_kind FROM news_summary")} == {"body"}
    # naming the cut reads its sub-news
    monkeypatch.setattr(
        sys, "argv", _summary_argv(tmp_path, results_db_path, "--cut-revision", CUT_REV)
    )
    cli.main()
    assert {r[0] for r in pair.execute("SELECT item_kind FROM news_summary")} == {
        "body",
        "sub_news",
    }
    capsys.readouterr()


def test_the_help_has_the_recipes_for_a_recut_and_for_summarising_again_and_they_work(
    pair: db_module.NewsNlpDatabase,
) -> None:
    """R3: step 2 is the first thing that reads sub_news, so a re-cut must delete its summaries first; the summary has its own recipe."""
    cli = _load_cli()
    assert "Nothing reads" not in cli.RERUN_HELP
    assert "<revision>" not in cli.RERUN_HELP  # M4: three different revisions, three names
    assert "<cut revision>" in cli.RERUN_HELP and "<summary revision>" in cli.RERUN_HELP
    for statement in (*cli.RECUT_SQL, *cli.RESUMMARIZE_SQL):
        assert statement in cli.RERUN_HELP
    _run(pair, StubSummarizer())
    assert pair.execute("SELECT COUNT(*) FROM news_summary").fetchone()[0] == 7
    for statement in cli.RECUT_SQL:  # before re-cutting the revision of the sub-news
        pair.execute(statement.replace("<cut revision>", CUT_REV))
    pair.commit()
    kinds = {
        r[0]: r[1]
        for r in pair.execute("SELECT item_kind, COUNT(*) FROM news_summary GROUP BY item_kind")
    }
    assert kinds == {"body": 4}  # the sub-news summaries are gone, the bodies' stay
    assert (
        pair.execute(
            "SELECT COUNT(*) FROM pipeline_step_state WHERE step = 'summary' AND item_kind = 'sub_news'"
        ).fetchone()[0]
        == 0
    )
    assert _run(pair, StubSummarizer(), items=("sub_news",))["summary"] == 1  # and they are redone
    for statement in cli.RESUMMARIZE_SQL:  # to summarise again at one revision
        pair.execute(statement.replace("<summary revision>", SUM_REV))
    pair.commit()
    assert pair.execute("SELECT COUNT(*) FROM news_summary").fetchone()[0] == 0
    counts = _run(pair, StubSummarizer())
    assert counts["summary"] + counts["original"] == 7


def test_the_command_refuses_a_model_folder_for_the_summary_which_loads_its_model_from_the_pin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """N1: `--model-dir` was accepted by the summary step and silently ignored."""
    monkeypatch.setattr(
        sys, "argv", ["news_nlp_steps.py", "--step", "summary", "--model-dir", "somewhere"]
    )
    with pytest.raises(SystemExit):
        _load_cli().main()
    assert "--model-dir does not apply to --step summary" in capsys.readouterr().err


def test_the_help_of_the_options_says_what_the_summary_does_with_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    text = cli.build_parser().format_help()
    assert "at most this many items" in text and "touch at most this many bodies" not in text
    assert "cut and summary" in text  # --scope-revision is read by both


def test_the_determinism_check_compares_with_the_stored_summaries_and_between_two_runs() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import summary_determinism_check_2026_11_12 as check  # noqa: PLC0415 -- a script, not a package

    items = [("a", "text a" * 400), ("b", "text b" * 400)]
    stored = {"a": "summary of 2400 chars", "b": "summary of 2400 chars"}
    steady = check.compare(
        lambda texts: [(f"summary of {len(t)} chars", 1) for t in texts], items, stored
    )
    assert steady == {"items": 2, "differ_from_stored": 0, "differ_between_two_runs": 0}
    counter = iter(range(100))
    flaky = check.compare(lambda texts: [(f"s{next(counter)}", 1) for _ in texts], items, stored)
    assert flaky["differ_from_stored"] == 2 and flaky["differ_between_two_runs"] == 2


def test_the_size_measure_prints_the_percentiles_and_the_table_the_threshold_was_chosen_from() -> (
    None
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import summary_size_measure_2026_11_12 as measure  # noqa: PLC0415 -- a script, not a package

    lengths = list(range(1, 101))
    assert measure.percentiles(lengths, (50, 90)) == {50: 51, 90: 91}
    table = measure.over_table(lengths, (10, 50, 100))
    assert table == [(10, 90, 0.9), (50, 50, 0.5), (100, 0, 0.0)]
    assert measure.hours(59432, 0.84) == pytest.approx(13.87, abs=0.01)


def test_the_size_measure_compares_the_whole_cut_with_its_first_bodies() -> None:
    """The sample behind the table (the first 500 bodies) read 1.6% against 3.05% for the whole cut: the script prints both, so the check can be repeated."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import summary_size_measure_2026_11_12 as measure  # noqa: PLC0415 -- a script, not a package

    raw = sqlite3.connect(":memory:")
    raw.execute("CREATE TABLE sub_news (article_id INTEGER, extractor_revision TEXT, text TEXT)")
    rows = [(1, "r", "a" * 10), (1, "r", "b" * 3000), (2, "r", "c" * 10), (3, "r", "d" * 3000)]
    rows += [(1, "other", "e" * 5000), (0, "other", "f" * 5000)]  # another revision: never read
    raw.executemany("INSERT INTO sub_news VALUES (?, ?, ?)", rows)
    assert sorted(measure.sub_news_lengths(raw, "r")) == [10, 10, 3000, 3000]
    first_two = measure.sub_news_lengths(raw, "r", first_bodies=2)  # bodies 1 and 2, by id
    assert sorted(first_two) == [10, 10, 3000]
    assert measure.sub_news_lengths(raw, "r", first_bodies=0) == []
    assert measure.sub_news_lengths(raw, "none") == []


# --- round 2 of the review of #173 (M1, M2) --------------------------------------------------------------------------------


def test_a_scope_revision_is_refused_when_only_sub_news_are_read_and_the_output_does_not_claim_it(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M1: `--scope-revision` is read only for single bodies; it was silently ignored with `--items sub_news` and still printed."""
    cli = _load_cli()
    parser = cli.build_parser()
    ok = parser.parse_args(
        ["--step", "summary", "--items", "single", "--scope-revision", SCOPE_REV]
    )
    cli.check_options(parser, ok)  # read: no error
    bad = parser.parse_args(
        [
            "--step",
            "summary",
            "--items",
            "sub_news",
            "--cut-revision",
            CUT_REV,
            "--scope-revision",
            SCOPE_REV,
        ]
    )
    with pytest.raises(SystemExit):
        cli.check_options(parser, bad)
    assert "--scope-revision names the single bodies" in capsys.readouterr().err
    monkeypatch.setattr(stage, "Summarizer", StubSummarizer)
    monkeypatch.setattr(
        sys,
        "argv",
        _summary_argv(tmp_path, results_db_path, "--items", "sub_news", "--cut-revision", CUT_REV),
    )
    cli.main()
    printed = json.loads(capsys.readouterr().out.split("}\n{")[0] + "}")
    assert "scope_revision" not in printed and printed["cut_revision"] == CUT_REV
    monkeypatch.setattr(sys, "argv", _summary_argv(tmp_path, results_db_path, "--items", "single"))
    cli.main()
    printed = json.loads(capsys.readouterr().out.split("}\n{")[0] + "}")
    assert printed["scope_revision"] == SCOPE_REV and "cut_revision" not in printed


def test_the_determinism_check_says_what_a_revision_mismatch_means_and_exits_apart() -> None:
    """M2: the rows must be at the revision of this code, or the check is not like for like; it exited 0 with a warning."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import summary_determinism_check_2026_11_12 as check  # noqa: PLC0415 -- a script, not a package

    clean = {"items": 2, "differ_from_stored": 0, "differ_between_two_runs": 0}
    dirty = {"items": 2, "differ_from_stored": 1, "differ_between_two_runs": 0}
    assert check.revision_problem("summary-a", "summary-a") is None
    problem = check.revision_problem("summary-a", "summary-b")
    assert problem and "summary-a" in problem and "summary-b" in problem
    assert check.exit_code(clean, same_revision=True, allow_other_revision=False) == 0
    assert check.exit_code(dirty, same_revision=True, allow_other_revision=False) == 1
    assert (
        check.exit_code(clean, same_revision=False, allow_other_revision=False) == 2
    )  # not like for like: not a pass
    assert (
        check.exit_code(clean, same_revision=False, allow_other_revision=True) == 0
    )  # asked for, and printed
    assert check.exit_code(dirty, same_revision=False, allow_other_revision=True) == 1


def test_the_determinism_check_main_refuses_stored_rows_of_another_revision_with_exit_2(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M2, run for real: `sys.exit("message")` exits 1, which reads as "a summary differs"; the refusal must be exit 2."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import summary_determinism_check_2026_11_12 as check  # noqa: PLC0415 -- a script, not a package

    _run(pair, StubSummarizer())  # rows at SUM_REV
    other = type("OtherCode", (StubSummarizer,), {"revision": "summary-other"})
    argv = [
        "check",
        "--results-db",
        str(results_db_path),
        "--source-db",
        str(tmp_path / "src.db"),
        "-n",
        "2",
    ]

    monkeypatch.setattr(stage, "Summarizer", other)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as refused:
        check.main()
    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert SUM_REV in err and "summary-other" in err and "not like for like" in err

    monkeypatch.setattr(sys, "argv", [*argv, "--allow-other-revision"])
    with pytest.raises(SystemExit) as allowed:
        check.main()
    assert allowed.value.code == 0  # asked for, and the stub is deterministic
    assert "not like for like" in capsys.readouterr().out  # and it is printed in the result

    monkeypatch.setattr(stage, "Summarizer", StubSummarizer)  # the same revision as the rows
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as same:
        check.main()
    assert same.value.code == 0


# --- round 3 of the review of #173 (B1, N1, N2) --------------------------------------------------------------------------


def test_normalized_code_ignores_comments_formatting_and_docstrings_and_sees_real_edits() -> None:
    plain = "def f(x):\n    return x > 5\n"
    dressed = '# a comment\ndef f( x ):\n    """A docstring, typo fixed."""\n    # another comment\n    return (x  >  5)\n'
    assert stage.normalized_code(plain) == stage.normalized_code(dressed)
    assert stage.normalized_code(plain) != stage.normalized_code(
        "def f(x):\n    return x >= 5\n"
    )  # a real edit
    assert stage.normalized_code("A = 'one'\n") == stage.normalized_code(
        'A = "one"\n'
    )  # quotes are formatting
    assert stage.normalized_code("A = 'one'\n") != stage.normalized_code(
        "A = 'two'\n"
    )  # a constant is code
    nested = (
        'class C:\n    """Doc."""\n\n    def m(self):\n        """Doc of m."""\n        return 1\n'
    )
    assert stage.normalized_code(nested) == stage.normalized_code(
        "class C:\n    def m(self):\n        return 1\n"
    )


def test_the_revision_survives_a_docstring_a_comment_or_a_reformat_and_not_a_real_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N1: a typo fix or a formatter upgrade must not make the step redo about 14 GPU hours; a change of behaviour must."""
    args = ("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000)
    base = stage.revision_of(*args)
    real_getsource = stage.inspect.getsource
    real_module = stage._read_module_source()

    def dressed(
        obj: Any,
    ) -> str:  # every hashed function gets a comment line and a docstring-like edit
        return real_getsource(obj).replace('"""', '"""(edited) ', 1) + "\n# a trailing comment\n"

    monkeypatch.setattr(stage.inspect, "getsource", dressed)
    monkeypatch.setattr(
        stage,
        "_read_module_source",
        lambda: "# new header comment\n" + real_module + "\n# footer\n",
    )
    assert stage.revision_of(*args) == base

    def edited(obj: Any) -> str:
        return real_getsource(obj).replace("num_beams=4", "num_beams=5")

    monkeypatch.setattr(stage.inspect, "getsource", edited)
    assert stage.revision_of(*args) != base


def test_the_determinism_check_is_not_a_pass_on_nothing_or_on_fewer_texts_than_stored() -> None:
    """B1: `compare` of no items is all zeros, which exit_code read as clean; a text missing from SOURCE was silently dropped."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import summary_determinism_check_2026_11_12 as check  # noqa: PLC0415 -- a script, not a package

    nothing = {"items": 0, "differ_from_stored": 0, "differ_between_two_runs": 0}
    one_of_two = {"items": 1, "differ_from_stored": 0, "differ_between_two_runs": 0}
    both = {"items": 2, "differ_from_stored": 0, "differ_between_two_runs": 0}
    kw = {"same_revision": True, "allow_other_revision": False}
    assert check.coverage_problem(2, 2) is None
    assert "found 0 of 2" in str(check.coverage_problem(2, 0)) and "found 1 of 2" in str(
        check.coverage_problem(2, 1)
    )
    assert check.exit_code(nothing, requested=2, **kw) == 2
    assert check.exit_code(one_of_two, requested=2, **kw) == 2
    assert check.exit_code(both, requested=2, **kw) == 0
    assert check.exit_code(both, requested=2, **{**kw, "same_revision": False}) == 2


def _determinism_argv(
    tmp_path: Path, results_db_path: Path, source: Path, *extra: str
) -> list[str]:
    return [
        "check",
        "--results-db",
        str(results_db_path),
        "--source-db",
        str(source),
        "-n",
        "2",
        *extra,
    ]


def test_the_determinism_check_main_exits_2_when_the_source_lacks_the_texts_and_reads_sub_news_too(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """B1 and N2, run for real through main with a stub: a source without the bodies, a source with one of the two, and --kind sub_news."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import summary_determinism_check_2026_11_12 as check  # noqa: PLC0415 -- a script, not a package

    _run(
        pair, StubSummarizer()
    )  # two long single bodies (1 and 5) and one long sub-news are stored as summaries
    monkeypatch.setattr(stage, "Summarizer", StubSummarizer)

    def source_with(name: str, ids: tuple[int, ...]) -> Path:
        path = tmp_path / name
        raw = sqlite3.connect(path)
        raw.executescript(ARTICLES_SCHEMA)
        stamp_source(raw)
        for i in ids:
            seed_article(raw, id=i, title="Long single", body_text=LONG, fetch_status="ok")
        raw.commit()
        raw.close()
        return path

    def run(source: Path, *extra: str) -> int:
        monkeypatch.setattr(
            sys, "argv", _determinism_argv(tmp_path, results_db_path, source, *extra)
        )
        with pytest.raises(SystemExit) as done:
            check.main()
        assert isinstance(
            done.value.code, int
        )  # the script exits with a number, never a message here
        return done.value.code

    assert run(source_with("none.db", ())) == 2  # no bodies: nothing compared is not a pass
    assert "found 0 of 2" in capsys.readouterr().err
    assert run(source_with("one.db", (1,))) == 2  # one of two found
    assert "found 1 of 2" in capsys.readouterr().err
    assert run(tmp_path / "src.db") == 0  # the real source: both found, equal
    out = capsys.readouterr().out
    assert "'items': 2" in out
    assert (
        "transformers" in out and "torch" in out
    )  # C3: what produced the summaries, printed with the result
    assert (
        run(tmp_path / "src.db", "--kind", "sub_news") == 0
    )  # N2: the sub-news path (string ids, the sub_news table)
    assert "'items': 1" in capsys.readouterr().out


# --- round 4 of the review of #173 (C2, C3) ---------------------------------------------------------------------------------


def test_every_command_in_the_usage_block_of_the_steps_cli_parses_and_a_summary_that_reads_sub_news_names_a_cut() -> (
    None
):
    """C2: `--step summary --limit 200` was the first example and failed since a cut must be named; nothing checked the examples."""
    import shlex  # noqa: PLC0415

    cli = _load_cli()
    prefix = "uv run cli/news_nlp_steps.py"
    lines = [ln.strip() for ln in cli.__doc__.splitlines() if ln.strip().startswith(prefix)]
    assert len(lines) >= 8  # scope, cut and summary examples are there
    parser = cli.build_parser()
    summary = 0
    for line in lines:
        args = parser.parse_args(shlex.split(line.split("#")[0])[len(prefix.split()) :])
        cli.check_options(parser, args)  # the options the step reads, none it does not
        if args.step == "summary" and not args.status:
            summary += 1
            assert args.items == "single" or args.cut_revision, line  # sub-news need a cut named
    assert summary >= 3
    usage = (
        (Path(__file__).resolve().parents[2] / "docs" / "summary_step.md")
        .read_text(encoding="utf-8")
        .splitlines()[2]
    )
    assert (
        "(--cut-revision REV | --items single)" in usage
    )  # the usage line says one of them is required
    module_doc = (
        Path(__file__).resolve().parents[2] / "docs" / "modules" / "news-nlp.md"
    ).read_text(encoding="utf-8")
    step2 = module_doc[module_doc.index("### Step 2: the summary") :].splitlines()
    # the same usage in the module doc (the final scan found it still showing the cut as optional)
    assert "(--cut-revision REV | --items single)" in step2[2]


def test_the_revision_follows_the_transformers_version_that_generates_the_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3: `transformers>=4.56` floats and beam search lives there: a lock bump must not leave old and new summaries under one revision."""
    args = ("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000)
    base = stage.revision_of(*args)
    assert stage.revision_of(*args) == base
    monkeypatch.setattr(stage.transformers, "__version__", "0.0.1")
    assert stage.revision_of(*args) != base
    monkeypatch.undo()
    monkeypatch.setattr(
        stage.torch, "__version__", "0.0.1+cpu"
    )  # not in the key: a CPU and a CUDA build share the revision
    assert stage.revision_of(*args) == base


# --- the progress bar counts what each phase does (reported by the owner, 258,226 on the bar) ----------------------------------


class RecordingBar:
    """A stand-in for tqdm that remembers each bar: its description, its total and how far it was moved."""

    bars: list[dict[str, Any]] = []  # noqa: RUF012 -- a test double, reset by the test that uses it

    def __init__(self, total: int | None = None, desc: str | None = None, **_: Any) -> None:
        self.info: dict[str, Any] = {"desc": desc, "total": total, "n": 0}
        RecordingBar.bars.append(self.info)

    def __enter__(self) -> RecordingBar:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, n: int = 1) -> None:
        self.info["n"] += n


def test_the_bars_count_the_items_stored_as_they_are_apart_from_the_ones_that_go_to_the_model(
    pair: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bar over every pending item mixed instant `original` rows (286 a second measured) with seconds per model item: the ETA meant nothing."""
    cli = _load_cli()  # one module instance: _load_cli() loads a new one on each call
    monkeypatch.setattr(cli, "tqdm", RecordingBar)
    RecordingBar.bars = []
    cli.run_summary(pair, StubSummarizer(), SCOPE_REV, CUT_REV, ("single", "sub_news"), None)
    got = {b["desc"]: (b["total"], b["n"]) for b in RecordingBar.bars}
    assert got == {
        "original": (4, 4),
        "summary": (3, 3),
    }  # 4 stored as they are, 3 over the size: 2 single bodies and 1 sub-news
    RecordingBar.bars = []
    cli.run_summary(
        pair, StubSummarizer(), SCOPE_REV, CUT_REV, ("single", "sub_news"), None
    )  # nothing pending: no bar at all
    assert RecordingBar.bars == []


def test_the_run_says_up_front_how_many_items_go_to_the_model(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(pair, StubSummarizer())
    err = capsys.readouterr().err
    assert "pending 7 items (4 single bodies, 3 sub-news)" in err
    assert "3 to summarise with the model (over 2,000 characters)" in err
    assert "4 stored as they are" in err
    _run(pair, StubSummarizer())
    assert "nothing pending" in capsys.readouterr().err
    pair.execute("DELETE FROM pipeline_step_state WHERE step = 'summary'")
    pair.execute("DELETE FROM news_summary")
    pair.commit()
    _run(pair, StubSummarizer(), limit=5)
    assert (
        "pending 5 items" in capsys.readouterr().err
    )  # the plan counts what this run will do, after --limit


def test_the_model_bar_moves_by_every_item_of_a_full_batch_and_of_the_last_partial_one(
    pair: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a batch of 2 the 3 long items are one full batch (moved inside the loop) and a last batch of 1 (moved after it)."""
    cli = _load_cli()
    monkeypatch.setattr(cli, "tqdm", RecordingBar)
    monkeypatch.setattr(cli, "SUMMARY_BATCH", 2)
    RecordingBar.bars = []
    counts = cli.run_summary(
        pair, StubSummarizer(), SCOPE_REV, CUT_REV, ("single", "sub_news"), None
    )
    assert counts["summary"] == 3 and counts["original"] == 4
    assert {b["desc"]: (b["total"], b["n"]) for b in RecordingBar.bars} == {
        "original": (4, 4),
        "summary": (3, 3),
    }


def _forget(pair: db_module.NewsNlpDatabase, text_source: str) -> None:
    """Make the items stored with this `text_source` pending again, so that a run has work of one phase only."""
    pair.execute(
        "DELETE FROM pipeline_step_state WHERE step = 'summary' AND model_revision = ? AND "
        "(item_kind, item_id) IN (SELECT item_kind, item_id FROM news_summary "
        "WHERE text_source = ? AND model_revision = ?)",
        (SUM_REV, text_source, SUM_REV),
    )
    pair.execute(
        "DELETE FROM news_summary WHERE text_source = ? AND model_revision = ?",
        (text_source, SUM_REV),
    )
    pair.commit()


@pytest.mark.parametrize(("only", "total"), [("summary", 3), ("original", 4)])
def test_a_run_with_work_in_one_phase_only_shows_that_bar_alone(
    pair: db_module.NewsNlpDatabase,
    monkeypatch: pytest.MonkeyPatch,
    only: str,
    total: int,
) -> None:
    """No bar of total 0 for the phase with nothing to do (`if kept` / `if over`): 3 items are over the size and 4 are not."""
    _run(pair, StubSummarizer())
    _forget(pair, only)
    cli = _load_cli()
    monkeypatch.setattr(cli, "tqdm", RecordingBar)
    RecordingBar.bars = []
    cli.run_summary(pair, StubSummarizer(), SCOPE_REV, CUT_REV, ("single", "sub_news"), None)
    assert [(b["desc"], b["total"], b["n"]) for b in RecordingBar.bars] == [(only, total, total)]


def test_the_model_bar_reaches_its_total_when_an_item_fails(
    pair: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bar counts the items the model was given, failed ones too; the plan line says how many there were and `failed` says how many did not make it."""
    cli = _load_cli()
    monkeypatch.setattr(cli, "tqdm", RecordingBar)
    RecordingBar.bars = []
    counts = cli.run_summary(
        pair,
        StubSummarizer(broken=("Different ending",)),
        SCOPE_REV,
        CUT_REV,
        ("single", "sub_news"),
        None,
    )
    assert counts["failed"] == 1 and counts["summary"] == 2
    assert {b["desc"]: (b["total"], b["n"]) for b in RecordingBar.bars}["summary"] == (3, 3)
