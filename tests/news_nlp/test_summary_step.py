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
