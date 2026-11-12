"""Step 2, the summary (T-164, T-166; PLAN D39 (2), (3.3), D40): single bodies and sub-news over a size are summarised, the rest is
used as it is, a rerun adds nothing, a failed item is retried. The model is a stub: no network, no GPU."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

import news_nlp as db_module
import news_summary_stage as summary_stage_module  # noqa: F401  # the module under test must import without torch weights
from news_nlp import steps
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source
from tests.news_nlp.test_steps import _cli

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


def _module() -> object:
    import news_summary_stage as module  # noqa: PLC0415

    return module


# --- the rule ---------------------------------------------------------------------------------------------------------


def test_only_a_text_over_the_size_is_summarised() -> None:
    module = _module()
    assert module.MIN_CHARS == 2000  # type: ignore[attr-defined]  # the owner's size, 2026-11-12
    assert module.needs_summary("x" * 2001) is True  # type: ignore[attr-defined]
    assert module.needs_summary("x" * 2000) is False  # type: ignore[attr-defined]  # the size itself is used as it is
    assert module.needs_summary("x" * 10, min_chars=5) is True  # type: ignore[attr-defined]


def test_the_input_of_a_body_is_the_title_and_the_body_as_the_earlier_summary_run_wrote_it() -> (
    None
):
    assert _module().input_of_body("A title", "A body.") == "Title-A title\nBody-A body."  # type: ignore[attr-defined]


def test_the_revision_follows_the_model_the_size_and_the_code() -> None:
    module = _module()
    base = module.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000)  # type: ignore[attr-defined]
    assert base.startswith("summary-") and len(base) == len("summary-") + 12
    assert base == module.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 2000)  # type: ignore[attr-defined]
    assert base != module.revision_of("sshleifer/distilbart-cnn-12-6", "a4f8f3ea", 3000)  # type: ignore[attr-defined]
    assert base != module.revision_of("sshleifer/distilbart-cnn-12-6", "other", 2000)  # type: ignore[attr-defined]


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
    cli = _cli()
    counts: dict[str, int] = cli.run_summary(pair, summarizer, SCOPE_REV, CUT_REV, items, limit)  # type: ignore[attr-defined]
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
    cli = _cli()
    monkeypatch.setattr(
        cli, "SUMMARY_BATCH", 1
    )  # one text a batch: the three long items are three batches
    all_broken = StubSummarizer(broken=("A long story", "B B"))
    with pytest.raises(cli.RunAborted):  # type: ignore[attr-defined]
        cli.run_summary(pair, all_broken, SCOPE_REV, CUT_REV, ("single", "sub_news"), None)  # type: ignore[attr-defined]
    failed = pair.execute(
        "SELECT COUNT(*) FROM pipeline_step_state WHERE status = 'failed'"
    ).fetchone()[0]
    assert failed == cli.MAX_FAILED_BATCHES  # type: ignore[attr-defined]


# --- status and the command --------------------------------------------------------------------------------------------


def test_the_status_counts_the_summaries_by_source_and_kind(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _run(pair, StubSummarizer())
    report = steps.status(pair, steps.STEP_SUMMARY, SUM_REV)
    assert report["by_state"] == {"done": 7}
    assert report["news_summary"] == {"summary": 3, "original": 4}  # type: ignore[comparison-overlap]
    assert report["by_item_kind"] == {"body": 4, "sub_news": 3}  # type: ignore[comparison-overlap]


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
    _cli().main()  # type: ignore[attr-defined]
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
            _cli().main()  # type: ignore[attr-defined]
        assert message in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["news_nlp_steps.py", "--step", "summary", "--items", "all"])
    with pytest.raises(SystemExit):
        _cli().main()  # type: ignore[attr-defined]  # "all" is not a choice: single, sub_news or both by default
    assert "invalid choice" in capsys.readouterr().err


def test_the_wrapper_summarises_through_the_existing_chunk_and_reduce_and_loads_no_model_until_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
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

    monkeypatch.setattr(module, "AutoTokenizer", FakeTokenizer)  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "AutoModelForSeq2SeqLM", FakeModel)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        module,  # type: ignore[arg-type]
        "hierarchical_summarize_batch",
        lambda texts, tokenizer, model, device, **kw: [(f"s{len(t)}", 1) for t in texts],
    )
    summarizer = module.Summarizer()  # type: ignore[attr-defined]
    assert loaded == []  # a summariser is cheap to build: --status and the tests need none
    assert summarizer.revision.startswith("summary-") and summarizer.min_chars == 2000
    assert summarizer.summarize(["a" * 3000, "b" * 2500]) == [("s3000", 1), ("s2500", 1)]
    assert loaded == [
        "tok sshleifer/distilbart-cnn-12-6@a4f8f3ea906ed274767e9906dbaede7531d660ff",
        "model sshleifer/distilbart-cnn-12-6@a4f8f3ea906ed274767e9906dbaede7531d660ff",
    ]
    summarizer.summarize(["c" * 3000])
    assert len(loaded) == 2  # loaded once
