"""Step 2 with DeepSeek as the summariser (T-206): the API summariser, its revision, its failures, and the step driving it. No network: ``urlopen`` is replaced."""

from __future__ import annotations

import io
import json
import sqlite3
import threading
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import news_nlp as db_module
import news_summary_llm as llm
from news_nlp import steps
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source
from tests.news_nlp.test_steps import _cli


class Reply(io.BytesIO):
    def __enter__(self) -> Reply:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _ok(text: str = "A summary.", finish: str = "stop") -> Reply:
    body = {
        "model": "api-model-1",
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    return Reply(json.dumps(body).encode())


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", None, None)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_URL", "https://api.example/v1/")
    monkeypatch.setenv("LLM_MODEL", "model-a")
    monkeypatch.setenv("LLM_API_KEY", "k")


def _script(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def fake(req: Any, timeout: int = 0) -> Reply:
        sent.append(json.loads(req.data))
        out = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(out, BaseException):
            raise out
        return Reply(out.getvalue()) if isinstance(out, Reply) else out

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    return sent


def test_the_request_has_the_prompt_the_text_the_model_the_limits_and_the_usage_says_who_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _script(monkeypatch, [_ok()])
    s = llm.LLMSummarizer(concurrency=1)
    out = s.summarize_each(["Title-T\nBody-B"])
    assert out == [("A summary.", llm.Usage("api-model-1", 10, 3, "model-a"))]
    assert sent[0]["model"] == "model-a" and sent[0]["temperature"] == 0
    assert sent[0]["max_tokens"] == llm.MAX_TOKENS
    assert sent[0]["messages"][0]["content"] == llm.prompt()
    assert sent[0]["messages"][1]["content"] == "Title-T\nBody-B"
    assert s.url == "https://api.example/v1/chat/completions"
    assert (s.prompt_tokens, s.completion_tokens) == (10, 3)
    assert s.summarize(["t"]) == [("A summary.", 0)]  # the contract of the local summariser


def test_the_step_reads_its_own_settings_and_falls_back_to_the_judges_one_by_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert llm.settings() == llm.Settings("https://api.example/v1/", "k", "model-a")
    monkeypatch.setenv("SUMMARY_LLM_MODEL", "summariser-b")
    assert (
        llm.settings().model == "summariser-b" and llm.settings().url == "https://api.example/v1/"
    )
    monkeypatch.setenv("LLM_MODEL", "the-judge")  # the judge changes: the step does not
    assert llm.LLMSummarizer().model == "summariser-b"
    monkeypatch.delenv("SUMMARY_LLM_URL", raising=False)
    monkeypatch.delenv("LLM_URL")
    monkeypatch.delenv("LLM_API_KEY")
    assert llm.missing_settings() == [
        "SUMMARY_LLM_URL (or LLM_URL)",
        "SUMMARY_LLM_API_KEY (or LLM_API_KEY)",
    ]
    with pytest.raises(KeyError, match="SUMMARY_LLM_URL"):
        llm.settings()


def test_a_new_judge_model_does_not_start_a_new_revision_of_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUMMARY_LLM_MODEL", "summariser-b")
    before = llm.LLMSummarizer().revision
    monkeypatch.setenv("LLM_MODEL", "another-judge")
    assert llm.LLMSummarizer().revision == before


def test_the_revision_follows_the_prompt_the_model_the_host_the_size_and_the_reply_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = llm.revision_of("model-a", url="https://api.example/v1")
    assert base.startswith("summary-llm-") and base == llm.revision_of(
        "model-a", url="https://api.example/v1"
    )
    assert (
        llm.revision_of("model-a", url="https://api.example/v1/") == base
    )  # a slash is not a host
    assert llm.revision_of("model-b", url="https://api.example/v1") != base
    assert (
        llm.revision_of("model-a", url="https://proxy.example/v1") != base
    )  # the same name through another host
    assert llm.revision_of("model-a", 3000, "https://api.example/v1") != base
    monkeypatch.setattr(llm, "MAX_TOKENS", 400)
    assert llm.revision_of("model-a", url="https://api.example/v1") != base
    monkeypatch.undo()
    prompt = tmp_path / "p.md"
    prompt.write_text("another prompt", encoding="utf-8")
    monkeypatch.setattr(llm, "PROMPT_FILE", prompt)
    assert llm.revision_of("model-a", url="https://api.example/v1") != base


@pytest.mark.parametrize("code", [429, 503])
def test_a_rate_limit_or_a_server_error_is_retried_then_the_text_fails_alone(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    _script(monkeypatch, [_http(code), _http(code), _ok()])
    assert llm.LLMSummarizer(concurrency=1).summarize(["t"]) == [("A summary.", 0)]
    sent = _script(monkeypatch, [_http(code)])
    out = llm.LLMSummarizer(concurrency=1).summarize_each(["t"])
    assert isinstance(out[0], llm.ApiError) and not isinstance(out[0], llm.FatalApiError)
    assert "4 attempts" in str(out[0]) and len(sent) == llm.MAX_ATTEMPTS


@pytest.mark.parametrize(("code", "word"), [(401, "key"), (402, "balance"), (422, "invalid")])
def test_a_wrong_key_no_balance_or_a_refusal_is_fatal_at_once(
    monkeypatch: pytest.MonkeyPatch, code: int, word: str
) -> None:
    sent = _script(monkeypatch, [_http(code)])
    out = llm.LLMSummarizer(concurrency=1).summarize_each(["t"])
    assert isinstance(out[0], llm.FatalApiError) and word in str(out[0])
    assert len(sent) == 1
    with pytest.raises(llm.FatalApiError, match=word):
        llm.LLMSummarizer(concurrency=1).summarize(["t"])


def test_a_reply_that_is_empty_cut_or_not_a_completion_is_an_error_not_a_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for bad in (_ok(" "), _ok(finish="length"), Reply(b'{"error": "x"}')):
        _script(monkeypatch, [bad])
        out = llm.LLMSummarizer(concurrency=1, max_calls=1).summarize_each(["t"])
        assert isinstance(out[0], llm.ApiError)


def test_a_reply_without_usage_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "model": "m",
        "choices": [{"message": {"content": "A summary."}, "finish_reason": "stop"}],
    }
    _script(monkeypatch, [Reply(json.dumps(body).encode())])
    out = llm.LLMSummarizer(concurrency=1, max_calls=1).summarize_each(["t"])
    assert isinstance(out[0], llm.ApiError) and "no usage" in str(out[0])


def test_the_cap_of_calls_counts_retries_and_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _script(monkeypatch, [_http(503)])
    out = llm.LLMSummarizer(concurrency=1, max_calls=2).summarize_each(["t"])
    assert isinstance(out[0], llm.FatalApiError) and "cap of 2 calls" in str(out[0])
    assert len(sent) == 2


def test_a_batch_keeps_the_order_of_its_texts_and_one_failure_costs_no_other_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake(req: Any, timeout: int = 0) -> Reply:
        text = json.loads(req.data)["messages"][1]["content"]
        calls.append(text)
        if text == "t3":
            raise _http(400)
        return _ok("s:" + text)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    out = llm.LLMSummarizer(concurrency=4).summarize_each([f"t{i}" for i in range(9)])
    assert [o[0] if not isinstance(o, llm.ApiError) else "ERR" for o in out] == [
        "ERR" if i == 3 else f"s:t{i}" for i in range(9)
    ]
    assert sorted(calls) == sorted(f"t{i}" for i in range(9))  # each text asked once


def test_a_batch_is_as_big_as_the_requests_at_once_and_they_really_run_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = llm.LLMSummarizer(concurrency=16)
    assert s.texts_per_batch == 16
    barrier = threading.Barrier(
        16, timeout=10
    )  # 16 requests must be in flight at the same time to pass

    def fake(req: Any, timeout: int = 0) -> Reply:
        barrier.wait()
        return _ok()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    assert all(not isinstance(o, llm.ApiError) for o in s.summarize_each(["t"] * 16))


def test_the_estimate_says_the_calls_the_tokens_and_for_deepseek_the_cost() -> None:
    line = llm.estimate([4310] * 100, "deepseek-chat")
    assert "about 100 calls" in line and "tokens in" in line and "(off-peak)" in line
    assert "$" not in llm.estimate([4310] * 100, "other-model")


# --- the step driving it -----------------------------------------------------------------------

LONG = "A long story about one company. " * 70


@pytest.fixture
def pair(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    source = tmp_path / "src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    for i, text in [(1, LONG), (2, "short " * 20), (3, LONG + " other"), (4, "x" * 120_000)]:
        seed_article(raw, id=i, title=f"T{i}", body_text=text, fetch_status="ok")
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    for article_id in (1, 2, 3, 4):
        body = next(
            b
            for b in steps.pending_bodies(conn, steps.STEP_SCOPE, "scope-x")
            if b.article_id == article_id
        )
        steps.record_scope(conn, body, "scope-x", "single", {}, "title+800")
    yield conn
    db_module.detach_source(conn)
    conn.close()


def _run(conn: db_module.NewsNlpDatabase, s: llm.LLMSummarizer, cli: Any = None) -> dict[str, int]:
    cli = cli or _cli()  # the same module as the caller's when it checks the class of an exception
    counts: dict[str, int] = cli.run_summary(conn, s, "scope-x", None, ("single",), None)
    return counts


def test_the_step_stores_the_summary_the_usage_and_keeps_a_text_over_the_limit_as_it_is(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    sent = _script(monkeypatch, [_ok("API summary.")])
    s = llm.LLMSummarizer(concurrency=2)
    assert _run(pair, s) == {"summary": 2, "original": 2, "failed": 0}
    assert len(sent) == 2  # the 120,000-character body and the short one were not sent
    by = dict(
        pair.execute(
            "SELECT text_source, COUNT(*) FROM news_summary WHERE model_revision = ? GROUP BY 1",
            (s.revision,),
        )
    )
    assert by == {"summary": 2, "original": 2}
    assert steps.status(pair, steps.STEP_SUMMARY, s.revision)["api"] == {
        "summaries": 2,
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "models": ["api-model-1"],
        "asked": ["model-a"],
    }


def test_the_status_has_no_api_figures_before_a_call_and_lists_every_model_after(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    s = llm.LLMSummarizer(concurrency=1)
    assert "api" not in steps.status(pair, steps.STEP_SUMMARY, s.revision)
    steps.record_summary(pair, "body", "h1", "r", "x", "summary", 0, ("served-1", "asked", 5, 2))
    steps.record_summary(pair, "body", "h2", "r", "x", "summary", 0, ("served-2", "asked", 5, 2))
    api = steps.status(pair, steps.STEP_SUMMARY, "r")["api"]
    assert isinstance(api, dict) and api["models"] == ["served-1", "served-2"]
    assert api["prompt_tokens"] == 10


def test_a_failed_text_fails_alone_and_nothing_is_asked_twice(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    calls: list[str] = []

    def fake(req: Any, timeout: int = 0) -> Reply:
        text = json.loads(req.data)["messages"][1]["content"]
        calls.append(text)
        if "other" in text:
            raise _http(400)
        return _ok()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    s = llm.LLMSummarizer(concurrency=2)
    assert _run(pair, s) == {"summary": 1, "original": 2, "failed": 1}
    assert len(calls) == len(set(calls))  # a batch with one bad text is not redone
    assert "HTTP 400" in str(steps.status(pair, steps.STEP_SUMMARY, s.revision)["failed_examples"])


def test_a_fatal_error_keeps_the_summaries_that_arrived_and_leaves_the_rest_pending(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    """The cap is reached in the middle of a batch: the first text is stored, the second is neither stored nor failed, the next run does only that one."""
    cli: Any = _cli()
    _script(monkeypatch, [_ok()])
    s = llm.LLMSummarizer(
        concurrency=1, max_calls=1
    )  # one request: the second text is over the cap
    s.texts_per_batch = 2  # both texts in one batch
    s.concurrency = 1
    with pytest.raises(cli.RunAborted, match="cap of 1 calls"):
        _run(pair, s, cli)
    states = dict(
        pair.execute(
            "SELECT status, COUNT(*) FROM pipeline_step_state WHERE step = 'summary' AND model_revision = ? GROUP BY 1",
            (s.revision,),
        )
    )
    assert (
        states.get("done", 0) == 3 and "failed" not in states
    )  # two originals and one summary; the other is pending
    assert (
        pair.execute(
            "SELECT COUNT(*) FROM news_summary WHERE model_revision = ? AND text_source = 'summary'",
            (s.revision,),
        ).fetchone()[0]
        == 1
    )
    _script(monkeypatch, [_ok()])
    again = llm.LLMSummarizer(concurrency=1)
    assert _run(pair, again) == {
        "summary": 1,
        "original": 0,
        "failed": 0,
    }  # only the one that was left


def test_a_payment_error_in_a_full_batch_stops_the_run_and_stores_what_arrived(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    cli: Any = _cli()

    def fake(req: Any, timeout: int = 0) -> Reply:
        if "other" in json.loads(req.data)["messages"][1]["content"]:
            raise _http(402)
        return _ok()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    s = llm.LLMSummarizer(concurrency=2)
    with pytest.raises(cli.RunAborted, match="balance"):
        _run(pair, s, cli)
    assert (
        pair.execute(
            "SELECT COUNT(*) FROM news_summary WHERE model_revision = ? AND text_source = 'summary'",
            (s.revision,),
        ).fetchone()[0]
        == 1
    )


def test_the_recipes_to_summarise_again_also_wipe_the_usage_so_the_new_figures_are_kept(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    cli: Any = _cli()
    s = llm.LLMSummarizer(concurrency=2)
    _script(monkeypatch, [_ok()])
    _run(pair, s)
    for statement in cli.RESUMMARIZE_SQL:
        assert statement in cli.RERUN_HELP
        pair.execute(statement.replace("<summary revision>", s.revision))
    pair.commit()
    assert pair.execute("SELECT COUNT(*) FROM news_summary_call").fetchone()[0] == 0
    body = {
        "model": "api-model-2",
        "choices": [{"message": {"content": "S"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1},
    }
    _script(monkeypatch, [Reply(json.dumps(body).encode())])
    _run(pair, llm.LLMSummarizer(concurrency=2))
    api = steps.status(pair, steps.STEP_SUMMARY, s.revision)["api"]
    assert isinstance(api, dict) and api["models"] == ["api-model-2"] and api["prompt_tokens"] == 14
    assert any("news_summary_call" in stmt for stmt in cli.RECUT_SQL)
    assert all(stmt in cli.RERUN_HELP for stmt in cli.RECUT_SQL)


def test_a_text_over_the_limit_is_kept_as_it_is_for_every_kind_of_item() -> None:
    cli: Any = _cli()
    s = llm.LLMSummarizer()
    too_long, fits, short = "x" * (llm.MAX_CHARS + 1), "y" * 5000, "z" * 100
    items = [
        cli.SummaryItem(kind, f"{kind}{i}", t)
        for kind in ("body", "sub_news")
        for i, t in enumerate((too_long, fits, short))
    ]
    over, kept = cli._split_by_size(items, s)
    assert [i.text for i in over] == [fits, fits] and len(kept) == 4


def test_the_command_defaults_to_the_api_and_refuses_api_options_elsewhere() -> None:
    cli: Any = _cli()
    parser = cli.build_parser()
    assert parser.parse_args(["--step", "summary"]).summariser == "deepseek"
    for argv in (
        ["--step", "scope", "--summariser", "local"],
        ["--step", "cut", "--max-calls", "5"],
        ["--step", "summary", "--summariser", "local", "--max-calls", "5"],
        ["--step", "summary", "--max-calls", "0"],
        ["--step", "summary", "--concurrency", "0"],
    ):
        with pytest.raises(SystemExit):
            cli.check_options(parser, parser.parse_args(argv))
    cli.check_options(parser, parser.parse_args(["--step", "summary", "--max-calls", "5"]))
    cli.check_options(parser, parser.parse_args(["--step", "summary", "--summariser", "local"]))


def test_without_the_api_settings_the_command_says_what_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli: Any = _cli()
    for k in ("LLM_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k)
    monkeypatch.setattr(db_module.env, "load_env_file", lambda *a, **k: None)
    args = cli.build_parser().parse_args(["--step", "summary"])
    with pytest.raises(SystemExit, match=r"SUMMARY_LLM_URL \(or LLM_URL\)"):
        cli._build_summarizer(args)


def test_the_command_builds_the_api_summariser_by_default_and_the_pinned_local_one_on_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli: Any = _cli()
    monkeypatch.setattr(db_module.env, "load_env_file", lambda *a, **k: None)
    api = cli._build_summarizer(
        cli.build_parser().parse_args(["--step", "summary", "--max-calls", "9"])
    )
    assert isinstance(api, llm.LLMSummarizer) and api.budget.max_calls == 9
    assert "leaves this machine" in capsys.readouterr().err

    import news_summary_stage as stage  # noqa: PLC0415

    class Local:
        model_name, model_revision, device, min_chars, revision = (
            "m",
            "0" * 40,
            "cpu",
            2000,
            "summary-local",
        )

    monkeypatch.setattr(stage, "Summarizer", Local)
    local = cli._build_summarizer(
        cli.build_parser().parse_args(["--step", "summary", "--summariser", "local"])
    )
    assert isinstance(local, Local) and not isinstance(local, cli.ApiSummarizing)
