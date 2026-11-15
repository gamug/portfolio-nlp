"""Step 2 with DeepSeek as the summariser (T-206): the API summariser, its revision, its failures, and the step driving it. No network: ``urlopen`` is replaced."""

from __future__ import annotations

import io
import json
import sqlite3
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


def test_the_request_has_the_prompt_the_text_the_model_and_temperature_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _script(monkeypatch, [_ok()])
    s = llm.LLMSummarizer(concurrency=1)
    assert s.summarize(["Title-T\nBody-B"]) == [("A summary.", 0)]
    assert sent[0]["model"] == "model-a" and sent[0]["temperature"] == 0
    assert sent[0]["messages"][0]["content"] == llm.prompt()
    assert sent[0]["messages"][1]["content"] == "Title-T\nBody-B"
    assert s.url == "https://api.example/v1/chat/completions"
    assert s.last_usage == [llm.Usage("api-model-1", 10, 3)]


def test_the_revision_follows_the_prompt_the_model_and_the_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = llm.revision_of("model-a")
    assert base.startswith("summary-llm-") and base == llm.revision_of("model-a")
    assert llm.revision_of("model-b") != base
    assert llm.revision_of("model-a", min_chars=3000) != base
    prompt = tmp_path / "p.md"
    prompt.write_text("another prompt", encoding="utf-8")
    monkeypatch.setattr(llm, "PROMPT_FILE", prompt)
    assert llm.revision_of("model-a") != base


@pytest.mark.parametrize("code", [429, 503])
def test_a_rate_limit_or_a_server_error_is_retried_then_the_text_fails_alone(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    _script(monkeypatch, [_http(code), _http(code), _ok()])
    assert llm.LLMSummarizer(concurrency=1).summarize(["t"]) == [("A summary.", 0)]
    sent = _script(monkeypatch, [_http(code)])
    with pytest.raises(llm.ApiError, match="4 attempts") as exc:
        llm.LLMSummarizer(concurrency=1).summarize(["t"])
    assert not isinstance(exc.value, llm.FatalApiError) and len(sent) == llm.MAX_ATTEMPTS


@pytest.mark.parametrize(("code", "word"), [(401, "key"), (402, "balance"), (422, "invalid")])
def test_a_wrong_key_no_balance_or_a_refusal_stops_at_once(
    monkeypatch: pytest.MonkeyPatch, code: int, word: str
) -> None:
    sent = _script(monkeypatch, [_http(code)])
    with pytest.raises(llm.FatalApiError, match=word):
        llm.LLMSummarizer(concurrency=1).summarize(["t"])
    assert len(sent) == 1


def test_a_reply_that_is_empty_cut_or_not_a_completion_is_an_error_not_a_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for bad in (_ok(" "), _ok(finish="length"), Reply(b'{"error": "x"}')):
        _script(monkeypatch, [bad])
        with pytest.raises(llm.ApiError):
            llm.LLMSummarizer(concurrency=1, max_calls=1).summarize(["t"])


def test_the_cap_of_calls_counts_retries_and_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _script(monkeypatch, [_http(503)])
    with pytest.raises(llm.FatalApiError, match="cap of 2 calls"):
        llm.LLMSummarizer(concurrency=1, max_calls=2).summarize(["t"])
    assert len(sent) == 2


def test_a_batch_keeps_the_order_of_its_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(req: Any, timeout: int = 0) -> Reply:
        return _ok("s:" + json.loads(req.data)["messages"][1]["content"])

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    got = llm.LLMSummarizer(concurrency=4).summarize([f"t{i}" for i in range(9)])
    assert [s for s, _ in got] == [f"s:t{i}" for i in range(9)]


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


def test_the_step_stores_the_summary_the_usage_and_keeps_a_text_over_the_limit_as_it_is(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    cli = _cli()
    sent = _script(monkeypatch, [_ok("API summary.")])
    s = llm.LLMSummarizer(concurrency=2)
    counts = cli.run_summary(pair, s, "scope-x", None, ("single",), None)  # type: ignore[attr-defined]
    assert counts == {"summary": 2, "original": 2, "failed": 0}
    assert len(sent) == 2  # the 120,000-character body and the short one were not sent
    by = dict(
        pair.execute(
            "SELECT text_source, COUNT(*) FROM news_summary WHERE model_revision = ? GROUP BY 1",
            (s.revision,),
        )
    )
    assert by == {"summary": 2, "original": 2}
    status = steps.status(pair, steps.STEP_SUMMARY, s.revision)
    assert status["api"] == {
        "summaries": 2,
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "models": ["api-model-1"],
    }


def test_a_fatal_api_error_stops_the_run_and_what_was_written_stays(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    cli = _cli()
    _script(monkeypatch, [_http(402)])
    s = llm.LLMSummarizer(concurrency=1)
    with pytest.raises(cli.RunAborted, match="balance"):  # type: ignore[attr-defined]
        cli.run_summary(pair, s, "scope-x", None, ("single",), None)  # type: ignore[attr-defined]
    assert (
        pair.execute("SELECT COUNT(*) FROM news_summary WHERE text_source = 'original'").fetchone()[
            0
        ]
        == 2
    )
    assert (
        pair.execute("SELECT COUNT(*) FROM news_summary WHERE text_source = 'summary'").fetchone()[
            0
        ]
        == 0
    )


def test_a_text_the_api_cannot_summarise_fails_alone_with_its_reason_and_is_retried(
    monkeypatch: pytest.MonkeyPatch, pair: db_module.NewsNlpDatabase
) -> None:
    cli = _cli()
    _script(monkeypatch, [_http(400)])
    s = llm.LLMSummarizer(concurrency=1)
    counts = cli.run_summary(pair, s, "scope-x", None, ("single",), None)  # type: ignore[attr-defined]
    assert counts["failed"] == 2 and counts["summary"] == 0
    assert "HTTP 400" in str(steps.status(pair, steps.STEP_SUMMARY, s.revision)["failed_examples"])
    _script(monkeypatch, [_ok()])
    summarizer = llm.LLMSummarizer(concurrency=1)
    again = cli.run_summary(pair, summarizer, "scope-x", None, ("single",), None)  # type: ignore[attr-defined]
    assert again["summary"] == 2 and again["failed"] == 0


def test_the_command_defaults_to_the_api_and_refuses_api_options_elsewhere() -> None:
    cli = _cli()
    parser = cli.build_parser()  # type: ignore[attr-defined]
    assert parser.parse_args(["--step", "summary"]).summariser == "deepseek"
    for argv in (
        ["--step", "scope", "--summariser", "local"],
        ["--step", "cut", "--max-calls", "5"],
        ["--step", "summary", "--summariser", "local", "--max-calls", "5"],
        ["--step", "summary", "--max-calls", "0"],
        ["--step", "summary", "--concurrency", "0"],
    ):
        with pytest.raises(SystemExit):
            cli.check_options(parser, parser.parse_args(argv))  # type: ignore[attr-defined]
    cli.check_options(parser, parser.parse_args(["--step", "summary", "--max-calls", "5"]))  # type: ignore[attr-defined]
    cli.check_options(parser, parser.parse_args(["--step", "summary", "--summariser", "local"]))  # type: ignore[attr-defined]


def test_without_the_api_settings_the_command_says_what_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    for k in ("LLM_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k)
    monkeypatch.setattr(db_module.env, "load_env_file", lambda *a, **k: None)
    args = cli.build_parser().parse_args(["--step", "summary"])  # type: ignore[attr-defined]
    with pytest.raises(SystemExit, match="LLM_URL, LLM_API_KEY, LLM_MODEL"):
        cli._build_summarizer(args)  # type: ignore[attr-defined]
