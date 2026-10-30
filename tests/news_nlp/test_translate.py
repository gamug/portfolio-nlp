"""news_nlp.translate: Spanish title and text next to the tagging files, cached, never in the key."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import httpx
import pytest

from news_nlp import retag, translate
from news_nlp.triage import Company

REPLY = "<<<TITLE>>>\nApple gana\n<<<TEXT>>>\nApple ganó mucho (earnings)."


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _ok(content: str = REPLY, usage: tuple[int, int] = (10, 20)) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        body = json.loads(request.content)
        assert body["temperature"] == 0 and body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
            },
        )

    return handler


def _translator(handler: object, retries: int = 2) -> translate.LLMTranslator:
    return translate.LLMTranslator(
        "k", "m", "https://x.test", client=_client(handler), retries=retries, pause=0
    )


def test_translate_parses_title_text_and_tokens() -> None:
    got = _translator(_ok()).translate("Apple wins", "Apple earned a lot.")
    assert got == translate.Translation("Apple gana", "Apple ganó mucho (earnings).", 10, 20)


def test_translate_retries_then_succeeds_and_then_gives_up() -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return _ok()(request)  # type: ignore[operator, no-any-return]

    assert _translator(flaky).translate("t", "x").title == "Apple gana"
    with pytest.raises(translate.TranslationError, match="after 2 tries"):
        _translator(_ok("no tags here")).translate("t", "x")


def test_from_env_names_the_missing_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LLM_API_KEY", "LLM_MODEL", "LLM_URL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY, LLM_MODEL, LLM_URL"):
        translate.LLMTranslator.from_env()


def test_cache_is_keyed_by_the_english_source(tmp_path: Path) -> None:
    cache = translate.TranslationCache(tmp_path / "c.csv")
    cache.put("1", "T", "text", translate.Translation("t", "texto"), "a")
    assert cache.get("1", "T", "text", "a") == ("t", "texto", 0, 0)
    assert cache.get("1", "T", "changed", "a") is None  # the source changed: not reused
    again = translate.TranslationCache(tmp_path / "c.csv")  # survives a restart
    assert again.get("1", "T", "text", "a") is not None


def test_translate_rows_uses_the_cache_limit_and_failures(tmp_path: Path) -> None:
    rows = [{"article_id": str(i), "title": f"T{i}", "text": f"x{i}"} for i in range(1, 5)]
    cache = translate.TranslationCache(tmp_path / "c.csv")
    cache.put("1", "T1", "x1", translate.Translation("t1", "x1es"), "a")
    seen: list[str] = []

    def fake(title: str, text: str) -> translate.Translation:
        seen.append(title)
        if title == "T3":
            raise translate.TranslationError("bad reply")
        return translate.Translation(title + "es", text + "es", 5, 7)

    got = translate.translate_rows(rows, fake, cache, workers=1, limit=2, tag="a")
    assert sorted(seen) == ["T2", "T3"]  # article 1 came from the cache, 4 is beyond the limit
    assert set(got.translations) == {"1", "2"} and got.failures == {"3": "bad reply"}
    assert (got.translated, got.prompt_tokens, got.completion_tokens) == (1, 5, 7)
    assert translate.TranslationCache(tmp_path / "c.csv").get("3", "T3", "x3", "a") is None


def test_with_spanish_goes_right_after_text_and_keeps_the_rest() -> None:
    row = {"article_id": "1", "title": "T", "text": "x", "v1_label": "other", "scope": "single"}
    (out,) = translate.with_spanish([row], {"1": translate.Translation("t", "tx")})
    assert list(out) == [
        "article_id",
        "title",
        "text",
        "spanish_title",
        "spanish_text",
        "v1_label",
        "scope",
    ]
    assert out["spanish_text"] == "tx" and out["scope"] == "single"
    (again,) = translate.with_spanish([out], {})  # no new translation: the old cells stay
    assert again == out


class FakeLLM:
    tag = "t1"

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def translate(self, title: str, text: str) -> translate.Translation:
        self.calls += 1
        self.prompt_tokens += 3
        self.completion_tokens += 4
        return translate.Translation(f"ES {title}", f"ES {text}", 3, 4)


def _review(path: Path, scope: str = "") -> None:
    fields = ["review_order", "article_id", "title", "text", "v1_label", *retag.TAG_COLUMNS]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                **dict.fromkeys(fields, ""),
                "article_id": "10",
                "title": "Hi",
                "text": "Body",
                "scope": scope,
            }
        )


def _script() -> object:
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    import retag_gold_2026_10_29 as cli  # noqa: PLC0415 -- the script is not a package

    return cli


def test_cmd_translate_pilot_leaves_the_file_then_full_run_backs_it_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _script()
    review = tmp_path / retag.REVIEW_NAME
    _review(review)
    before = review.read_bytes()
    llm = FakeLLM()
    cli.cmd_translate(tmp_path, translator=llm, limit=1, yes=True)  # type: ignore[attr-defined]
    assert review.read_bytes() == before and "ES Hi" in capsys.readouterr().out
    cli.cmd_translate(tmp_path, translator=llm, yes=True)  # type: ignore[attr-defined]
    assert llm.calls == 1  # the pilot's translation came from the cache
    (row,) = list(csv.DictReader(review.open(encoding="utf-8-sig")))
    assert list(row)[:8] == [
        "review_order",
        "article_id",
        "title",
        "text",
        "spanish_title",
        "spanish_text",
        "v1_label",
        "scope",
    ]
    assert row["spanish_title"] == "ES Hi" and row["spanish_text"] == "ES Body"
    (backup,) = list((tmp_path / "backup").glob("*_before_translate_*.csv"))
    assert backup.read_bytes() == before


def test_cmd_translate_keeps_every_tag_already_made(tmp_path: Path) -> None:
    cli = _script()
    review = tmp_path / retag.REVIEW_NAME
    _review(review, scope="single")
    cli.cmd_translate(tmp_path, translator=FakeLLM(), yes=True)  # type: ignore[attr-defined]
    (row,) = list(csv.DictReader(review.open(encoding="utf-8-sig")))
    assert row["scope"] == "single" and row["spanish_text"] == "ES Body"


def test_export_adds_the_spanish_columns_and_never_pays_for_a_tagged_file(tmp_path: Path) -> None:
    cli = _script()
    v1 = {
        "review_order": "1",
        "article_id": "10",
        "ticker": "AAPL",
        "company": "Apple",
        "pub_date": "2020-01-01",
        "source_domain": "x.com",
        "title": "Hi",
        "text": "Body",
        "human_label": "other",
        "human_notes": "",
    }
    for name, row in (
        ("category_gold_review.csv", v1),
        (
            "category_gold_review_key.csv",
            {"review_order": "1", "article_id": "10", "llm_label": "other"},
        ),
    ):
        with (tmp_path / name).open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    firms = [Company("AAPL", "Apple", "x", "y")]
    llm = FakeLLM()
    cli.cmd_export(tmp_path, universe_companies=firms, translator=llm, yes=True)  # type: ignore[attr-defined]
    (row,) = list(csv.DictReader((tmp_path / retag.REVIEW_NAME).open(encoding="utf-8-sig")))
    assert list(row)[6:10] == ["title", "text", "spanish_title", "spanish_text"]
    assert "ES" not in (tmp_path / retag.KEY_NAME).read_text(encoding="utf-8-sig")
    _review(tmp_path / retag.REVIEW_NAME, scope="single")  # now tagged: refuse before translating
    other = FakeLLM()
    with pytest.raises(SystemExit):
        cli.cmd_export(tmp_path, universe_companies=firms, translator=other, yes=True)  # type: ignore[attr-defined]
    assert other.calls == 0


def test_nothing_is_bought_without_yes_or_above_max_new(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _script()
    _review(tmp_path / retag.REVIEW_NAME)
    llm = FakeLLM()
    with pytest.raises(SystemExit, match="--yes"):
        cli.cmd_translate(tmp_path, translator=llm)  # type: ignore[attr-defined]
    assert "1 articles to translate" in capsys.readouterr().out and llm.calls == 0
    with pytest.raises(SystemExit, match="--max-new 0"):
        cli.cmd_translate(tmp_path, translator=llm, yes=True, max_new=0)  # type: ignore[attr-defined]
    assert llm.calls == 0


def test_a_fatal_status_stops_the_run_and_failures_in_a_row_too(tmp_path: Path) -> None:
    seen = {"n": 0}

    def refused(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(402, text="Insufficient Balance")

    rows = [{"article_id": str(i), "title": "t", "text": "x"} for i in range(20)]
    cache = translate.TranslationCache(tmp_path / "c.csv")
    done = translate.translate_rows(
        rows, _translator(refused, retries=3).translate, cache, workers=1, tag="a"
    )
    assert "402" in done.aborted and seen["n"] == 1 and not done.failures  # one call, no retries

    def garbage(request: httpx.Request) -> httpx.Response:
        return _ok("no tags")(request)  # type: ignore[operator, no-any-return]

    done = translate.translate_rows(
        rows, _translator(garbage, retries=1).translate, cache, workers=1, tag="a"
    )
    assert done.aborted == "5 failures in a row" and len(done.failures) == 5


def test_a_cut_or_implausible_reply_is_a_failure() -> None:
    def cut(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "length", "message": {"content": REPLY}}]},
        )

    with pytest.raises(translate.TranslationError, match="max_tokens"):
        _translator(cut, retries=1).translate("t", "x")
    long_source = "word " * 100  # 500 characters: the reply of 28 is far too short
    with pytest.raises(translate.TranslationError, match="times the English"):
        _translator(_ok(), retries=1).translate("t", long_source)


def test_cache_key_has_the_model_and_prompt_tag(tmp_path: Path) -> None:
    cache = translate.TranslationCache(tmp_path / "c.csv")
    cache.put("1", "T", "x", translate.Translation("t", "tx"), tag="old")
    assert cache.get("1", "T", "x", "old") is not None
    assert cache.get("1", "T", "x", "new") is None  # another model or prompt: a miss
    assert _translator(_ok()).tag != translate.LLMTranslator("k", "m2", "u").tag


def test_legacy_rows_match_no_tag_until_adopted_and_then_a_new_prompt_misses(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.csv"
    legacy.write_text(
        f"article_id,source_hash,spanish_title,spanish_text\n1,{translate.source_hash('T', 'x')},t,tx\n",
        encoding="utf-8-sig",
    )
    cache = translate.TranslationCache(legacy)
    assert cache.untagged() == 1 and cache.get("1", "T", "x", "t1") is None
    assert cache.adopt_legacy("t1") == 1 and cache.adopt_legacy("t1") == 0
    again = translate.TranslationCache(legacy)  # persisted
    assert again.untagged() == 0 and again.get("1", "T", "x", "t1") is not None
    assert (
        again.get("1", "T", "x", "t2") is None
    )  # the prompt changed: the legacy row is not served


def test_translate_refuses_to_buy_while_the_cache_has_untagged_rows(tmp_path: Path) -> None:
    cli = _script()
    _review(tmp_path / retag.REVIEW_NAME)
    (tmp_path / translate.CACHE_NAME).write_text(
        "article_id,source_hash,tag,spanish_title,spanish_text\n9,h,,t,tx\n", encoding="utf-8-sig"
    )
    llm = FakeLLM()
    with pytest.raises(SystemExit, match="adopt-legacy"):
        cli.cmd_translate(tmp_path, translator=llm, yes=True)  # type: ignore[attr-defined]
    assert llm.calls == 0
    cache_file = tmp_path / translate.CACHE_NAME
    before = cache_file.read_bytes()
    with pytest.raises(SystemExit, match="--yes"):
        cli.cmd_adopt_legacy(tmp_path, translator=llm)  # type: ignore[attr-defined]
    assert cache_file.read_bytes() == before  # nothing changed without --yes
    cli.cmd_adopt_legacy(tmp_path, translator=llm, yes=True)  # type: ignore[attr-defined]
    cli.cmd_translate(tmp_path, translator=llm, yes=True)  # type: ignore[attr-defined]
    assert llm.calls == 1


def test_every_paid_call_is_counted_retries_and_rejected_replies_included() -> None:
    rejected = _translator(_ok("no tags", usage=(11, 22)), retries=3)
    with pytest.raises(translate.TranslationError):
        rejected.translate("t", "x")
    assert (rejected.calls, rejected.prompt_tokens, rejected.completion_tokens) == (3, 33, 66)
    good = _translator(_ok(usage=(5, 6)))
    good.translate("t", "x")
    assert (good.calls, good.prompt_tokens, good.completion_tokens) == (1, 5, 6)


def test_a_failed_article_means_nothing_is_written_and_the_backup_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _script()
    review = tmp_path / retag.REVIEW_NAME
    _review(review)
    before = review.read_bytes()

    class Failing(FakeLLM):
        def translate(self, title: str, text: str) -> translate.Translation:
            raise translate.TranslationError("bad reply")

    with pytest.raises(SystemExit, match="not written"):
        cli.cmd_translate(tmp_path, translator=Failing(), yes=True)  # type: ignore[attr-defined]
    assert review.read_bytes() == before and not (tmp_path / "backup").exists()

    class Fixed:
        def now(self) -> object:
            import datetime  # noqa: PLC0415

            return datetime.datetime(2026, 10, 30, 12, 0, 0)

    monkeypatch.setattr(cli, "datetime", Fixed())  # type: ignore[attr-defined]
    cli.cmd_translate(tmp_path, translator=FakeLLM(), yes=True)  # type: ignore[attr-defined]
    cli.cmd_translate(tmp_path, translator=FakeLLM(), yes=True)  # type: ignore[attr-defined]
    assert len(list((tmp_path / "backup").glob("*_before_translate_*.csv"))) == 2


def test_an_old_cache_without_the_tag_column_is_upgraded_in_place(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    old.write_text(
        f"article_id,source_hash,spanish_title,spanish_text\n1,{translate.source_hash('T', 'x')},t,tx\n",
        encoding="utf-8-sig",
    )
    cache = translate.TranslationCache(old)
    cache.put("2", "U", "y", translate.Translation("u", "uy"), tag="new")
    rows = list(csv.DictReader(old.open(encoding="utf-8-sig")))
    assert [(r["article_id"], r["tag"], r["spanish_text"]) for r in rows] == [
        ("1", "", "tx"),
        ("2", "new", "uy"),
    ]
    assert translate.TranslationCache(old).untagged() == 1  # adopt it before it is served


def test_a_failed_rewrite_leaves_the_cache_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "c.csv"
    cache = translate.TranslationCache(path)
    cache.put("1", "T", "x", translate.Translation("t", "tx"), tag="")  # an untagged row
    before = path.read_bytes()

    def broken(self: object, rows: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(csv.DictWriter, "writerows", broken)
    with pytest.raises(OSError, match="disk full"):
        cache.adopt_legacy("t1")
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp"))
