"""The DeepSeek arm of the summary experiment (T-206): how the prototype treats the API's failures. No network: ``urlopen`` is replaced."""

from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _arm() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "arm", ROOT / "scripts" / "summary_deepseek_arm_2026_11_15.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Reply(io.BytesIO):
    def __enter__(self) -> Reply:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _ok(text: str = "A summary.") -> Reply:
    body = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    return Reply(json.dumps(body).encode())


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", None, None)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_URL", "https://api.example")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")


def _script(monkeypatch: pytest.MonkeyPatch, mod: ModuleType, outcomes: list[Any]) -> list[int]:
    calls: list[int] = []

    def fake(_req: object, timeout: int = 0) -> Reply:
        calls.append(1)
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, Reply)
        return outcome

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
    return calls


def test_a_rate_limit_is_retried_and_the_summary_comes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _arm()
    calls = _script(monkeypatch, mod, [_http(429), _http(503), _ok("Done.")])
    assert mod.call("t", "b", mod.CallBudget(None), sleep=0) == ("Done.", 10, 3)
    assert len(calls) == 3


def test_a_wrong_key_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _arm()
    calls = _script(monkeypatch, mod, [_http(401)])
    with pytest.raises(mod.ApiError, match="key or the endpoint"):
        mod.call("t", "b", mod.CallBudget(None), sleep=0)
    assert len(calls) == 1


def test_a_failure_that_does_not_end_is_an_error_after_the_attempts_not_an_empty_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _arm()
    calls = _script(monkeypatch, mod, [urllib.error.URLError("down")])
    with pytest.raises(mod.ApiError, match=f"{mod.MAX_ATTEMPTS} attempts"):
        mod.call("t", "b", mod.CallBudget(None), sleep=0)
    assert len(calls) == mod.MAX_ATTEMPTS
    _script(monkeypatch, mod, [_ok("   ")])
    with pytest.raises(mod.ApiError, match="empty"):
        mod.call("t", "b", mod.CallBudget(None), sleep=0)


def test_the_cap_of_calls_is_hard_and_counts_the_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _arm()
    calls = _script(monkeypatch, mod, [_http(500)])
    with pytest.raises(mod.ApiError, match="cap of calls"):
        mod.call("t", "b", mod.CallBudget(2), sleep=0)
    assert len(calls) == 2


def test_generate_keeps_what_arrived_and_stops_after_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _arm()
    sample = [{"article_id": i, "title": f"t{i}", "body": f"b{i}"} for i in range(1, 21)]
    (tmp_path / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    outcomes = [_ok("one"), _ok("two")] + [_http(404)] * 40
    _script(monkeypatch, mod, outcomes)
    with pytest.raises(SystemExit, match="stopped with"):
        mod.generate(None)
    kept = json.loads((tmp_path / "deepseek.json").read_text(encoding="utf-8"))
    assert len(kept) >= 1 and all(r["summary"] for r in kept)  # what arrived is on disk
    assert len(kept) < len(sample)


def test_attach_does_not_overwrite_arm_d_without_force_and_keeps_a_backup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _arm()
    monkeypatch.setattr(mod, "OUT", tmp_path)
    arms = [{"article_id": 1, "arms": {"B": "b"}, "arm_chunks": {"B": 2}}]
    (tmp_path / "arms.json").write_text(json.dumps(arms), encoding="utf-8")
    (tmp_path / "deepseek.json").write_text(
        json.dumps([{"article_id": 1, "summary": "d", "in": 1, "out": 1}]), encoding="utf-8"
    )
    mod.attach(False)
    assert json.loads((tmp_path / "arms.json").read_text(encoding="utf-8"))[0]["arms"]["D"] == "d"
    assert json.loads((tmp_path / "arms_before_D.json").read_text(encoding="utf-8")) == arms
    with pytest.raises(SystemExit, match="--force"):
        mod.attach(False)
    mod.attach(True)
