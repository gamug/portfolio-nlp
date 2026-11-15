"""Step 2 with DeepSeek as the summariser (T-206, owner 2026-11-15): an OpenAI-compatible ``chat/completions`` call for each text over the size.

Same contract as ``news_summary_stage.Summarizer`` (``revision``, ``min_chars``, ``summarize``) so ``run_summary`` drives both, but no torch and no GPU.
The endpoint, the key and the model come from ``LLM_URL``, ``LLM_API_KEY`` and ``LLM_MODEL`` of ``.env``; the prompt is ``config/summary_prompt.md``.

Failure policy (owner): fail fast and resumable. A rate limit, a server error or a dropped connection is retried with a growing wait (``MAX_ATTEMPTS``);
a wrong key, no balance or a refused request is a ``FatalApiError`` that stops the run at once; a text the API cannot summarise fails alone and is retried
by the next run. There is no fallback to another model and an empty summary is never stored. ``CallBudget`` is a hard cap on the requests made, retries included.

The revision is ``summary-llm-`` plus 12 hex digits of the SHA-256 of the prompt, the model name, the temperature, the size rule and the input format
(owner, 2026-11-15). The API's own model id and the tokens of each call are kept (``news_summary_call``), because the same name can be served by another
model later. At temperature 0 the text of a call is **not** repeatable (30 articles twice: 0 identical, T-206 evidence), so a revision promises
the rows, not the words.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

MIN_CHARS = 2000  # the same size as the local summariser (owner, 2026-11-12)
INPUT_FORMAT = "Title-{title}\nBody-{body}"
MAX_CHARS = (
    100_000  # a text over it is not sent: the step keeps it as the original (owner, 2026-11-15)
)
TEMPERATURE = 0
MAX_ATTEMPTS = 4
CONCURRENCY = 8
RETRYABLE = {408, 429, 500, 502, 503, 504}
HTTP_UNAUTHORIZED = (401, 403)
#: the vendor's codes that are not worth a retry, with what to read in them
API_REASONS = {402: "insufficient balance on the account", 422: "invalid parameters in the request"}
PROMPT_FILE = Path(__file__).resolve().parent.parent / "config" / "summary_prompt.md"


class ApiError(RuntimeError):
    """One text that could not be summarised: the step records it as failed and the next run retries it."""


class FatalApiError(ApiError):
    """A failure no other text will escape (wrong key, no balance, the cap of calls): the run stops."""


class Usage(NamedTuple):
    """What one call cost and who answered."""

    api_model: str
    prompt_tokens: int
    completion_tokens: int


class CallBudget:
    """The requests made so far, with a hard cap (retries included)."""

    def __init__(self, max_calls: int | None = None) -> None:
        self.max_calls = max_calls
        self.calls = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                return False
            self.calls += 1
            return True


def prompt() -> str:
    """The prompt of the step, from its tracked file."""
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def revision_of(model: str, min_chars: int = MIN_CHARS) -> str:
    """The revision of the step: the prompt, the model name, the temperature, the size rule and the input format."""
    key = json.dumps(
        {
            "prompt": prompt(),
            "model": model,
            "temperature": TEMPERATURE,
            "min_chars": min_chars,
            "max_chars": MAX_CHARS,
            "input": INPUT_FORMAT,
        },
        sort_keys=True,
    )
    return "summary-llm-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def parse_reply(d: dict[str, Any]) -> tuple[str, Usage]:
    """The summary and the usage of a chat completion; anything else is an ``ApiError`` (a proxy page, a cut reply, an empty summary)."""
    try:
        choice = d["choices"][0]
        text = choice["message"]["content"].strip()
        usage = Usage(
            str(d.get("model", "")),
            int(d["usage"]["prompt_tokens"]),
            int(d["usage"]["completion_tokens"]),
        )
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        raise ApiError(f"the reply is not a chat completion ({type(exc).__name__}: {exc})") from exc
    if choice.get("finish_reason") == "length":
        raise ApiError("the reply was cut at the token limit")
    if not text:
        raise ApiError("the reply was empty")
    return text, usage


class LLMSummarizer:
    """The summary step's summariser over an OpenAI-compatible API: ``summarize`` returns ``(summary, 0)`` for each text (no chunks)."""

    max_chars = MAX_CHARS

    def __init__(
        self,
        min_chars: int = MIN_CHARS,
        max_calls: int | None = None,
        concurrency: int = CONCURRENCY,
        sleep: float = 2.0,
    ) -> None:
        self.min_chars = min_chars
        self.model = os.environ["LLM_MODEL"]
        self.url = os.environ["LLM_URL"].rstrip("/") + "/chat/completions"
        self.revision = revision_of(self.model, min_chars)
        self.budget = CallBudget(max_calls)
        self.concurrency = concurrency
        self.sleep = sleep
        #: the usage of the texts of the last ``summarize``, in the order given (``None`` for a text that failed)
        self.last_usage: list[Usage | None] = []

    def _call(self, text: str) -> tuple[str, Usage]:
        payload = {
            "model": self.model,
            "temperature": TEMPERATURE,
            "messages": [
                {"role": "system", "content": prompt()},
                {"role": "user", "content": text},
            ],
        }
        last = "no attempt"
        for attempt in range(MAX_ATTEMPTS):
            if not self.budget.take():
                raise FatalApiError(f"the cap of {self.budget.max_calls} calls was reached")
            req = urllib.request.Request(  # noqa: S310 -- the owner's configured endpoint
                self.url,
                json.dumps(payload).encode(),
                {
                    "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                    return parse_reply(json.load(r))
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"
                if exc.code in HTTP_UNAUTHORIZED:
                    raise FatalApiError(f"{last}: the key or the endpoint is wrong") from exc
                if exc.code in API_REASONS:
                    raise FatalApiError(f"{last}: {API_REASONS[exc.code]}") from exc
                if exc.code not in RETRYABLE:
                    raise ApiError(last) from exc
            except json.JSONDecodeError as exc:  # a proxy page, a cut reply: worth another attempt
                last = f"the reply is not JSON ({exc})"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(self.sleep * 2**attempt)
        raise ApiError(f"{MAX_ATTEMPTS} attempts failed (the last: {last})")

    def summarize(self, texts: Sequence[str]) -> list[tuple[str, int]]:
        """Summarise each text with up to ``concurrency`` calls at a time; the first text that fails raises (the step then redoes the batch item by item)."""
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as pool:
            replies = list(pool.map(self._call, texts))
        self.last_usage = [u for _, u in replies]
        return [(summary, 0) for summary, _ in replies]
