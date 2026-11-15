"""Step 2 with DeepSeek as the summariser (T-206, owner 2026-11-15): an OpenAI-compatible ``chat/completions`` call for each text over the size.

Same contract as ``news_summary_stage.Summarizer`` (``revision``, ``min_chars``, ``summarize``) so ``run_summary`` drives both, but no torch and no GPU.
The endpoint, the key and the model come from ``SUMMARY_LLM_URL``, ``SUMMARY_LLM_API_KEY`` and ``SUMMARY_LLM_MODEL`` of ``.env``, each falling back to the
``LLM_*`` one of the same name when it is not set (``settings``): the judge of ``news_nlp.eval`` reads ``LLM_*``, so a judge that is another model must not change what the
step writes with (a new model is a new revision, and a run costs money). The prompt is ``config/summary_prompt.md``.

Failure policy (owner): fail fast and resumable. A rate limit, a server error or a dropped connection is retried with a growing wait (``MAX_ATTEMPTS``);
a wrong key, no balance or a refused request is a ``FatalApiError`` that stops the run at once; a text the API cannot summarise fails alone and is retried
by the next run. There is no fallback to another model and an empty summary is never stored. ``CallBudget`` is a hard cap on the requests made, retries included.

The revision is ``summary-llm-`` plus 12 hex digits of the SHA-256 of the prompt, the model name, the host of the endpoint, the temperature, the reply limit, the size rule
and the input format (owner, 2026-11-15). The API's own model id and the tokens of each call are kept (``news_summary_call``), because the same name can be served by another
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
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

MIN_CHARS = 2000  # the same size as the local summariser (owner, 2026-11-12)
MAX_TOKENS = 300  # a summary is about 60 words (87 tokens measured); the limit caps what a runaway reply can cost
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
    asked: str = ""  # the model name the request carried (``LLM_MODEL``): the API can answer with another name (``deepseek-chat`` -> ``deepseek-flash``)


class Settings(NamedTuple):
    """The endpoint, the key and the model of the step."""

    url: str
    api_key: str
    model: str


def settings(env: Mapping[str, str] | None = None) -> Settings:
    """``SUMMARY_LLM_*`` of the environment, each falling back to the ``LLM_*`` of the judge; a missing one is a ``KeyError`` naming both."""
    env = os.environ if env is None else env

    def pick(name: str) -> str:
        value = env.get(f"SUMMARY_LLM_{name}") or env.get(f"LLM_{name}")
        if not value:
            raise KeyError(f"SUMMARY_LLM_{name} (or LLM_{name})")
        return value

    return Settings(pick("URL"), pick("API_KEY"), pick("MODEL"))


def missing_settings(env: Mapping[str, str] | None = None) -> list[str]:
    """The settings the step lacks, each as ``SUMMARY_LLM_X (or LLM_X)``, for a message to the owner."""
    env = os.environ if env is None else env
    return [
        f"SUMMARY_LLM_{n} (or LLM_{n})"
        for n in ("URL", "API_KEY", "MODEL")
        if not (env.get(f"SUMMARY_LLM_{n}") or env.get(f"LLM_{n}"))
    ]


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


def revision_of(model: str, min_chars: int = MIN_CHARS, url: str = "") -> str:
    """The revision of the step: the prompt, the model name, the host of the endpoint (the same name through another host is not the same text),
    the temperature, the reply limit, the size rule and the input format."""
    key = json.dumps(
        {
            "prompt": prompt(),
            "model": model,
            "host": urllib.parse.urlsplit(url).netloc.lower(),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
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
        if isinstance(d, dict) and d.get("choices") and "usage" not in d:
            raise ApiError(
                "the reply has no usage (tokens in and out): the endpoint is not compatible"
            ) from exc
        raise ApiError(f"the reply is not a chat completion ({type(exc).__name__}: {exc})") from exc
    if choice.get("finish_reason") == "length":
        raise ApiError("the reply was cut at the token limit")
    if not text:
        raise ApiError("the reply was empty")
    return text, usage


#: What a text came to: its summary with what the call cost, or the error that ended it.
Outcome = tuple[str, Usage] | ApiError

#: The rates the estimate uses (USD per million tokens, cache miss; off-peak and peak) of https://api-docs.deepseek.com/quick_start/pricing/ on 2026-11-15:
#: an estimate to read before a run, not a bill; confirm them on the page.
RATES_PER_MILLION = {"input": (0.15, 0.30), "output": (0.60, 1.20)}
CHARS_PER_TOKEN = 4.31  # measured on the 300 articles of the T-206 evidence
PROMPT_OVERHEAD_TOKENS = 85  # the system prompt and the framing of one call
OUTPUT_TOKENS = 87  # a summary, measured on the same articles


def estimate(chars: Sequence[int], model: str) -> str:
    """One line to say before a run: how many calls, about how many tokens in and out and, for a DeepSeek model, about what it costs."""
    calls = len(chars)
    tokens_in = int(sum(chars) / CHARS_PER_TOKEN) + PROMPT_OVERHEAD_TOKENS * calls
    tokens_out = OUTPUT_TOKENS * calls
    line = f"about {calls:,} calls, {tokens_in:,} tokens in and {tokens_out:,} out"
    if model.lower().startswith("deepseek"):
        low, high = (
            (
                tokens_in * RATES_PER_MILLION["input"][i]
                + tokens_out * RATES_PER_MILLION["output"][i]
            )
            / 1e6
            for i in (0, 1)
        )
        line += f", about ${low:,.2f} (off-peak) to ${high:,.2f} (peak) at the rates of 2026-11-15"
    return line


class LLMSummarizer:
    """The summary step's summariser over an OpenAI-compatible API. ``summarize_each`` is what the step calls: one ``Outcome`` for each text, so a text that failed
    does not cost the others of its batch; ``summarize`` is the plain contract of the local summariser (the first error raises)."""

    max_chars = MAX_CHARS

    def __init__(
        self,
        min_chars: int = MIN_CHARS,
        max_calls: int | None = None,
        concurrency: int = CONCURRENCY,
        sleep: float = 2.0,
    ) -> None:
        self.min_chars = min_chars
        configured = settings()
        self.model = configured.model
        self.url = configured.url.rstrip("/") + "/chat/completions"
        self._key = configured.api_key
        self.revision = revision_of(self.model, min_chars, configured.url)
        self.budget = CallBudget(max_calls)
        self.concurrency = max(1, concurrency)
        #: the texts the step gives at once: as many as there are requests at once (a batch waits for its slowest call)
        self.texts_per_batch = self.concurrency
        self.sleep = sleep
        self.prompt_tokens = 0  # of this run, every call that answered
        self.completion_tokens = 0
        self._lock = threading.Lock()

    def _call(self, text: str) -> tuple[str, Usage]:
        payload = {
            "model": self.model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
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
                {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                    reply = parse_reply(json.load(r))
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
            else:
                with self._lock:
                    self.prompt_tokens += reply[1].prompt_tokens
                    self.completion_tokens += reply[1].completion_tokens
                return reply[0], reply[1]._replace(asked=self.model)
            time.sleep(self.sleep * 2**attempt)
        raise ApiError(f"{MAX_ATTEMPTS} attempts failed (the last: {last})")

    def _outcome(self, text: str) -> Outcome:
        try:
            return self._call(text)
        except ApiError as exc:
            return exc

    def summarize_each(self, texts: Sequence[str]) -> list[Outcome]:
        """One ``Outcome`` for each text, in the order given, with up to ``concurrency`` calls at a time. Nothing raises: the step stores the summaries that
        arrived, records the texts that failed, and stops at the first ``FatalApiError``."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._outcome, texts))

    def summarize(self, texts: Sequence[str]) -> list[tuple[str, int]]:
        """The contract of the local summariser, ``(summary, 0)`` for each text; the first text that failed raises its error."""
        out: list[tuple[str, int]] = []
        for outcome in self.summarize_each(texts):
            if isinstance(outcome, ApiError):
                raise outcome
            out.append((outcome[0], 0))
        return out
