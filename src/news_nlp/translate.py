"""Spanish title and text next to every tagging file (Work item 18, T-146b and later gold exports).

The owner reads the English finance jargon with difficulty, so each file to tag carries
``spanish_title`` and ``spanish_text`` right after ``text``. :class:`LLMTranslator` calls the
OpenAI-compatible chat endpoint (``LLM_API_KEY`` / ``LLM_MODEL`` / ``LLM_URL``, DeepSeek in
practice) once per article; :class:`TranslationCache` keeps every result on disk keyed by the hash of
the English source, the model and the prompt, so a re-run or a later export never pays twice. The
translation is a reading aid for the owner: it never enters the key file or any model input. The
article is untrusted text given to a model together with an instruction, so a hostile article could
bend its own translation; the English stays next to it and the reply format is checked, which is
enough for a reading aid. Paying is explicit: callers confirm the number of new articles first
(:func:`pending`, :func:`estimate`), a fatal API error (key, balance) stops the whole run, and so
do several failures in a row.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import httpx

SPANISH_COLUMNS = ("spanish_title", "spanish_text")
CACHE_NAME = "gold_translations_es.csv"
_CACHE_FIELDS = ("article_id", "source_hash", "tag", "spanish_title", "spanish_text")
#: What the 720 run used per article (558,469 prompt and 641,963 completion tokens in 717 calls).
PROMPT_TOKENS_PER_ARTICLE = 800
COMPLETION_TOKENS_PER_ARTICLE = 900
#: A translation outside these multiples of the English length is cut or invented: not accepted.
MIN_RATIO, MAX_RATIO = 0.7, 2.0
_RATIO_FROM = 200  # shorter sources have no meaningful ratio
_FATAL_STATUS = (401, 402, 403)
CONSECUTIVE_FAILURES = 5

SYSTEM_PROMPT = (
    "You translate English financial-news articles into Spanish for a Spanish-speaking reader who "
    "is not a finance specialist and will classify the article afterwards. Translate faithfully "
    "and completely: do not summarise, shorten, explain, comment or add anything. Keep numbers, "
    "dates, currency symbols, tickers (for example AAPL), company names, product names and proper "
    "nouns unchanged. For finance jargon use the standard Spanish term and, the first time it "
    "appears, put the English original in parentheses, for example 'recompra de acciones "
    "(buyback)'. Reply in exactly this format and nothing else:\n"
    "<<<TITLE>>>\n<translated title>\n<<<TEXT>>>\n<translated text>"
)
_REPLY = re.compile(r"<<<TITLE>>>\s*(.*?)\s*<<<TEXT>>>\s*(.*)\Z", re.DOTALL)


class Translation(NamedTuple):
    title: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TranslationError(RuntimeError):
    """The model did not return a usable translation."""


class FatalTranslationError(TranslationError):
    """The API refused the call for a reason a retry cannot fix (key, balance): stop the run."""


def source_hash(title: str, text: str) -> str:
    return hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()


@dataclass
class LLMTranslator:
    api_key: str
    model: str
    url: str
    client: httpx.Client | None = None
    retries: int = 3
    pause: float = 2.0
    #: Every paid call, retried or rejected replies included (a rejected reply is paid too).
    calls: int = field(default=0, init=False)
    prompt_tokens: int = field(default=0, init=False)
    completion_tokens: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def tag(self) -> str:
        """Identifies the model and the prompt that make a translation (part of the cache key)."""
        return hashlib.sha256(f"{self.model}\n{SYSTEM_PROMPT}".encode()).hexdigest()[:12]

    @classmethod
    def from_env(cls) -> LLMTranslator:
        missing = [k for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_URL") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"set {', '.join(missing)} (see .env.example) to translate")
        return cls(os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"], os.environ["LLM_URL"])

    def translate(self, title: str, text: str) -> Translation:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 8000,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<<<TITLE>>>\n{title}\n<<<TEXT>>>\n{text}"},
            ],
        }
        client = self.client or httpx.Client(timeout=180)
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._attempt(client, body, text)
            except FatalTranslationError:
                raise
            except (httpx.HTTPError, TranslationError, KeyError, ValueError) as exc:
                last = exc
                time.sleep(self.pause * (attempt + 1))
        raise TranslationError(f"translation failed after {self.retries} tries: {last}")

    def _attempt(self, client: httpx.Client, body: dict[str, Any], text: str) -> Translation:
        response = client.post(
            self.url.rstrip("/") + "/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        _fail_if_fatal(response)
        response.raise_for_status()
        data = response.json()
        self._count(data)
        done = self._parse(data)
        _check_length(text, done)
        return done

    def _count(self, data: Mapping[str, Any]) -> None:
        usage = data.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))

    @staticmethod
    def _parse(data: Mapping[str, Any]) -> Translation:
        if data["choices"][0].get("finish_reason") == "length":
            raise TranslationError("the reply was cut at max_tokens")
        content = data["choices"][0]["message"]["content"]
        match = _REPLY.search(str(content).strip())
        if not match or not match.group(2).strip():
            raise TranslationError("reply is not in the <<<TITLE>>> / <<<TEXT>>> format")
        usage = data.get("usage") or {}
        return Translation(
            match.group(1).strip(),
            match.group(2).strip(),
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )


def _fail_if_fatal(response: httpx.Response) -> None:
    if response.status_code in _FATAL_STATUS:
        raise FatalTranslationError(
            f"the API answered {response.status_code} (key or balance): {response.text[:200]}"
        )


def _check_length(source: str, done: Translation) -> None:
    if len(source) < _RATIO_FROM:
        return
    ratio = len(done.text) / len(source)
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        raise TranslationError(f"the Spanish text is {ratio:.2f} times the English one")


class TranslationCache:
    """``article_id`` -> translation, valid while the English source, the model and the prompt
    (``tag``) are unchanged. Rows written before the tag existed have an empty tag and match no tag
    until :meth:`adopt_legacy` stamps them. Appended to disk at every put, so an interrupted run
    loses nothing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, str]] = {}
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                old_header = reader.fieldnames or []
                for row in reader:
                    row.setdefault("tag", "")
                    self._rows[row["article_id"]] = row
            if "tag" not in old_header:  # a cache written before the tag column: add it once
                self._rewrite()

    def _rewrite(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CACHE_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows.values())

    def untagged(self) -> int:
        """How many cached rows were written before the tag existed."""
        return sum(1 for row in self._rows.values() if not row["tag"])

    def adopt_legacy(self, tag: str) -> int:
        """Stamp the untagged rows with *tag* (the model and prompt known to have made them) and
        return how many there were. A deliberate one-off: afterwards every row is checked."""
        with self._lock:
            n = 0
            for row in self._rows.values():
                if not row["tag"]:
                    row["tag"] = tag
                    n += 1
            if n:
                self._rewrite()
            return n

    def get(self, article_id: str, title: str, text: str, tag: str) -> Translation | None:
        row = self._rows.get(article_id)
        if row and row["source_hash"] == source_hash(title, text) and row["tag"] == tag:
            return Translation(row["spanish_title"], row["spanish_text"])
        return None

    def put(self, article_id: str, title: str, text: str, done: Translation, tag: str) -> None:
        row = {
            "article_id": article_id,
            "source_hash": source_hash(title, text),
            "tag": tag,
            "spanish_title": done.title,
            "spanish_text": done.text,
        }
        with self._lock:
            new = not self.path.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CACHE_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
            self._rows[article_id] = row


class Outcome(NamedTuple):
    translations: dict[str, Translation]
    failures: dict[str, str]
    translated: int
    prompt_tokens: int
    completion_tokens: int
    aborted: str = ""  # why the run stopped early, if it did


def pending(
    rows: Sequence[Mapping[str, str]], cache: TranslationCache, tag: str
) -> list[Mapping[str, str]]:
    """The rows with no valid cached translation: what a run would pay for."""
    return [r for r in rows if not cache.get(r["article_id"], r["title"], r["text"], tag)]


def estimate(articles: int) -> tuple[int, int]:
    """(prompt, completion) tokens of *articles* new translations, from the 720 run. They scale
    with the length of the articles (those averaged about 2,900 characters): longer texts cost more."""
    return articles * PROMPT_TOKENS_PER_ARTICLE, articles * COMPLETION_TOKENS_PER_ARTICLE


def translate_rows(
    rows: Sequence[Mapping[str, str]],
    translator: Callable[[str, str], Translation],
    cache: TranslationCache,
    *,
    workers: int = 8,
    limit: int | None = None,
    tag: str,
) -> Outcome:
    """Translations for the rows (by ``article_id``), from the cache where the source is unchanged
    and from *translator* otherwise; at most *limit* new translations. A failed article is listed in
    ``failures`` and left out; it is not retried here and nothing about it is cached. A fatal API
    error, or :data:`CONSECUTIVE_FAILURES` failures in a row, stops the run (``aborted``): the
    articles not started are neither paid for nor listed as failures."""
    found: dict[str, Translation] = {}
    todo: list[Mapping[str, str]] = []
    for row in rows:
        cached = cache.get(row["article_id"], row["title"], row["text"], tag)
        if cached:
            found[row["article_id"]] = cached
        else:
            todo.append(row)
    if limit is not None:
        todo = todo[:limit]
    failures: dict[str, str] = {}
    usage = [0, 0, 0, 0]  # prompt tokens, completion tokens, failures in a row, translated
    stop: list[str] = []
    lock = threading.Lock()

    def one(row: Mapping[str, str]) -> None:
        if stop:
            return
        try:
            done = translator(row["title"], row["text"])
        except FatalTranslationError as exc:
            with lock:
                stop.append(str(exc))
            return
        except TranslationError as exc:
            with lock:
                failures[row["article_id"]] = str(exc)
                usage[2] += 1
                if usage[2] >= CONSECUTIVE_FAILURES:
                    stop.append(f"{usage[2]} failures in a row")
            return
        cache.put(row["article_id"], row["title"], row["text"], done, tag)
        with lock:
            usage[2] = 0
            usage[3] += 1
            found[row["article_id"]] = Translation(done.title, done.text)
            usage[0] += done.prompt_tokens
            usage[1] += done.completion_tokens

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return Outcome(
        found,
        failures,
        usage[3],
        usage[0],
        usage[1],
        stop[0] if stop else "",
    )


def with_spanish(
    rows: Sequence[Mapping[str, str]], translations: Mapping[str, Translation]
) -> list[dict[str, str]]:
    """The rows with ``spanish_title`` and ``spanish_text`` right after ``text`` (an untranslated
    article gets empty cells); any earlier Spanish columns are replaced, every other column and its
    order are kept."""
    out: list[dict[str, str]] = []
    for row in rows:
        done = translations.get(row["article_id"])
        new: dict[str, str] = {}
        for key, value in row.items():
            if key in SPANISH_COLUMNS:
                continue
            new[key] = value
            if key == "text":
                new["spanish_title"] = done.title if done else (row.get("spanish_title") or "")
                new["spanish_text"] = done.text if done else (row.get("spanish_text") or "")
        out.append(new)
    return out
