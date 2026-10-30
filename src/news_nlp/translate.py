"""Spanish title and text next to every tagging file (Work item 18, T-146b and later gold exports).

The owner reads the English finance jargon with difficulty, so each file to tag carries
``spanish_title`` and ``spanish_text`` right after ``text``. :class:`LLMTranslator` calls the
OpenAI-compatible chat endpoint (``LLM_API_KEY`` / ``LLM_MODEL`` / ``LLM_URL``, DeepSeek in
practice) once per article; :class:`TranslationCache` keeps every result on disk keyed by the hash of
the English source, so a re-run or a later export never pays twice. The translation is a reading aid
for the owner: it never enters the key file or any model input.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import httpx

SPANISH_COLUMNS = ("spanish_title", "spanish_text")
CACHE_NAME = "gold_translations_es.csv"
_CACHE_FIELDS = ("article_id", "source_hash", "spanish_title", "spanish_text")

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
    """The model did not return a translation in the expected format."""


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
                response = client.post(
                    self.url.rstrip("/") + "/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                data = response.json()
                return self._parse(data)
            except (httpx.HTTPError, TranslationError, KeyError, ValueError) as exc:
                last = exc
                time.sleep(self.pause * (attempt + 1))
        raise TranslationError(f"translation failed after {self.retries} tries: {last}")

    @staticmethod
    def _parse(data: Mapping[str, Any]) -> Translation:
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


class TranslationCache:
    """``article_id`` -> translation, valid while the English source is unchanged. Appended to
    disk at every put, so an interrupted run loses nothing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, str]] = {}
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    self._rows[row["article_id"]] = row

    def get(self, article_id: str, title: str, text: str) -> Translation | None:
        row = self._rows.get(article_id)
        if row and row["source_hash"] == source_hash(title, text):
            return Translation(row["spanish_title"], row["spanish_text"])
        return None

    def put(self, article_id: str, title: str, text: str, done: Translation) -> None:
        row = {
            "article_id": article_id,
            "source_hash": source_hash(title, text),
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


def translate_rows(
    rows: Sequence[Mapping[str, str]],
    translator: Callable[[str, str], Translation],
    cache: TranslationCache,
    *,
    workers: int = 8,
    limit: int | None = None,
) -> Outcome:
    """Translations for the rows (by ``article_id``), from the cache where the source is unchanged
    and from *translator* otherwise; at most *limit* new translations. A failed article is listed in
    ``failures`` and left out; it is not retried here and nothing about it is cached."""
    found: dict[str, Translation] = {}
    todo: list[Mapping[str, str]] = []
    for row in rows:
        cached = cache.get(row["article_id"], row["title"], row["text"])
        if cached:
            found[row["article_id"]] = cached
        else:
            todo.append(row)
    if limit is not None:
        todo = todo[:limit]
    failures: dict[str, str] = {}
    usage = [0, 0]
    lock = threading.Lock()

    def one(row: Mapping[str, str]) -> None:
        try:
            done = translator(row["title"], row["text"])
        except TranslationError as exc:
            with lock:
                failures[row["article_id"]] = str(exc)
            return
        cache.put(row["article_id"], row["title"], row["text"], done)
        with lock:
            found[row["article_id"]] = Translation(done.title, done.text)
            usage[0] += done.prompt_tokens
            usage[1] += done.completion_tokens

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return Outcome(found, failures, len(todo) - len(failures), usage[0], usage[1])


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
