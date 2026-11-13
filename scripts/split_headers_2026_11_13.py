"""Splits a multi-company body written as ``Company — text`` paragraphs into one segment per header.

Why not ``news_nlp.company_texts.split_company_texts``: it opens a new segment only where a *universe* company leads, so a
header naming a company outside the universe ("Fossil -", "Pep Boys —") is read as a continuation and its text lands in the
previous company's row. Here every header opens a segment (inside the universe or not); a segment whose header is not exactly
one universe company is dropped, never given to the company before it.
"""

from __future__ import annotations

import re

from news_nlp.triage import CompanyResolver

DASH = r"(?:[—–]|\s-\s|\s--\s)"
HEADER = re.compile(
    rf"^\s*(?P<name>[A-Z0-9][^\n—–]{{1,70}}?)\s*(?P<tick>\([^)]{{1,30}}\))?\s*{DASH}\s*(?=\S)"
)
_NAME = r"[A-Z0-9][\w&.'’-]*(?:(?:\s?,\s?|\s)[\w&.'’-]+){0,6}"
INLINE = re.compile(
    rf"(?<=[.!?\"”’])\s+(?={_NAME}\s?[—–]\s?[A-Z0-9$])"
)  # a capital after the dash: "X — a key metric" is not a header
MIN_SEGMENT = 20
MIN_HEADERS = 3
MIN_HEADER_SHARE = 0.6
MAX_NAME_WORDS = 7


def paragraphs(body: str) -> list[str]:
    body = INLINE.sub("\n", body)
    return [
        p
        for p in (s.strip() for s in re.split(r"\n+", body))
        if len(p) >= MIN_SEGMENT and not p.startswith(("—", "–"))
    ]


def header_name(paragraph: str) -> str | None:
    m = HEADER.match(paragraph)
    if not m or len(m.group("name").split()) > MAX_NAME_WORDS:
        return None
    return (m.group("name") + " " + (m.group("tick") or "")).strip()


def split(body: str, resolver: CompanyResolver) -> tuple[dict[str, str], int] | None:
    """``({ticker: segment text}, headers in the body)`` or ``None`` when the body is not written as header paragraphs.
    The text of a segment is its header paragraph and the paragraphs after it up to the next header."""
    paras = paragraphs(body)
    names = [header_name(p) for p in paras]
    heads = sum(n is not None for n in names)
    if heads < MIN_HEADERS or heads / max(len(paras), 1) < MIN_HEADER_SHARE:
        return None
    out: dict[str, str] = {}
    current: str | None = None
    for p, n in zip(paras, names, strict=False):
        if n is not None:
            hit = resolver.resolve(n, skip_attribution=False)
            current = next(iter(hit)) if len(hit) == 1 else None
            if current is not None and current not in out:
                out[current] = p
            else:
                current = None if current in out else current
        elif current is not None:
            out[current] += "\n" + p
    return out, heads


# ruff: noqa: RUF001  -- one-shot research script
