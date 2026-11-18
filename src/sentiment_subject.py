"""Does a text name the company it is about: the one helper of the v1 sentiment stage that the sentiment training scripts still use (T-220).

The v1 stage (``sentiment_stage``) was removed with the v1 chain; this is its ``_text_mentions_subject``, unchanged, so the training scripts and
``tests/news_nlp/test_train_sentiment_pair_script.py`` keep working. No step hashes this file."""

from __future__ import annotations

import re

# Strips common corporate suffixes so "Acme Corp." / "Acme Corporation" both match a sentence naming just "Acme".
_CORP_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|llc|holdings?|group)\.?\b",
    re.IGNORECASE,
)


def _normalize_company_name(name: str) -> str:
    """Strip corporate suffixes and punctuation for a looser company-name comparison."""
    name = _CORP_SUFFIX_RE.sub(" ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    return " ".join(name.split()).strip().lower()


def _text_mentions_subject(text: str, company: str | None, ticker: str | None) -> bool:
    """True if ``text`` names the subject company or its ticker."""
    lowered = text.lower()
    if ticker and re.search(rf"\b{re.escape(ticker.lower())}\b", lowered):
        return True
    if company:
        normalized = _normalize_company_name(company)
        if normalized and normalized in _normalize_company_name(text):
            return True
    return False
