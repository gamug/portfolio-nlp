"""The mention-and-recency baseline of step 4 (Work item 18, T-148b2, D27): the comparison the
trained attribution model has to beat on the owner's faithfulness check.

A sentence naming one or more universe companies belongs to every company it names; a sentence
naming none belongs to the companies of the sentence before it (recency); the sentences before the
first mention belong to nobody. No model is involved.
"""

from __future__ import annotations

from chunking import split_sentences
from news_nlp.triage import CompanyResolver


def attribute_sentences(body: str, resolver: CompanyResolver) -> list[tuple[str, tuple[str, ...]]]:
    """``(sentence, tickers)`` for each sentence of *body*, ``()`` for the ones before any mention."""
    current: tuple[str, ...] = ()
    out: list[tuple[str, tuple[str, ...]]] = []
    for sentence, _, _ in split_sentences(body):
        named = tuple(sorted(resolver.resolve(sentence, skip_attribution=False)))
        if named:
            current = named
        out.append((sentence, current))
    return out


def baseline_company_texts(body: str, resolver: CompanyResolver) -> dict[str, str]:
    """``{ticker: text}`` in order of first appearance; a company named in the body always gets the
    sentence that names it, so no mentioned company comes back empty."""
    groups: dict[str, list[str]] = {}
    for sentence, tickers in attribute_sentences(body, resolver):
        for ticker in tickers:
            groups.setdefault(ticker, []).append(sentence)
    return {ticker: " ".join(sentences) for ticker, sentences in groups.items()}
