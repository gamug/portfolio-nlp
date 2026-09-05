"""Pure, no-SQL composition logic for the ``sector_summary`` stage: turning
the rows ``news_nlp.sector_summary.queries`` fetches into the deterministic
prose (``compose_sector_summary``), the structured facts payload
(``build_sector_facts``), and the one model-generated sentence
(``build_sector_intro_seed`` -- the *only* text ever handed to a model in
this pipeline stage, which is what makes cross-company / cross-topic
blending structurally impossible here, not a property of model behavior).

None of the functions in this module take a ``conn``/``db`` argument or
contain SQL text -- see ``news_nlp.sector_summary.queries`` for the DB layer
that feeds them.
"""

from __future__ import annotations

import re
import sqlite3

from news_nlp.taxonomy import CATEGORY_LABELS, OTHER_LABEL

# Category display names/ordering for compose_sector_summary and
# build_sector_intro_seed, sourced from the canonical taxonomy (not a second
# hardcoded copy of it) -- taxonomy order, with 'other' forced last since
# it's not itself an NLI candidate label (see taxonomy.py).
_CATEGORY_DISPLAY_NAMES = {slug: display for slug, display, _ in CATEGORY_LABELS} | {
    OTHER_LABEL: "Other"
}
_CATEGORY_ORDER = [slug for slug, _, _ in CATEGORY_LABELS] + [OTHER_LABEL]


def _group_rows_by_category(rows: list[sqlite3.Row]) -> list[tuple[str, list[sqlite3.Row]]]:
    """Group rows by category_label in taxonomy order. Categories with no
    contributing rows are omitted rather than emitted as empty sections."""
    by_label: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by_label.setdefault(r["category_label"], []).append(r)
    return [(slug, by_label[slug]) for slug in _CATEGORY_ORDER if slug in by_label]


def _sentiment_counts(rows: list[sqlite3.Row]) -> dict[str, int]:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in rows:
        counts[r["sentiment_label"]] = counts.get(r["sentiment_label"], 0) + 1
    return counts


def _sentiment_pct(counts: dict[str, int], total: int) -> dict[str, int]:
    if total == 0:
        return dict.fromkeys(counts, 0)
    return {label: round(100 * n / total) for label, n in counts.items()}


def compose_sector_summary(
    gics_sector: str,
    gics_sub_industry: str,
    week_start: str,
    week_end: str,
    intro_text: str,
    rows: list[sqlite3.Row],
    entity_stats: list[dict],
) -> str:
    """Deterministic, non-generative composition of the sector_summary body:
    a header, the model-generated `intro_text` (built purely from aggregate
    stats -- see build_sector_intro_seed, the only text ever handed to a
    model in this pipeline stage), an overview stats block, then one section
    per NLP category present among `rows` (taxonomy order), each listing its
    contributing companies' c_summary text verbatim, attributed to its own
    ticker. No company's text is ever blended with another's, and no text
    ever crosses a category-section boundary -- this is what makes
    cross-company/cross-topic blending structurally impossible here (the
    original "frankenstein" bug's root cause), not a property of model
    behavior."""
    total_articles = len(rows)
    num_companies = len({r["company"] for r in rows})
    sentiment_pct = _sentiment_pct(_sentiment_counts(rows), total_articles)
    entities_line = (
        ", ".join(f"{e['text']} ({e['count']})" for e in entity_stats) if entity_stats else "none"
    )

    lines = [
        f"SECTOR: {gics_sector} / {gics_sub_industry}",
        f"WEEK: {week_start} to {week_end}",
        "",
        intro_text,
        "",
        f"OVERVIEW: {total_articles} article(s) across {num_companies} "
        f"compan{'y' if num_companies == 1 else 'ies'} -- "
        f"{sentiment_pct['positive']}% positive, {sentiment_pct['negative']}% negative, "
        f"{sentiment_pct['neutral']}% neutral sentiment.",
        f"TOP ENTITIES: {entities_line}",
    ]

    for slug, category_rows in _group_rows_by_category(rows):
        lines.append("")
        lines.append(f"{_CATEGORY_DISPLAY_NAMES[slug].upper()} ({len(category_rows)} article(s)):")
        lines.extend(
            f"- {r['ticker']} ({r['company']}): {r['summary_text']}" for r in category_rows
        )

    return "\n".join(lines)


def clean_generated_text(text: str) -> str:
    """Whitespace-normalize a model-generated snippet and drop a trailing
    sentence fragment left when generation got cut off at max_length (a
    partial clause with no closing '.', '!', or '?'). Applied to
    build_sector_intro_seed's model output before it's stored as its own
    `intro_text` column -- cosmetic issues that were easy to miss when that
    sentence only ever appeared inline as one line inside the larger
    composed `summary_text` body are surfaced directly now that the sentence
    is also surfaced standalone."""
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized or normalized[-1] in ".!?":
        return normalized
    cut = max(normalized.rfind(ch) for ch in ".!?")
    return normalized[: cut + 1] if cut != -1 else normalized


def build_sector_facts(
    gics_sector: str,
    gics_sub_industry: str,
    week_start: str,
    week_end: str,
    rows: list[sqlite3.Row],
    entity_stats: list[dict],
) -> dict:
    """Structured, non-narrative counterpart to compose_sector_summary's
    prose: the same aggregate stats (sentiment/category/entity breakdowns)
    plus one attributed record per contributing row, each tagged with its
    own ticker/company -- meant for programmatic consumers (e.g.
    knowledge-graph ingestion) that want grounded facts without parsing
    prose or an intro sentence. Same no-cross-company-blending guarantee as
    compose_sector_summary: every `companies` entry's `summary` is one row's
    own c_summary text, never merged with another row's."""
    total_articles = len(rows)
    num_companies = len({r["company"] for r in rows})
    sentiment_counts = _sentiment_counts(rows)
    sentiment_pct = _sentiment_pct(sentiment_counts, total_articles)

    categories = [
        {
            "label": slug,
            "display_name": _CATEGORY_DISPLAY_NAMES[slug],
            "num_articles": len(category_rows),
            "tickers": sorted({r["ticker"] for r in category_rows}),
        }
        for slug, category_rows in _group_rows_by_category(rows)
    ]

    companies = [
        {
            "article_id": r["article_id"],
            "ticker": r["ticker"],
            "company": r["company"],
            "category": r["category_label"],
            "sentiment": r["sentiment_label"],
            "summary": r["summary_text"],
        }
        for r in rows
    ]

    return {
        "gics_sector": gics_sector,
        "gics_sub_industry": gics_sub_industry,
        "week_start": week_start,
        "week_end": week_end,
        "num_articles": total_articles,
        "num_companies": num_companies,
        "sentiment": {"counts": sentiment_counts, "pct": sentiment_pct},
        "categories": categories,
        "top_entities": entity_stats,
        "companies": companies,
    }


def build_sector_intro_seed(
    gics_sector: str,
    gics_sub_industry: str,
    week_start: str,
    week_end: str,
    rows: list[sqlite3.Row],
) -> str:
    """The *only* text ever handed to the summarization model for the
    sector-level intro sentence: one small templated sentence built purely
    from aggregate numbers derived from `rows`. Deliberately contains no
    ticker, company name, or c_summary substring -- entity mentions are
    deliberately left out too, since NER-extracted entities are frequently
    the company names themselves, which would silently reintroduce the same
    risk this function exists to eliminate. That's what makes cross-company
    blending structurally impossible here, not model behavior (see the
    now-removed build_sector_summary_input, the original source of the
    "frankenstein" bug)."""
    total_articles = len(rows)
    num_companies = len({r["company"] for r in rows})
    sentiment_pct = _sentiment_pct(_sentiment_counts(rows), total_articles)

    category_counts = {
        slug: len(category_rows) for slug, category_rows in _group_rows_by_category(rows)
    }
    top_slugs = sorted(category_counts, key=category_counts.__getitem__, reverse=True)[:2]
    topics = (
        " and ".join(_CATEGORY_DISPLAY_NAMES[slug].lower() for slug in top_slugs) or "general news"
    )

    return (
        f"This week, the {gics_sub_industry} sub-industry within {gics_sector} saw "
        f"{total_articles} article(s) across {num_companies} "
        f"compan{'y' if num_companies == 1 else 'ies'}, primarily about {topics}. "
        f"Sentiment was {sentiment_pct['positive']}% positive, {sentiment_pct['negative']}% "
        f"negative, and {sentiment_pct['neutral']}% neutral."
    )
