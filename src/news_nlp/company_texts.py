"""Step 4 of the triage-first pipeline (Work item 18): the text a multi-company body says about each
company, for the bodies written as one segment per company (``COMPANY - sentence``, one company
per paragraph).

A segment *leads* with a company when exactly one universe company is named in its first
``LEAD_CHARS`` characters. A segment that leads with none continues the previous company's. A body
is *structured* when at least ``MIN_LED_SHARE`` of its segments lead with a company and at least
two companies are led; any other multi body is left to the exception path (an LLM, D12 as amended
for this step) and ``split_company_texts`` returns ``None`` for it.
"""

from __future__ import annotations

import re

from news_nlp.triage import CompanyResolver

LEAD_CHARS = 50
MIN_SEGMENT = 20  # shorter lines are headers or bylines
MIN_LED_SHARE = 0.6
MIN_COMPANIES = 2


def segments(body: str) -> list[str]:
    """The non-trivial paragraphs of *body*, in order."""
    return [s for s in (p.strip() for p in re.split(r"\n+", body)) if len(s) >= MIN_SEGMENT]


def split_company_texts(body: str, resolver: CompanyResolver) -> dict[str, str] | None:
    """``{ticker: text}`` in order of first appearance, or ``None`` when the body is not structured
    as one segment per company."""
    parts = segments(body)
    groups: dict[str, list[str]] = {}
    current: str | None = None
    led = 0
    for part in parts:
        lead = resolver.resolve(part[:LEAD_CHARS], skip_attribution=False)
        if len(lead) == 1:
            current = next(iter(lead))
            led += 1
        if current is not None:  # text before the first company (an introduction) belongs to none
            groups.setdefault(current, []).append(part)
    if not parts or len(groups) < MIN_COMPANIES or led / len(parts) < MIN_LED_SHARE:
        return None
    return {ticker: "\n".join(texts) for ticker, texts in groups.items()}
