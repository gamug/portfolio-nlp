"""Step 4 of the triage-first pipeline (Work item 18): the text a multi-company body says about each
company, for the bodies written as one segment per company (``COMPANY - sentence``, one company
per paragraph).

A segment *leads* with a company when exactly one universe company is named in its first
``LEAD_CHARS`` characters. A segment that leads with none *continues* the previous company's. A
body is *structured* when it has at least ``MIN_SEGMENTS`` segments, at least ``MIN_LED_SHARE`` of
them lead with a company and at least ``MIN_COMPANIES`` companies are led; any other multi body is
left to the trained attribution model (D27) and ``label_segments`` returns ``None`` for it.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from news_nlp.triage import CompanyResolver

LEAD_CHARS = 50
MIN_SEGMENT = 20  # shorter lines are headers or bylines
MIN_SEGMENTS = 4
MIN_LED_SHARE = 0.5
MIN_COMPANIES = 3


class SegmentLabel(NamedTuple):
    position: int  # position in ``segments(body)``
    ticker: str  # canonical ticker of the company the segment is attributed to
    kind: str  # "lead" (names the company at its start) or "continuation" (inherited)


def segments(body: str) -> list[str]:
    """The non-trivial paragraphs of *body*, in order."""
    return [s for s in (p.strip() for p in re.split(r"\n+", body)) if len(s) >= MIN_SEGMENT]


def segment_text(body: str, index: int) -> str:
    """The text of segment *index* of *body*: how a labelled segment is rebuilt from SOURCE, so the
    label files hold positions and tickers and no news text."""
    return segments(body)[index]


def label_segments(body: str, resolver: CompanyResolver) -> list[SegmentLabel] | None:
    """The company of each segment of a structured body, or ``None`` when the body is not written
    as one segment per company. Text before the first company (an introduction) has no label."""
    parts = segments(body)
    labels: list[SegmentLabel] = []
    current: str | None = None
    led = 0
    for position, part in enumerate(parts):
        lead = resolver.resolve(part[:LEAD_CHARS], skip_attribution=False)
        if len(lead) == 1:
            current = next(iter(lead))
            led += 1
            kind = "lead"
        else:
            kind = "continuation"
        if current is not None:
            labels.append(SegmentLabel(position, current, kind))
    companies = {label.ticker for label in labels}
    if (
        len(parts) < MIN_SEGMENTS
        or len(companies) < MIN_COMPANIES
        or led / len(parts) < MIN_LED_SHARE
    ):
        return None
    return labels


def split_company_texts(body: str, resolver: CompanyResolver) -> dict[str, str] | None:
    """``{ticker: text}`` in order of first appearance, or ``None`` when the body is not structured
    as one segment per company."""
    labels = label_segments(body, resolver)
    if labels is None:
        return None
    parts = segments(body)
    groups: dict[str, list[str]] = {}
    for label in labels:
        groups.setdefault(label.ticker, []).append(parts[label.position])
    return {ticker: "\n".join(texts) for ticker, texts in groups.items()}
