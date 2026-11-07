"""What the attribution reads, cut the way the owner reads it (Work item 18, T-148b4, D27).

The first reading of the faithfulness check (2026-11-08) found two causes of company texts mixing
companies that the model cannot fix, because they come before it: the sentence splitter ignores
line breaks and list separators, so a list of one line per company is one "sentence" that every
company it names receives whole; and the footers of a news site (disclosures, credits, "more
from", newsletter calls) are text that is about nobody, with tickers in parentheses that attach it
to companies. This module gives the attribution units that are cut where the text is cut, and a
body without its footer. ``chunking.split_sentences`` is not changed: the sentiment and NER stages
use it as it is.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from typing import NamedTuple, Protocol

from chunking import split_sentences
from news_nlp.triage import CompanyResolver

#: Lines that are a site's own text, not the article's: they end a body (a run of them is dropped
#: from the end) and a credit line is dropped wherever it is.
_FOOTER_LINE = re.compile(
    r"""^\W*(
        questions|comments|talk\s+back|got\s+something\s+to\s+say|send\s+(us\s+)?comments|email\s+us
        |facebook\s+us|follow\s+(us|@)|subscribe|plus,\s+sign\s+up|sign\s+up|like\s+this\s+story
        |for\s+more\s+(insight|information)|members\s+of\s+the\s+media|don.?t\s+miss
        |more\s+from|read\s+more|related(\s+(story|stories|articles?))?\s*:|click\s+here|source\s*:
        |copyright|all\s+rights\s+reserved|disclaimer|disclosures?\s*:|seeking\s+alpha.s\s+disclosure
        |cnbc\.com\s+with\s+wires|reporting\s+by|editing\s+by|additional\s+reporting
        |photo\s*:|image\s*:|the\s+opinions\s+expressed|[_\-*=]{4,}|\u00a9
    )""",
    re.I | re.X,
)
#: "— CNBC's Michael Bloom contributed to this report": a credit, at the end or inside the text.
_CREDIT_LINE = re.compile(
    r"^\W*(by\s+)?(reuters|cnbc|cnbc.s\s+[\w.\- ]+|[\w.\- ]+)\s+contributed\s+to\s+this\s+report",
    re.I,
)
#: A disclosure block runs to the end of the body; a "more from" header is followed by link titles.
_DISCLOSURES = re.compile(r"^\W*disclosures?\s*:", re.I)
_MORE_FROM = re.compile(r"^\W*more\s+from\b", re.I)
MAX_LINK_LINE = 140  # characters: the lines under "more from" are titles; a longer one is the story
#: A link to another story, in the middle of the text or at its end: dropped wherever it is.
_TEASER_LINE = re.compile(r"^\W*read\s+more\b", re.I)
TAIL_SHARE = 0.6  # a block opens in the last 40% of the body, or it is part of the story
MIN_KEPT = 200  # characters: a body that would be left with less keeps its footer

_BULLET = re.compile("^[\\-\u2022\u00b7\u25aa*\u2013\u2014]\\s+")
_PIECE_BREAK = re.compile("[;]\\s+|\\s+[\u2022\u00b7\u25aa|]\\s+")
#: Where a clause can end between the mentions of two companies.
_CLAUSE_END = re.compile("[;]|\\s[\u2014\u2013-]\\s|,\\s|\\s(?:but|while|whereas)\\s")
_CONNECTORS = frozenset({"and", "or", "vs", "versus", "the", "of", "also", "as", "with", "to"})
MIN_GAP_WORDS = 2  # words between two mentions that make them two statements, not a list
#: "Goldman Sachs, Wal-Mart Stores - Berkshire cut its stake": the names before the dash own the line.
_HEADER = re.compile('^(?P<head>[^.!?:;"]{2,80}?)\\s+(?:\u2014|\u2013|-)\\s+(?P<rest>\\S.*)$')
_HEAD_JOIN = frozenset({"and", "of", "the", "&", "for", "de"})
MAX_HEAD_WORDS = 8


def strip_footer(body: str) -> str:
    """*body* without the site's footer: credit lines and "read more" links anywhere, a "more
    from" header with the short link lines under it, a block that opens in the last 40% of the body
    with a disclosure line (dropped to the end), and a run of footer lines at the end. A body that
    would be left with fewer than :data:`MIN_KEPT` characters is returned unchanged."""
    total = max(len(body), 1)
    offset = 0
    kept: list[str] = []
    in_links = False
    for line in body.split("\n"):
        text = line.strip()
        if _DISCLOSURES.match(text) and offset / total >= TAIL_SHARE:
            break
        offset += len(line) + 1
        if _MORE_FROM.match(text):
            in_links = True
            continue
        if in_links and len(text) <= MAX_LINK_LINE:
            continue
        in_links = False
        if _CREDIT_LINE.match(text) or _TEASER_LINE.match(text):
            continue
        kept.append(line)
    while kept and (not kept[-1].strip() or _FOOTER_LINE.match(kept[-1].strip())):
        kept.pop()
    text = "\n".join(kept)
    return text if len(text.strip()) >= MIN_KEPT else body


def _clauses(piece: str, resolver: Mentions) -> list[str]:
    """*piece* cut between the mentions of two different companies where something is said between
    them (a clause end and at least :data:`MIN_GAP_WORDS` words), so "Apple rose 3%, Microsoft fell"
    is two statements and "Amazon and eBay announced" or "Kraft Heinz, Time Warner Cable - these
    stocks" stay whole."""
    spans = resolver.mention_spans(piece)
    cuts: list[int] = []
    for (_, end, ticker), (start, _, after) in pairwise(spans):
        if ticker == after:
            continue
        gap = piece[end:start]
        words = [w for w in re.findall(r"[A-Za-z]{2,}", gap) if w.lower() not in _CONNECTORS]
        ends = [m.end() for m in _CLAUSE_END.finditer(gap)]
        if len(words) >= MIN_GAP_WORDS and ends:
            cuts.append(end + ends[-1])
    bounds = [0, *cuts, len(piece)]
    return [p for a, b in pairwise(bounds) if (p := piece[a:b].strip())]


class Unit(NamedTuple):
    """A piece of the body. *owners* is ``None`` when the attribution decides, and the companies
    that own it (maybe none) when the line says so itself (a header before a dash)."""

    text: str
    owners: tuple[str, ...] | None = None


def header_owners(line: str, resolver: Mentions) -> tuple[str, ...] | None:
    """The companies a "NAMES - text" line belongs to: the universe companies named before the
    dash, when what comes before it is a short list of capitalised names. ``()`` when the names
    are of no company of the universe (the line is about somebody else); ``None`` when the line
    has no such header."""
    match = _HEADER.match(line)
    if not match:
        return None
    head = match["head"].strip()
    words = [w for w in re.split(r"[\s,]+", head) if w]
    if len(words) > MAX_HEAD_WORDS or not all(
        w[:1].isupper() or w[:1].isdigit() or w.lower() in _HEAD_JOIN for w in words
    ):
        return None
    return tuple(sorted({ticker for _, _, ticker in resolver.mention_spans(head)}))


def attribution_pieces(body: str, resolver: Mentions, *, headers: bool = False) -> list[Unit]:
    """The pieces of *body* (footer removed) that are attributed one by one: a line, a bullet, a
    sentence, a part between semicolons, and a sentence that names several companies cut where one
    ends and the next begins. In reading order; nothing is dropped but the footer and the bullets.
    With *headers* a line that opens with a list of names and a dash belongs to those names alone
    (its pieces carry them as ``owners``)."""
    units: list[Unit] = []
    for raw in strip_footer(body).split("\n"):
        line = _BULLET.sub("", raw.strip())
        if not line:
            continue
        owners = header_owners(line, resolver) if headers else None
        for sentence, _, _ in split_sentences(line):
            for piece in _PIECE_BREAK.split(sentence):
                if not piece.strip():
                    continue
                texts = [piece.strip()] if owners is not None else _clauses(piece.strip(), resolver)
                units += [Unit(text, owners) for text in texts]
    return units


def attribution_units(body: str, resolver: Mentions) -> list[str]:
    """The texts of :func:`attribution_pieces` without headers."""
    return [unit.text for unit in attribution_pieces(body, resolver)]


class Mentions(Protocol):
    """What the units need of a resolver: :class:`CompanyResolver` or :class:`GuardedResolver`."""

    def mention_spans(self, text: str) -> list[tuple[int, int, str]]: ...

    def resolve(
        self, text: str, org_spans: Iterable[str] = (), *, skip_attribution: bool = True
    ) -> Counter[str]: ...


class GuardedResolver:
    """A resolver for the attribution that does not take a common word for a company. A company
    that the body names only by the first word of its name ("Bloom", "Union", "Home", "Baker") does
    not count: a short form is a mention only when the full name, or a ticker form, of the same
    company is somewhere in *body*. The first reading of the faithfulness check found such texts
    34% faithful (11 of 32) against 90 to 100% for every other way a company is named. The scope
    rules keep the resolver they were measured with: this is for the attribution only."""

    def __init__(self, resolver: CompanyResolver, body: str) -> None:
        self.resolver = resolver
        self.confirmed = {t for _, _, t in resolver.mention_spans(body, aliases=False)}

    def mention_spans(self, text: str) -> list[tuple[int, int, str]]:
        full = {(a, b) for a, b, _ in self.resolver.mention_spans(text, aliases=False)}
        return [
            (a, b, t)
            for a, b, t in self.resolver.mention_spans(text)
            if (a, b) in full or t in self.confirmed
        ]

    def resolve(
        self, text: str, org_spans: Iterable[str] = (), *, skip_attribution: bool = True
    ) -> Counter[str]:
        return Counter(t for _, _, t in self.mention_spans(text))
