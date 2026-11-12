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
        questions\s*(?:[?:,]|$)|questions\s+for\s+\w+\s*\??\s*(?:\S+@\S+)?$|comments\s*(?:[?:,]|$)
        |disclaimer\s*(?:[?:]|$)
        |talk\s+back|got\s+something\s+to\s+say|send\s+(?:us\s+)?comments|email\s+us
        |facebook\s+us|follow\s+(?:us|@)|subscribe(?:\s+(?:to|now|here|for|today)\b|\s*$)
        |plus,\s+sign\s+up|sign\s+up\b|like\s+this\s+story
        |for\s+more\s+(?:insight|information)|members\s+of\s+the\s+media|don.?t\s+miss
        |more\s+from|read\s+more|related(?:\s+(?:story|stories|articles?))?\s*:|click\s+here
        |source\s*:|copyright\s*(?:\u00a9|\(c\)|\d{4}|:)|all\s+rights\s+reserved|disclosures?\s*:
        |seeking\s+alpha.s\s+disclosure|cnbc\.com\s+with\s+wires|reporting\s+by|editing\s+by
        |additional\s+reporting|photo\s*:|image\s*:|the\s+opinions\s+expressed|[_\-*=]{4,}|\u00a9
    )""",
    re.I | re.X,
)
MAX_FOOTER_LINE = 300  # characters: a longer line is a paragraph of the story...
#: ...unless it is a legal notice, which is long by nature.
_LEGAL_LINE = re.compile(
    r"^\W*(?:disclosures?\s*:|disclaimer|seeking\s+alpha.s\s+disclosure"
    r"|copyright\s*(?:\u00a9|\(c\)|\d{4}|:)|all\s+rights)",
    re.I,
)
#: Credits and labels that are one short line: a long paragraph that starts alike is the story
#: ("Reporting by Bloomberg showed that deliveries fell...").
_CREDITISH = re.compile(
    r"^\W*(?:reporting\s+by|editing\s+by|additional\s+reporting|source\s*:|sign\s+up\b"
    r"|plus,\s+sign\s+up|copyright)",
    re.I,
)
MAX_CREDIT_LINE = 120  # characters
#: "— CNBC's Michael Bloom contributed to this report": a credit, at the end or inside the text.
_CREDIT_LINE = re.compile(
    r"^\W*(by\s+)?(reuters|cnbc|cnbc.s\s+[\w.\- ]+|[\w.\- ]+)\s+contributed\s+to\s+this\s+report",
    re.I,
)
#: A disclosure block runs to the end of the body; a "more from" header is followed by link titles.
_DISCLOSURES = re.compile(r"^\W*disclosures?\s*:", re.I)
_MORE_FROM = re.compile(r"^\W*more\s+from\b", re.I)
MAX_LINK_LINE = 140  # characters: the lines under "more from" are titles; a longer one is the story
#: A title does not end like a sentence: a short line that does (a one-line quote) is the story.
_SENTENCE_END = re.compile("[.!][\"\u201d')]?$")
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


def _is_footer(text: str) -> bool:
    """Whether a line is the site's own text (short, and starting like a footer label)."""
    if not _FOOTER_LINE.match(text):
        return False
    if _LEGAL_LINE.match(text):
        return True
    limit = MAX_CREDIT_LINE if _CREDITISH.match(text) else MAX_FOOTER_LINE
    return len(text) <= limit


def strip_footer(body: str) -> str:
    """*body* without the site's footer: credit lines and "read more" links anywhere, a "more
    from" header with the short link lines under it (a short line that ends like a sentence is the
    story), a block that opens in the last 40% of the body
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
        if in_links and len(text) <= MAX_LINK_LINE and not _SENTENCE_END.search(text):
            continue
        in_links = False
        if _CREDIT_LINE.match(text) or _TEASER_LINE.match(text):
            continue
        kept.append(line)
    while kept and (not kept[-1].strip() or _is_footer(kept[-1].strip())):
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


def starts_a_sentence(text: str) -> bool:
    """Whether *text* starts like the text of a line (a capital, a digit, a quote or a bracket) and
    not like the middle of a sentence ("Miami Beach - as in New York ...")."""
    first = text.lstrip()[:1]
    if not first:  # empty or blank: it starts nothing ("" is "in" every string)
        return False
    return first.isupper() or first.isdigit() or first in "\"'\u201c\u2018(["


#: "Wendy's (WEN)": a ticker in brackets after a name, as the news roundups write it.
_TICKER_BRACKET = re.compile(r"\([A-Z][A-Z.\-]{0,5}\)")


#: ``units5`` (T-177): a head may hold a period inside a name ("Amazon.com", "Box Inc.", "Fox Corp.").
_ABBREVIATIONS = re.compile(r"\b(Inc|Corp|Co|Ltd|Cos|Bros|Plc|Cl)\.")
_DOT = "\u2024"  # a stand-in for the period of an abbreviation while the header is matched
_HEADER5 = re.compile(
    '^(?P<head>(?:[^.!?:;"]|(?<=\\w)\\.(?=\\w)|' + _DOT + "){2,80}?)"
    "\\s+(?:\u2014|\u2013|-)\\s+(?P<rest>\\S.*)$"
)
#: The last word of a group label ("Memory stocks", "Cruise lines"): the line is about a group, not a company.
_GROUP_NOUNS = frozenset(
    (  # noqa: SIM905 -- a word list, easier to read as text
        "stocks names shares companies operators lines plays makers producers carriers airlines banks lenders "
        "retailers builders miners developers providers manufacturers firms groups equities funds brokers "
        "restaurants chains sector infrastructure"
    ).split()
)
_HEAD_JOIN5 = _HEAD_JOIN | {"in", "on"}  # "Jack in the Box"
MAX_GROUP_WORDS = 4
MAX_CAPS_LETTERS = 4  # "SAP", "BP", "UBS", "HSBC": a company; "DETROIT" is a dateline
MAX_CAPS_WORDS = 2  # "AT & T"
#: Short datelines, agencies, wire labels and terms in capitals that are not a company ("FDA - The agency approved Apple's
#: device": the sentence would go to nobody and Apple would lose it). A ticker in brackets still makes any of them a company.
_CAPS_NOT_COMPANIES = frozenset(
    (  # noqa: SIM905 -- a word list, easier to read as text
        "ROME OSLO LIMA BAKU DOHA KYIV KIEV "
        "FDA SEC FTC DOJ EPA FCC FBI CIA IRS NASA OPEC NATO IMF ECB BOE BOJ PBOC FED WHO WTO OECD "
        "US UK EU UN UAE USA "
        "NEWS LIVE TOP WATCH BREAKING ALERT UPDATE QUOTE BUZZ "
        "CEO CFO COO IPO ETF GDP CPI PMI AI EPS FY"
    ).split()
)


def _is_company_in_capitals(words: list[str], head: str) -> bool:
    """``units5``: a head in capitals that is one short word ("SAP", "UBS"), "AT & T", or carries a ticker in brackets; not an agency, a wire label or a place."""
    if _TICKER_BRACKET.search(head):
        return True
    letters = [w for w in words if w != "&"]
    return (
        not any(w in _CAPS_NOT_COMPANIES for w in letters)
        and 1 <= len(letters) <= MAX_CAPS_WORDS
        and all(w.isalpha() and len(w) <= MAX_CAPS_LETTERS for w in letters)
        and (len(letters) == 1 or "&" in words)
    )


def _is_group_label(words: list[str]) -> bool:
    """``units5``: "Memory stocks", "Cruise lines", "China electric vehicle stocks": a few plain words ending in a group noun."""
    return (
        1 < len(words) <= MAX_GROUP_WORDS
        and words[0][:1].isupper()
        and all(w.isalpha() or w == "&" for w in words)
        and words[-1].lower() in _GROUP_NOUNS
    )


def header_owners(
    line: str, resolver: Mentions, *, tickers_in_head: bool = False, units5: bool = False
) -> tuple[str, ...] | None:
    """The companies a "NAMES - text" line belongs to: the universe companies named before the
    dash, when what comes before it is a short list of capitalised names. ``()`` when the names
    are of no company of the universe (the line is about somebody else); ``None`` when the line
    has no such header, when what comes before the dash is written in capitals and names no company
    of the universe (a dateline such as "DETROIT" or a label such as "REFINING"), or when what
    follows the dash does not start like a sentence (a dash inside a sentence). With
    *tickers_in_head* a ticker in brackets after a name ("Wendy's (WEN) - ...") is part of the
    head, as the roundups write it; the other readings do not take it for one (T-176). With *units5*
    (T-177) the head may hold a period in a name ("Amazon.com", "Box Inc."), be a group label
    ("Memory stocks", "Cruise lines": a line of nobody, whose members' sentences go to them), be a
    short name in capitals ("SAP", "UBS", or any head with a ticker in brackets) and not a dateline, and the text after the
    dash may start in lowercase when the head has its ticker in brackets."""
    if units5:
        line = _ABBREVIATIONS.sub(lambda m: m.group(1) + _DOT, line)
    match = (_HEADER5 if units5 else _HEADER).match(line)
    if not match:
        return None
    head = match["head"].strip().replace(_DOT, ".")
    shown = _TICKER_BRACKET.sub(" ", head) if tickers_in_head else head
    words = [w for w in re.split(r"[\s,]+", shown) if w]
    sentence_start = starts_a_sentence(match["rest"].replace(_DOT, "."))
    join = _HEAD_JOIN5 if units5 else _HEAD_JOIN
    group = units5 and _is_group_label(words)
    shaped = group or all(w[:1].isupper() or w[:1].isdigit() or w.lower() in join for w in words)
    owners = tuple(sorted({ticker for _, _, ticker in resolver.mention_spans(head)}))
    in_capitals = all(
        w.isupper() or not any(c.isalpha() for c in w) or w.lower() in join for w in words
    )
    # a dash inside a sentence ("Apple - synonymous with the classic phone - has spawned..."): units5 reads a
    # lowercase text only after a head with its ticker in brackets, as the roundups write it ("Whirlpool (WHR) - ...");
    # a company's name alone is not enough (3,000 bodies: 9 of 14 such lines were prose dashes)
    if (
        not shaped
        or len(words) > MAX_HEAD_WORDS
        or not (sentence_start or (units5 and _TICKER_BRACKET.search(head)))
    ):
        return None
    if not owners and in_capitals:  # "DETROIT - Ford ...", "REFINING - ...": a dateline or a label
        return () if units5 and _is_company_in_capitals(words, head) else None
    return owners


def attribution_pieces(
    body: str,
    resolver: Mentions,
    *,
    headers: bool = False,
    named_after: bool = False,
    per_sentence: bool = False,
    units5: bool = False,
) -> list[Unit]:
    """The pieces of *body* (footer removed) that are attributed one by one: a line, a bullet, a
    sentence, a part between semicolons, and a sentence that names several companies cut where one
    ends and the next begins. In reading order; nothing is dropped but the footer and the bullets.
    With *headers* a line that opens with a list of names and a dash belongs to those names alone
    (its pieces carry them as ``owners``). With *named_after* a piece of a line headed by universe
    companies also belongs to the universe companies it names itself (a roundup line that lists
    several, T-148b6); a line headed by somebody else still belongs to nobody. With *per_sentence*
    (with *headers*) a sentence in the middle of a line that opens with a header of its own ("A - x.
    B - y." written as one paragraph) starts a new owner, and the sentences after it belong to that
    one (T-176): a header is not read once for a whole line. A headed sentence is not cut into
    clauses, so a header stays with its text. With *units5* (T-177) the headers are read as
    :func:`header_owners` does with the same flag, and a sentence that has no header of its own
    after a header of nobody belongs to the universe companies it names (or to nobody if it names
    none): "Airlines - The group fell. United slipped 2%." gives the second sentence to United."""
    units: list[Unit] = []
    for raw in strip_footer(body).split("\n"):
        line = _BULLET.sub("", raw.strip())
        if not line:
            continue
        owners = (
            header_owners(line, resolver, tickers_in_head=per_sentence, units5=units5)
            if headers
            else None
        )
        for sentence, _, _ in split_sentences(line):
            headed = owners is not None
            if headers and per_sentence:
                own = header_owners(sentence.strip(), resolver, tickers_in_head=True, units5=units5)
                headed = own is not None
                if own is not None:
                    owners = own  # "B - text" in the middle of a line: B owns it and what follows
            # units5: no header of its own after a header of nobody -> the companies the sentence names
            carried = units5 and owners == () and not headed
            for piece in _PIECE_BREAK.split(sentence):
                if not piece.strip():
                    continue
                if owners is None:
                    units += [Unit(text) for text in _clauses(piece.strip(), resolver)]
                    continue
                named = {t for _, _, t in resolver.mention_spans(piece)} if named_after else set()
                if carried:
                    units.append(Unit(piece.strip(), tuple(sorted(named))))
                    continue
                units.append(
                    Unit(piece.strip(), tuple(sorted(set(owners) | named) if owners else ()))
                )
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
    that the body names only by the first word of its name ("Bloom", "Union", "Home", "Baker"), or
    by a name that is also a surname or a noun ("Williams", "Block", "Travelers"), does not count:
    such a word is a mention only when the full name, the short form or a ticker form of the same
    company is somewhere in *body* (``CompanyResolver.firm_spans``). The first reading of the
    faithfulness check found such texts 34% faithful (11 of 32) against 90 to 100% for every other
    way a company is named. The scope rules count a body with ``CompanyResolver.resolve``, which
    applies the same rule to the whole body (PLAN D38); this is for the single sentences the
    attribution masks and cuts."""

    def __init__(self, resolver: CompanyResolver, body: str) -> None:
        self.resolver = resolver
        self.confirmed = {t for _, _, t in resolver.firm_spans(body)}

    def mention_spans(self, text: str) -> list[tuple[int, int, str]]:
        full = {(a, b) for a, b, _ in self.resolver.firm_spans(text)}
        return [
            (a, b, t)
            for a, b, t in self.resolver.mention_spans(text)
            if (a, b) in full or t in self.confirmed
        ]

    def resolve(
        self, text: str, org_spans: Iterable[str] = (), *, skip_attribution: bool = True
    ) -> Counter[str]:
        return Counter(t for _, _, t in self.mention_spans(text))
