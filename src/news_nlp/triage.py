"""Rules the triage-first pipeline starts from (PLAN.md Work item 18, P1; TASKS T-144).

Pure and CPU-only: no model, no database. Given the S&P 500 universe (ticker, company, GICS sector
and sub-industry, as the SOURCE ``articles`` rows carry them), it

* hashes an article body (:func:`body_hash`), the unit of Work item 18 (D11): the crawler stores
  one row per (URL, ticker) pair, so the same body appears once per tagged ticker;
* finds which universe companies a body names (:class:`CompanyResolver`): the company's name
  (legal suffixes dropped), an explicit ticker form (``$AAPL``, ``(NASDAQ: AAPL)``) or a bare
  upper-case ticker of three letters or more that is not an English word, plus the NER ``ORG``
  spans already stored for the body;
* proposes a scope (:func:`scope_candidate`): one company, several, or none, the last split by
  keyword counts into market, economy, sector or unknown;
* guesses the GICS sub-industry a body is about from the companies it names (:func:`gics_guess`).

These are candidates for sizing and for checking against human labels, not the triage itself:
names are matched without the upstream alias table (T-132), so companies known by another name
("Google" for Alphabet) are missed unless the NER span or a ticker form catches them.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

#: Legal-form words dropped from a company name before matching it.
_LEGAL_WORDS = frozenset(
    [
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "companies",
        "ltd",
        "limited",
        "plc",
        "llc",
        "lp",
        "holdings",
        "holding",
        "group",
        "sa",
        "nv",
        "ag",
        "se",
        "the",
        "class",
    ]
)
#: Tickers that are also English words or common abbreviations: never matched bare.
_WORD_TICKERS = frozenset(
    [
        "A",
        "ALL",
        "ARE",
        "BALL",
        "BEN",
        "BRO",
        "CAT",
        "COST",
        "CPU",
        "DAY",
        "DD",
        "DOW",
        "ED",
        "EL",
        "FAST",
        "GD",
        "GE",
        "GL",
        "HAS",
        "HD",
        "HOLX",
        "HUM",
        "IT",
        "J",
        "K",
        "KEY",
        "KIM",
        "L",
        "LIN",
        "LOW",
        "MA",
        "MAR",
        "META",
        "MO",
        "MS",
        "NOW",
        "O",
        "ON",
        "PEAK",
        "PH",
        "PM",
        "PSA",
        "RL",
        "SO",
        "SPG",
        "T",
        "TECH",
        "TEL",
        "V",
        "WELL",
        "WM",
        "ZTS",
        "ALLE",
        "CARR",
        "FOX",
        "FOXA",
        "NI",
    ]
)
#: Companies whose name mostly appears as something else (an exchange or index, a data credit, a
#: news brand): matched only by an explicit ticker form.
NAME_UNSAFE_TICKERS = frozenset({"NDAQ", "DOW", "FDS", "NWS", "NWSA", "FOX", "FOXA"})
#: Company names that, once legal words are dropped, are everyday English words: matched only
#: capitalised ("Target", not "price target").
_WORD_NAMES = frozenset(
    [
        "target",
        "southern",
        "progressive",
        "ball",
        "news",
        "fox",
        "match",
        "waters",
        "pool",
        "mosaic",
    ]
)
_EXCHANGES = r"(?:NYSE|NASDAQ|Nasdaq|NYSEARCA|NYSE American|AMEX|BATS|Cboe)"
#: Keywords of a body that names no company (whole words, case-insensitive).
MARKET_TERMS = (
    "s&p 500",
    "dow jones",
    "nasdaq composite",
    "wall street",
    "stock market",
    "stocks",
    "index",
    "indexes",
    "indices",
    "futures",
    "rally",
    "sell-off",
    "selloff",
    "bull market",
    "bear market",
)
ECONOMY_TERMS = (
    "federal reserve",
    "fed",
    "inflation",
    "interest rates",
    "rate cut",
    "rate hike",
    "gdp",
    "recession",
    "unemployment",
    "jobs report",
    "payrolls",
    "consumer price index",
    "cpi",
    "treasury yields",
    "economy",
    "economic",
    "tariffs",
)
#: How many keyword hits make a no-company body a market / economy / sector candidate.
MIN_KEYWORD_HITS = 2
#: Share of the subject mentions above which one company dominates a multi-company body (a strict
#: majority). Chosen on the owner's 720 tagged articles (a development set): 0.5 agrees with the
#: owner's scope on 0.81 of them, 0.6 on 0.75, 0.7 on 0.71 (2026-10-29).
DOMINANT_SHARE = 0.5
#: Share of a body's companies that must share a sub-industry for it to be guessed.
GICS_MAJORITY = 0.5
MIN_COMPANIES_FOR_GICS = 2


def body_hash(text: str) -> str:
    """SHA-256 of the body with whitespace collapsed, so re-fetched copies of one article that
    differ only in spacing share a hash."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def normalize_name(name: str) -> str:
    """Lower case, share class, punctuation and legal-form words dropped:
    ``"Alphabet Inc. (Class A)"`` -> ``"alphabet"``.
    Returns ``""`` if nothing is left."""
    lowered = re.sub(r"\bclass\s+[a-z]\b", " ", name.lower().replace(".com", ""))
    words = re.sub(r"[^a-z0-9& ]+", " ", lowered).split()
    return " ".join(w for w in words if w not in _LEGAL_WORDS)


class Company(NamedTuple):
    ticker: str
    name: str
    gics_sector: str
    gics_sub_industry: str


def _words_pattern(phrases: Iterable[str]) -> re.Pattern[str]:
    alternatives = sorted({re.escape(p) for p in phrases if p}, key=len, reverse=True)
    return re.compile(r"(?<![\w&])(?:" + "|".join(alternatives) + r")(?![\w&])", re.IGNORECASE)


#: A company named as the source of an opinion, not as a subject: "Goldman Sachs analysts",
#: "analysts at Morgan Stanley".
_ATTRIBUTION_AFTER = re.compile(
    r"(?:'s)?\s+(?:analysts?|strategists?|economists?|equity research)\b", re.IGNORECASE
)
_ATTRIBUTION_BEFORE = re.compile(
    r"\b(?:analysts?|strategists?|economists?)\s+(?:at|from|with)\s+$", re.IGNORECASE
)
#: Sentences that are site furniture or data credits, not content.
_BOILERPLATE = re.compile(
    r"[^.\n]*(?:compiled by FactSet|according to FactSet|FactSet (?:data|estimates)|data (?:from|by|provided by) "
    r"FactSet|Got a confidential news tip|Sign up for|Subscribe to|Click here|Disclosure:|Related Tickers)"
    r"[^.\n]*[.\n]?",
    re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """The text without data credits and site furniture (each matching sentence removed)."""
    return _BOILERPLATE.sub(" ", text)


#: Headlines (or URL slugs, hyphens read as spaces) of multi-company round-ups: market movers,
#: watch lists, stock-pick lists.
ROUNDUP_TITLE = re.compile(
    r"biggest moves|early movers|\bmovers\b|stocks making|after[- ]hours buzz|premarket|pre-market|"
    r"midday|stocks to watch|things to (?:know|watch)|what to watch|week ahead|market wrap|"
    r"morning brief|top stocks|stocks? (?:that|to) buy|these stocks|stock picks",
    re.IGNORECASE,
)
#: A body is a round-up if its title names at least this many companies, or its text carries at
#: least :data:`ROUNDUP_TICKER_FORMS` explicit ticker forms.
ROUNDUP_TITLE_COMPANIES = 3
ROUNDUP_TICKER_FORMS = 5


@dataclass
class CompanyResolver:
    """Finds the universe companies a body names. Share classes of one company (``GOOG`` /
    ``GOOGL``) resolve to the first ticker seen for that normalized name."""

    companies: Sequence[Company]
    _by_name: dict[str, str] = field(init=False)
    _by_ticker: dict[str, str] = field(init=False)
    _names: re.Pattern[str] = field(init=False)
    _word_names: re.Pattern[str] = field(init=False)
    _tickers: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        self._by_name = {}
        self._by_ticker = {}
        for company in self.companies:
            key = normalize_name(company.name)
            if key and company.ticker.upper() not in NAME_UNSAFE_TICKERS:
                canonical = self._by_name.setdefault(key, company.ticker)
            else:
                canonical = company.ticker
            self._by_ticker[company.ticker.upper()] = canonical
        self._names = _words_pattern(k for k in self._by_name if k not in _WORD_NAMES)
        capitalised = sorted(k.capitalize() for k in self._by_name if k in _WORD_NAMES)
        self._word_names = re.compile(
            r"(?<![\w&])(?:"
            + "|".join([re.escape(c) for c in capitalised] or ["(?!)"])
            + r")(?![\w&])"
        )
        tickers = sorted(self._by_ticker, key=len, reverse=True)
        explicit = "|".join(re.escape(t) for t in tickers)
        bare = "|".join(re.escape(t) for t in tickers if len(t) >= 3 and t not in _WORD_TICKERS)  # noqa: PLR2004
        self._tickers = re.compile(
            rf"\$({explicit})\b|\(\s*{_EXCHANGES}\s*:\s*({explicit})\s*\)|(?<![\w$])({bare})(?![\w])"
        )

    @property
    def by_ticker(self) -> dict[str, str]:
        return self._by_ticker

    def ticker_forms(self, text: str) -> int:
        """How many explicit ticker forms (``$X``, ``(NYSE: X)``) *text* carries."""
        return sum(1 for m in self._tickers.finditer(text) if m.group(1) or m.group(2))

    def resolve(
        self, text: str, org_spans: Iterable[str] = (), *, skip_attribution: bool = True
    ) -> Counter[str]:
        """Mentions per canonical ticker in *text* and in the stored NER ``ORG`` spans of the body
        (a span counts once if it is a universe name). A name used as an opinion's source
        ("Goldman Sachs analysts") is not counted."""
        found: Counter[str] = Counter()
        for pattern in (self._names, self._word_names):
            for match in pattern.finditer(text):
                if not (skip_attribution and _attributed(text, match)):
                    found[self._by_name[normalize_name(match.group(0))]] += 1
        for match in self._tickers.finditer(text):
            ticker = next(g for g in match.groups() if g)
            found[self._by_ticker[ticker.upper()]] += 1
        for span in org_spans:
            key = normalize_name(span)
            if key in self._by_name:
                found[self._by_name[key]] += 1
        return found


def _attributed(text: str, match: re.Match[str]) -> bool:
    return bool(
        _ATTRIBUTION_AFTER.match(text, match.end())
        or _ATTRIBUTION_BEFORE.search(text[max(0, match.start() - 30) : match.start()])
    )


def is_roundup(title: str, slug: str, title_companies: int, ticker_forms: int) -> bool:
    """A multi-company round-up: a round-up headline or URL slug, a title naming at least
    :data:`ROUNDUP_TITLE_COMPANIES` companies, or at least :data:`ROUNDUP_TICKER_FORMS` explicit
    ticker forms in the text."""
    return bool(
        ROUNDUP_TITLE.search(title)
        or ROUNDUP_TITLE.search(slug.replace("-", " "))
        or title_companies >= ROUNDUP_TITLE_COMPANIES
        or ticker_forms >= ROUNDUP_TICKER_FORMS
    )


class Scope(NamedTuple):
    scope: (
        str  # single | multi | multi_dominated | roundup | market | economy | sector | no_company
    )
    companies: int
    top_share: float  # mentions of the most-named company / all company mentions (0 if none)


def keyword_hits(text: str, terms: Sequence[str]) -> int:
    return len(_words_pattern(terms).findall(text))


#: Mentions that make a company a subject of the body rather than an incidental name.
MIN_SUBJECT_MENTIONS = 2


def subjects(mentions: Counter[str], title_mentions: Iterable[str] = ()) -> Counter[str]:
    """The companies a body is about: named at least :data:`MIN_SUBJECT_MENTIONS` times, or in
    the title. One passing mention (an analyst's bank, a comparison) is not a subject."""
    in_title = set(title_mentions)
    return Counter(
        {t: n for t, n in mentions.items() if n >= MIN_SUBJECT_MENTIONS or t in in_title}
    )


def scope_candidate(
    mentions: Counter[str],
    text: str,
    sector_terms: Sequence[str] = (),
    *,
    roundup: bool = False,
) -> Scope:
    """The rule-based scope of a body from its *mentions* (pass :func:`subjects` to ignore passing
    names). A *roundup* (:func:`is_roundup`) is one whatever its counts. With no company, the largest of the market, economy and sector keyword counts decides
    if it reaches :data:`MIN_KEYWORD_HITS`; ties go to economy, then market."""
    total = sum(mentions.values())
    if roundup:  # see is_roundup: the structure decides before the counts
        return Scope("roundup", len(mentions), max(mentions.values()) / total if total else 0.0)
    if not total:
        hits = {
            "economy": keyword_hits(text, ECONOMY_TERMS),
            "market": keyword_hits(text, MARKET_TERMS),
            "sector": keyword_hits(text, sector_terms) if sector_terms else 0,
        }
        best = max(hits, key=lambda k: (hits[k], k == "economy", k == "market"))
        return Scope(best if hits[best] >= MIN_KEYWORD_HITS else "no_company", 0, 0.0)
    top = max(mentions.values()) / total
    if len(mentions) == 1:
        return Scope("single", 1, 1.0)
    return Scope("multi_dominated" if top > DOMINANT_SHARE else "multi", len(mentions), top)


def gics_guess(mentions: Counter[str], universe: dict[str, Company]) -> tuple[str, str] | None:
    """``(sector, sub_industry)`` shared by at least :data:`GICS_MAJORITY` of the companies a body
    names (at least :data:`MIN_COMPANIES_FOR_GICS` of them), else ``None``."""
    if len(mentions) < MIN_COMPANIES_FOR_GICS:
        return None
    counts = Counter(
        (universe[t].gics_sector, universe[t].gics_sub_industry)
        for t in mentions
        if t in universe and universe[t].gics_sub_industry
    )
    if not counts:
        return None
    (pair, n) = counts.most_common(1)[0]
    return pair if n / len(mentions) >= GICS_MAJORITY else None


# --- checking the rules against human labels (T-144) ---------------------------------------------

#: Scopes the owner's 720 gold articles can be mapped to.
GOLD_SCOPES = ("single", "multi", "sector", "market_economy", "unrelated")
_UNRELATED_LABELS = frozenset({"unrelated", "non-related", "not-related"})
_MULTI_NOTE = re.compile(
    r"multiple companies|many companies|other company|two companies|two stocks|comparison",
    re.IGNORECASE,
)
_SECTOR_NOTE = re.compile(r"sector|industry", re.IGNORECASE)
_MARKET_NOTE = re.compile(
    r"econom|s&p ?500|market (?:trent|trend|behavior)|marked behavior", re.IGNORECASE
)


def gold_scope(label: str, note: str) -> str:
    """The scope an owner's tag and note imply: ``unrelated`` tags, then economy / whole-market
    notes, then sector notes, then multi-company tags (``miscellany``) or notes, else single."""
    label = label.strip().lower()
    if label in _UNRELATED_LABELS:
        return "unrelated"
    if _MARKET_NOTE.search(note):
        return "market_economy"
    if _SECTOR_NOTE.search(note):
        return "sector"
    if label == "miscellany" or _MULTI_NOTE.search(note):
        return "multi"
    return "single"


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval of a proportion; ``(nan, nan)`` for n = 0."""
    if not n:
        return float("nan"), float("nan")
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return centre - half, centre + half
