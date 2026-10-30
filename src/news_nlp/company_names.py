"""Company names, nicknames and tickers -> the universe's canonical tickers (Work item 18, T-146b).

The owner tags ``companies`` / ``mentions`` / ``company_labels`` of the gold with the names as the
article writes them; :class:`NameBook` turns them into tickers. It never guesses: a text that
matches no company, or more than one, is reported with its candidates and left as typed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from news_nlp.triage import _WORD_TICKERS, Company, normalize_name

#: Well-known names other than the legal one, by ticker. A ticker absent from the universe is
#: skipped, so the list may name companies that left the index.
NICKNAMES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple Computer", "Apple Inc"),
    "ABBV": ("Abbvie",),
    "ABT": ("Abbott",),
    "ADBE": ("Adobe Systems",),
    "AMD": ("AMD",),
    "AMZN": ("Amazon.com",),
    "AVGO": ("Broadcom Inc",),
    "AXP": ("Amex",),
    "BA": ("Boeing Company",),
    "BAC": ("BofA", "BoA", "Bank of America Merrill Lynch"),
    "BBY": ("Best Buy Co",),
    "BDX": ("BD", "Becton Dickinson and Company"),
    "BF.B": ("Brown Forman",),
    "BKNG": ("Booking.com", "Booking"),
    "BLK": ("Black Rock",),
    "BMY": ("Bristol-Myers Squibb", "Bristol Myers"),
    "BNY": ("Bank of New York Mellon", "Bank of New York", "BNY Mellon"),
    "BRK.B": ("Berkshire", "Berkshire Hathaway Inc", "Warren Buffett's Berkshire"),
    "BX": ("Blackstone Group",),
    "C": ("Citi", "Citigroup Inc", "Citibank"),
    "CAT": ("Caterpillar",),
    "CHTR": ("Charter",),
    "CI": ("Cigna Group",),
    "CL": ("Colgate",),
    "CMCSA": ("Comcast Corp",),
    "COF": ("Capital One Financial",),
    "COP": ("Conoco", "Conoco Phillips"),
    "COST": ("Costco Wholesale",),
    "CRM": ("Salesforce.com", "Salesforce Inc"),
    "CSCO": ("Cisco Systems",),
    "CVS": ("CVS Pharmacy", "CVS Caremark"),
    "CVX": ("Chevron",),
    "DAL": ("Delta Airlines",),
    "DE": ("Deere", "John Deere"),
    "DIS": ("Disney", "Walt Disney", "Walt Disney Co"),
    "DOW": ("Dow Chemical", "Dow"),
    "DUK": ("Duke",),
    "EBAY": ("eBay",),
    "EL": ("Estee Lauder", "Estée Lauder"),
    "F": ("Ford", "Ford Motor"),
    "FDX": ("Federal Express", "Fedex"),
    "FIS": ("Fidelity National Information Services",),
    "FISV": ("Fiserv Inc", "Fiserv"),
    "GD": ("General Dynamics",),
    "GE": ("General Electric", "GE"),
    "GILD": ("Gilead",),
    "GIS": ("General Mills",),
    "GM": ("General Motors", "GM"),
    "GOOGL": ("Google", "Google Inc"),
    "GS": ("Goldman", "Goldman Sachs Group", "Goldman Sachs & Co"),
    "HD": ("Home Depot", "The Home Depot"),
    "HON": ("Honeywell", "Honeywell International"),
    "HPQ": ("HP", "Hewlett-Packard", "Hewlett Packard"),
    "IBM": ("International Business Machines", "Big Blue"),
    "INTC": ("Intel Corp", "Intel Corporation"),
    "JNJ": ("J&J", "Johnson and Johnson", "Johnson & Johnson", "Johnson&Johnson"),
    "JPM": ("JP Morgan", "J.P. Morgan", "JPMorgan", "J P Morgan", "JPMorgan Chase & Co"),
    "KHC": ("Kraft", "Heinz", "Kraft-Heinz"),
    "KO": ("Coke", "Coca Cola", "Coca-Cola", "The Coca-Cola Company"),
    "LLY": ("Eli Lilly", "Lilly", "Eli Lilly and Company"),
    "LMT": ("Lockheed",),
    "LOW": ("Lowe's", "Lowes"),
    "LUV": ("Southwest",),
    "LYV": ("Live Nation",),
    "MA": ("Master Card",),
    "MCD": ("McDonalds", "McDonald's Corp"),
    "META": ("Facebook", "Facebook Inc", "FB"),
    "MMM": ("3M Company", "3M Co", "Minnesota Mining"),
    "MO": ("Altria Group", "Philip Morris USA"),
    "MRK": ("Merck", "Merck & Co", "Merck and Co"),
    "MS": ("Morgan Stanley", "Morgan Stanley & Co"),
    "MSFT": ("Microsoft Corp", "Microsoft Corporation"),
    "NFLX": ("Netflix Inc",),
    "NKE": ("Nike Inc",),
    "NVDA": ("Nvidia Corp", "NVIDIA", "Nvidia Corporation"),
    "ORCL": ("Oracle",),
    "PEP": ("Pepsi", "Pepsico Inc"),
    "PFE": ("Pfizer Inc",),
    "PG": ("P&G", "Procter and Gamble", "Procter & Gamble"),
    "PM": ("Philip Morris", "PMI"),
    "PYPL": ("PayPal Holdings", "Paypal"),
    "QCOM": ("Qualcomm Inc",),
    "RTX": ("Raytheon", "Raytheon Technologies", "United Technologies"),
    "SBUX": ("Starbucks Corp",),
    "SCHW": ("Schwab", "Charles Schwab"),
    "SLB": ("Schlumberger Limited",),
    "SPGI": ("S&P Global", "Standard & Poor's"),
    "T": ("ATT", "AT&T Inc", "AT and T"),
    "TGT": ("Target",),
    "TMO": ("Thermo Fisher",),
    "TMUS": ("T-Mobile", "TMobile", "T Mobile"),
    "TSLA": ("Tesla Motors", "Tesla Inc"),
    "TXN": ("TI", "Texas Instruments Inc"),
    "UAL": ("United Airlines",),
    "UNH": ("UnitedHealth", "United Health", "UnitedHealthcare"),
    "UNP": ("Union Pacific",),
    "UPS": ("United Parcel Service", "United Parcel"),
    "USB": ("US Bancorp", "U.S. Bank", "US Bank"),
    "V": ("Visa",),
    "VZ": ("Verizon Communications", "Verizon Wireless"),
    "WBD": ("Warner Bros", "Warner Brothers"),
    "WFC": ("Wells Fargo & Co", "Wells Fargo Bank"),
    "WMT": ("Wal-Mart", "Walmart Inc", "Wal Mart"),
    "XOM": ("Exxon", "Exxon Mobil", "Exxon Mobil Corporation"),
    "XYZ": ("Block Inc",),
    "YUM": ("Yum Brands",),
}

#: Brands, subsidiaries and products of a company: a name that stands for the parent is a labelling
#: decision, so the owner confirms each use (:class:`NameBook` reports it as ``brand``).
BRANDS: dict[str, tuple[str, ...]] = {
    "AMZN": ("Amazon Web Services", "AWS", "Whole Foods"),
    "BAC": ("Merrill Lynch", "Merrill"),
    "BF.B": ("Jack Daniel's",),
    "BKNG": ("Priceline",),
    "CHTR": ("Spectrum",),
    "CMCSA": ("NBCUniversal", "Xfinity"),
    "CVS": ("Aetna",),
    "DIS": ("ESPN", "Pixar"),
    "FOXA": ("Fox News",),
    "GM": ("Chevrolet", "Cadillac"),
    "GOOGL": ("YouTube", "Waymo"),
    "JPM": ("Chase",),
    "LYV": ("Ticketmaster",),
    "META": ("Instagram", "WhatsApp"),
    "MSFT": ("LinkedIn", "Activision"),
    "PEP": ("Frito-Lay",),
    "TMUS": ("Sprint",),
    "UNH": ("Optum",),
    "WBD": ("HBO",),
    "WMT": ("Sam's Club",),
    "XYZ": ("Square", "Cash App"),
    "YUM": ("KFC", "Taco Bell", "Pizza Hut"),
}

#: Companies the triage keeps as two tickers (class B / class A): a name stands for class A, an
#: explicit ticker for itself.
CLASS_A = {"FOX": "FOXA", "NWS": "NWSA"}
_MIN_FIRST_WORD = 3
_SHORT_TICKER = 2  # one- and two-letter tickers (A, C, F, T, V, IT, ON) are also everyday letters
_FOLD = str.maketrans({"\u2019": "'", "\u00e9": "e", "\u2013": "-"})


def _key(text: str) -> str:
    """The comparison form of a name: ``normalize_name`` plus quotes, accents and dashes folded."""
    folded = text.lower().translate(_FOLD)
    return normalize_name(re.sub(r"'s\b", "s", folded).replace("'", "")).strip()


class Match(NamedTuple):
    """The outcome of one lookup: a ticker, or why there is none; ``note`` says how an inferred
    match was made (``first word``, ``brand``, ``word ticker``, ``ticker, other case``) and is
    empty when the text is a ticker, a legal name or a nickname."""

    ticker: str | None
    why: str | None
    note: str = ""


def _is_word_ticker(ticker: str) -> bool:
    return len(ticker) <= _SHORT_TICKER or ticker in _WORD_TICKERS


@dataclass
class NameBook:
    """``lookup(text)`` -> :class:`Match`; ``resolve(text)`` -> (ticker, why-not).

    A text is looked up as (1) an upper-case ticker, (2) a name, a nickname, a brand or the first
    word of one name, (3) a ticker in any case. Share classes of one company resolve to one
    ticker (the universe's ``by_ticker`` map). Matches that are inferred, not exact, carry a
    ``note`` for the owner to confirm: a first word (``Home``), a brand (``Instagram``), a ticker
    that is also a word or has one or two letters (``ALL``, ``IT``), a ticker typed in lower case."""

    companies: Sequence[Company]
    canonical: dict[str, str]  # ticker -> canonical ticker of its share-class family
    _names: dict[str, set[str]] = field(init=False, default_factory=lambda: defaultdict(set))
    _inferred: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._names = defaultdict(set)
        self._inferred = {}
        for company in self.companies:
            if key := _key(company.name):
                self._names[key].add(self._name_ticker(company.ticker))
        firsts: dict[str, set[str]] = defaultdict(set)
        for key, tickers in list(self._names.items()):
            words = key.split()
            if len(words) > 1 and len(words[0]) >= _MIN_FIRST_WORD:
                firsts[words[0]] |= tickers
        for word, tickers in firsts.items():
            if word not in self._names:  # a full name beats a first word ("FedEx", "Williams")
                self._names[word] = tickers
                self._inferred[word] = "first word"
        self._add(NICKNAMES, "")
        self._add(BRANDS, "brand")

    def _add(self, table: dict[str, tuple[str, ...]], note: str) -> None:
        for ticker, names in table.items():
            if ticker.upper() not in self.canonical:
                continue
            for key in filter(None, map(_key, names)):
                self._names[key].add(self._name_ticker(ticker))
                if note and key not in self._inferred:
                    self._inferred[key] = note
                elif not note:
                    self._inferred.pop(key, None)

    def _name_ticker(self, ticker: str) -> str:
        canon = self.canonical.get(ticker.upper(), ticker.upper())
        return CLASS_A.get(canon, canon)

    def lookup(self, text: str) -> Match:
        raw = text.strip().strip("$")
        if not raw:
            return Match(None, "empty")
        if raw.isupper() and raw in self.canonical:
            return Match(self.canonical[raw], None, "word ticker" if _is_word_ticker(raw) else "")
        key = _key(raw)
        found = self._names.get(key, set())
        if len(found) == 1:
            return Match(next(iter(found)), None, self._inferred.get(key, ""))
        if found:
            return Match(None, "ambiguous: " + ", ".join(sorted(found)))
        if raw.upper() in self.canonical:
            return Match(self.canonical[raw.upper()], None, "ticker, other case")
        return Match(None, "unknown")

    def resolve(self, text: str) -> tuple[str | None, str | None]:
        match = self.lookup(text)
        return match.ticker, match.why


def _split_list(value: str) -> list[str]:
    return [p.strip() for p in value.replace("\n", ";").split(";") if p.strip()]


def normalise_row(
    book: NameBook, row: dict[str, str]
) -> tuple[dict[str, str], list[str], list[str]]:
    """The row with ``companies``, ``mentions`` and the company part of ``company_labels`` as
    tickers, the problems found (the text of each is left as typed) and the inferred matches the
    owner should confirm. A company listed twice (``Apple; AAPL``) becomes one entry: ``check``
    on the original would have reported it."""
    out = dict(row)
    problems: list[str] = []
    confirm: list[str] = []

    def lookup(column: str, text: str) -> str | None:
        match = book.lookup(text)
        if match.ticker is None:
            problems.append(f"{column}: {text!r} {match.why}")
        elif match.note:
            confirm.append(f"{column}: {text!r} -> {match.ticker} ({match.note})")
        return match.ticker

    def tickers(column: str) -> None:
        resolved: list[str] = []
        kept: list[str] = []
        for part in _split_list(row.get(column) or ""):
            ticker = lookup(column, part)
            if ticker is None:
                kept.append(part)
            elif ticker not in resolved:
                resolved.append(ticker)
        out[column] = "; ".join([*resolved, *kept])

    tickers("companies")
    tickers("mentions")
    out["company_labels"] = _normalise_labels(
        row.get("company_labels") or "", lambda name: lookup("company_labels", name), problems
    )
    return out, problems, confirm


def _normalise_labels(value: str, lookup: Callable[[str], str | None], problems: list[str]) -> str:
    """``Apple: other; Google: uncertain`` -> ``AAPL:other; GOOG:uncertain``; what cannot be
    resolved stays as typed."""
    labels: dict[str, str] = {}
    kept: list[str] = []
    for part in _split_list(value):
        name, sep, label = part.rpartition(":")
        label = label.strip()
        ticker = lookup(name.strip()) if sep else None
        if not sep:
            problems.append(f"company_labels: {part!r} has no label")
        if ticker is None:
            kept.append(part)
        elif labels.get(ticker, label) != label:
            problems.append(f"company_labels: {ticker} given {labels[ticker]!r} and {label!r}")
        else:
            labels[ticker] = label
    return "; ".join([*(f"{t}:{lb}" for t, lb in labels.items()), *kept])
