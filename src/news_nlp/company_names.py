"""Company names, nicknames and tickers -> the universe's canonical tickers (Work item 18, T-146b).

The owner tags ``companies`` / ``mentions`` / ``company_labels`` of the gold with the names as the
article writes them; :class:`NameBook` turns them into tickers. It never guesses: a text that
matches no company, or more than one, is reported with its candidates and left as typed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from news_nlp.triage import Company, normalize_name

#: Well-known names other than the legal one, by ticker. A ticker absent from the universe is
#: skipped, so the list may name companies that left the index.
NICKNAMES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple Computer", "Apple Inc"),
    "ABBV": ("Abbvie",),
    "ABT": ("Abbott",),
    "ADBE": ("Adobe Systems",),
    "AMD": ("AMD",),
    "AMZN": ("Amazon.com", "Amazon Web Services", "AWS", "Whole Foods"),
    "AVGO": ("Broadcom Inc",),
    "AXP": ("Amex",),
    "BA": ("Boeing Company",),
    "BAC": ("BofA", "BoA", "Bank of America Merrill Lynch", "Merrill Lynch"),
    "BBY": ("Best Buy Co",),
    "BDX": ("BD", "Becton Dickinson and Company"),
    "BF.B": ("Brown Forman", "Jack Daniel's"),
    "BK": ("Bank of New York Mellon",),
    "BKNG": ("Booking.com", "Priceline", "Booking"),
    "BLK": ("Black Rock",),
    "BMY": ("Bristol-Myers Squibb", "Bristol Myers"),
    "BNY": ("Bank of New York Mellon", "Bank of New York", "BNY Mellon"),
    "BRK.B": ("Berkshire", "Berkshire Hathaway Inc", "Warren Buffett's Berkshire"),
    "BX": ("Blackstone Group",),
    "C": ("Citi", "Citigroup Inc", "Citibank"),
    "CAT": ("Caterpillar",),
    "CHTR": ("Charter", "Spectrum"),
    "CI": ("Cigna Group",),
    "CL": ("Colgate",),
    "CMCSA": ("Comcast Corp", "NBCUniversal", "Xfinity"),
    "COF": ("Capital One Financial",),
    "COP": ("Conoco", "Conoco Phillips"),
    "COST": ("Costco Wholesale",),
    "CRM": ("Salesforce.com", "Salesforce Inc"),
    "CSCO": ("Cisco Systems",),
    "CVS": ("CVS Pharmacy", "CVS Caremark", "Aetna"),
    "CVX": ("Chevron",),
    "DAL": ("Delta Airlines",),
    "DE": ("Deere", "John Deere"),
    "DIS": ("Disney", "Walt Disney", "Walt Disney Co", "ESPN", "Pixar"),
    "DOW": ("Dow Chemical", "Dow"),
    "DUK": ("Duke",),
    "EBAY": ("eBay",),
    "EL": ("Estee Lauder", "Estée Lauder"),
    "F": ("Ford", "Ford Motor"),
    "FDX": ("Federal Express", "Fedex"),
    "FIS": ("Fidelity National Information Services",),
    "FISV": ("Fiserv Inc", "Fiserv"),
    "FOXA": ("Fox News",),
    "GD": ("General Dynamics",),
    "GE": ("General Electric", "GE"),
    "GILD": ("Gilead",),
    "GIS": ("General Mills",),
    "GM": ("General Motors", "GM", "Chevrolet", "Cadillac"),
    "GOOGL": ("Google", "YouTube", "Waymo", "Google Inc"),
    "GS": ("Goldman", "Goldman Sachs Group", "Goldman Sachs & Co"),
    "HD": ("Home Depot", "The Home Depot"),
    "HON": ("Honeywell", "Honeywell International"),
    "HPQ": ("HP", "Hewlett-Packard", "Hewlett Packard"),
    "IBM": ("International Business Machines", "Big Blue"),
    "INTC": ("Intel Corp", "Intel Corporation"),
    "JNJ": ("J&J", "Johnson and Johnson", "Johnson & Johnson", "Johnson&Johnson"),
    "JPM": ("JP Morgan", "J.P. Morgan", "JPMorgan", "J P Morgan", "Chase", "JPMorgan Chase & Co"),
    "KHC": ("Kraft", "Heinz", "Kraft-Heinz"),
    "KO": ("Coke", "Coca Cola", "Coca-Cola", "The Coca-Cola Company"),
    "LLY": ("Eli Lilly", "Lilly", "Eli Lilly and Company"),
    "LMT": ("Lockheed",),
    "LOW": ("Lowe's", "Lowes"),
    "LUV": ("Southwest",),
    "LYV": ("Live Nation", "Ticketmaster"),
    "MA": ("Master Card",),
    "MCD": ("McDonalds", "McDonald's Corp"),
    "META": ("Facebook", "Facebook Inc", "Instagram", "WhatsApp", "FB"),
    "MMM": ("3M Company", "3M Co", "Minnesota Mining"),
    "MO": ("Altria Group", "Philip Morris USA"),
    "MRK": ("Merck", "Merck & Co", "Merck and Co"),
    "MS": ("Morgan Stanley", "Morgan Stanley & Co"),
    "MSFT": ("Microsoft Corp", "Microsoft Corporation", "LinkedIn", "Activision"),
    "NFLX": ("Netflix Inc",),
    "NKE": ("Nike Inc",),
    "NVDA": ("Nvidia Corp", "NVIDIA", "Nvidia Corporation"),
    "ORCL": ("Oracle",),
    "PEP": ("Pepsi", "Pepsico Inc", "Frito-Lay"),
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
    "TMUS": ("T-Mobile", "TMobile", "T Mobile", "Sprint"),
    "TSLA": ("Tesla Motors", "Tesla Inc"),
    "TXN": ("TI", "Texas Instruments Inc"),
    "UAL": ("United Airlines",),
    "UNH": ("UnitedHealth", "United Health", "UnitedHealthcare", "Optum"),
    "UNP": ("Union Pacific",),
    "UPS": ("United Parcel Service", "United Parcel"),
    "USB": ("US Bancorp", "U.S. Bank", "US Bank"),
    "V": ("Visa",),
    "VZ": ("Verizon Communications", "Verizon Wireless"),
    "WBD": ("Warner Bros", "Warner Brothers", "HBO"),
    "WFC": ("Wells Fargo & Co", "Wells Fargo Bank"),
    "WMT": ("Wal-Mart", "Walmart Inc", "Wal Mart", "Sam's Club"),
    "XOM": ("Exxon", "Exxon Mobil", "Exxon Mobil Corporation"),
    "XYZ": ("Square", "Block Inc", "Cash App"),
    "YUM": ("Yum Brands", "KFC", "Taco Bell", "Pizza Hut"),
}

#: Companies the triage keeps as two tickers (class B / class A): a name stands for class A, an
#: explicit ticker for itself.
CLASS_A = {"FOX": "FOXA", "NWS": "NWSA"}
_MIN_FIRST_WORD = 3
_FOLD = str.maketrans({"\u2019": "'", "\u00e9": "e", "\u2013": "-"})


def _key(text: str) -> str:
    """The comparison form of a name: ``normalize_name`` plus quotes, accents and dashes folded."""
    folded = text.lower().translate(_FOLD)
    return normalize_name(re.sub(r"'s\b", "s", folded).replace("'", "")).strip()


@dataclass
class NameBook:
    """``resolve(text)`` -> (canonical ticker, ``None``) or (``None``, why not).

    A text is looked up as (1) an upper-case ticker, (2) a name, a nickname or the first word of
    one name, (3) a ticker in any case. Share classes of one company resolve to one ticker (the
    universe's ``by_ticker`` map)."""

    companies: Sequence[Company]
    canonical: dict[str, str]  # ticker -> canonical ticker of its share-class family
    _names: dict[str, set[str]] = field(init=False, default_factory=lambda: defaultdict(set))

    def __post_init__(self) -> None:
        self._names = defaultdict(set)
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
        for ticker, nicknames in NICKNAMES.items():
            if ticker.upper() in self.canonical:
                for key in filter(None, map(_key, nicknames)):
                    self._names[key].add(self._name_ticker(ticker))

    def _name_ticker(self, ticker: str) -> str:
        canon = self.canonical.get(ticker.upper(), ticker.upper())
        return CLASS_A.get(canon, canon)

    def resolve(self, text: str) -> tuple[str | None, str | None]:
        raw = text.strip().strip("$")
        if not raw:
            return None, "empty"
        if raw.isupper() and raw in self.canonical:
            return self.canonical[raw], None
        found = self._names.get(_key(raw), set())
        if len(found) == 1:
            return next(iter(found)), None
        if found:
            return None, "ambiguous: " + ", ".join(sorted(found))
        if raw.upper() in self.canonical:
            return self.canonical[raw.upper()], None
        return None, "unknown"


def _split_list(value: str) -> list[str]:
    return [p.strip() for p in value.replace("\n", ";").split(";") if p.strip()]


def normalise_row(book: NameBook, row: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """The row with ``companies``, ``mentions`` and the company part of ``company_labels`` as
    tickers, and the problems found (the text of each is left as typed)."""
    out = dict(row)
    problems: list[str] = []

    def tickers(column: str) -> list[str]:
        resolved: list[str] = []
        kept: list[str] = []
        for part in _split_list(row.get(column) or ""):
            ticker, why = book.resolve(part)
            if ticker is None:
                problems.append(f"{column}: {part!r} {why}")
                kept.append(part)
            elif ticker not in resolved:
                resolved.append(ticker)
        out[column] = "; ".join([*resolved, *kept])
        return resolved

    tickers("companies")
    tickers("mentions")
    labels: dict[str, str] = {}
    kept_labels: list[str] = []
    for part in _split_list(row.get("company_labels") or ""):
        name, sep, label = part.rpartition(":")
        if not sep:
            problems.append(f"company_labels: {part!r} has no label")
            kept_labels.append(part)
            continue
        ticker, why = book.resolve(name)
        if ticker is None:
            problems.append(f"company_labels: {name.strip()!r} {why}")
            kept_labels.append(part)
        elif labels.get(ticker, label.strip()) != label.strip():
            problems.append(
                f"company_labels: {ticker} given {labels[ticker]!r} and {label.strip()!r}"
            )
        else:
            labels[ticker] = label.strip()
    out["company_labels"] = "; ".join([*(f"{t}:{lb}" for t, lb in labels.items()), *kept_labels])
    return out, problems
