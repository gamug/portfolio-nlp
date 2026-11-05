"""The owner's re-tag of the 720 articles under the new taxonomy: gold v2 (Work item 18, T-146b).

The first tagging ("tags v1") has a label and a note per article. Gold v2 has, per body: the
**scope** (one of :data:`~news_nlp.triage.GOLD_SCOPES`), the **company_labels** (one ``TICKER:label``
per analysed company, separated by commas: single: one; multi: two or more; the analysed companies
are exactly the labelled ones, so there is no separate companies column), the **mentions** (companies named but not
analysed, D23) and, for a sector body, the GICS **sub-industry**. :func:`retag_rows` builds
the file the owner fills in (the v1 tag shown as guidance, never the LLM's label), and
:func:`check_row` / :func:`load_gold_v2` validate it when it comes back. The rules follow
``docs/labelling-spec-v2.md`` (T-146a): D15's labels in :data:`D15_LABELS`, the per-scope shape in
:func:`_scope_errors`; a test keeps the spec and this module in step. The checker expects tickers;
names the owner typed are converted first by :mod:`news_nlp.company_names` (``normalise``).
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from news_nlp.company_names import split_list
from news_nlp.gics_codes import SUB_INDUSTRY_CODES
from news_nlp.taxonomy import CATEGORY_LABELS, CATEGORY_SLUGS
from news_nlp.triage import GOLD_SCOPES, Company, CompanyResolver, gold_scope

#: D15: the 9 current labels, the 4 new ones, and ``other`` / ``uncertain`` / ``unrelated``.
D15_LABELS = (
    *CATEGORY_SLUGS,
    "reputational_harm",
    "competitive_actions",
    "investor_activity",
    "peer_comparison",
    "other",
    "uncertain",
    "unrelated",
)
TAG_COLUMNS = ("scope", "company_labels", "mentions", "sub_industry", "notes")
#: Scopes that analyse no company: no company or label is expected.
NO_COMPANY_SCOPES = ("market_economy", "investing_general", "unrelated", "sector")
#: ``unrelated`` is a body-level scope, not a label of one company; ``uncertain`` is allowed.
COMPANY_LABELS = tuple(label for label in D15_LABELS if label != "unrelated")
LEGACY_COLUMN = "companies"  # the first format; its content is now company_labels
REVIEW_NAME = "category_gold_v2_review.csv"
KEY_NAME = "category_gold_v2_key.csv"
CARRIED = (
    "review_order",
    "article_id",
    "ticker",
    "company",
    "pub_date",
    "source_domain",
    "title",
    "text",
)


@dataclass(frozen=True)
class Universe:
    """The SOURCE universe: ticker -> canonical ticker (share classes of one company resolve to
    one, like :class:`~news_nlp.triage.CompanyResolver`) and the GICS sub-industry names."""

    tickers: dict[str, str]
    sub_industries: dict[str, str]  # lower case -> source spelling

    def resolve_sub_industry(self, text: str) -> tuple[str | None, str | None]:
        """The source spelling of a sub-industry given as a name (case and ``&`` / ``and`` do not
        matter) or as its 8-digit GICS code, or ``(None, why not)``. A code is looked up in
        :data:`~news_nlp.gics_codes.SUB_INDUSTRY_CODES`; one whose name no company of the universe
        has is refused, not guessed."""
        raw = text.strip()
        by_key = {_sub_key(name): name for name in self.sub_industries.values()}
        if raw.isdigit():
            name = SUB_INDUSTRY_CODES.get(raw)
            if name is None:
                return None, f"{raw!r} is not a GICS sub-industry code"
            if _sub_key(name) not in by_key:
                return None, f"code {raw} is {name!r}, which no company of the universe has"
            return by_key[_sub_key(name)], None
        known = self.sub_industries.get(raw.lower()) or by_key.get(_sub_key(raw))
        if known is None:
            return None, f"{raw!r} is not a GICS sub-industry (name or code) of the universe"
        return known, None

    @classmethod
    def of(cls, companies: Sequence[Company]) -> Universe:
        return cls(
            CompanyResolver(companies).by_ticker,
            {
                c.gics_sub_industry.lower(): c.gics_sub_industry
                for c in companies
                if c.gics_sub_industry
            },
        )


def retag_rows(
    v1_review: Sequence[Mapping[str, str]], v1_key: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The v2 review and key rows from the v1 review and key rows (same articles, same order). The
    review rows show the v1 label and note and the scope the notes imply (``v1_scope_derived``: a
    regex mapping, the one behind the rules' agreement figure, not the owner's own decision), with
    EMPTY tag columns; the LLM's label stays in the key only."""
    if [r["article_id"] for r in v1_review] != [k["article_id"] for k in v1_key]:
        raise ValueError("the v1 review and key files list different articles or another order")
    review: list[dict[str, object]] = []
    for row in v1_review:
        v1_label, v1_notes = row["human_label"], row["human_notes"]
        review.append(
            {
                **{k: row[k] for k in CARRIED},
                "v1_label": v1_label,
                "v1_notes": v1_notes,
                "v1_scope_derived": gold_scope(v1_label, v1_notes),
                **dict.fromkeys(TAG_COLUMNS, ""),
            }
        )
    return review, [dict(k) for k in v1_key]


def _sub_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower().replace("&", "and")).strip()


_split = split_list


@dataclass(frozen=True)
class Gold:
    """One validated gold-v2 article."""

    article_id: int
    scope: str
    companies: tuple[str, ...]
    labels: dict[str, str]  # company -> D15 label
    mentions: tuple[str, ...]
    sub_industry: str
    outside: tuple[
        str, ...
    ] = ()  # companies (analysed or mentioned) outside the universe, as typed


_MAX_TICKER = 5  # the longest ticker of the universe is five characters (BF.B has the dot)


def _ticker_like(text: str) -> bool:
    """Short and upper case, no space: written as a ticker, so a typo of one, not a name."""
    return text.isupper() and " " not in text and len(text) <= _MAX_TICKER


def _company(
    raw: str, universe: Universe | None, errors: list[str], outside: list[str]
) -> str | None:
    """The canonical ticker of *raw*; with a universe, a name that is no ticker of it is a company
    outside the universe (**[owner call 9]**) and is kept as typed, listed in *outside* so that the
    report shows it (a typo would look the same). ``None`` after an error."""
    if universe is None:
        return raw.upper()
    ticker = raw.upper()
    if ticker in universe.tickers:
        return universe.tickers[ticker]
    if _ticker_like(raw):
        errors.append(f"{ticker}: not a ticker of the universe")
        return None
    if ";" in raw or ":" in raw:
        errors.append(f"{raw!r}: looks like several entries; separate them with commas")
        return None
    outside.append(raw)
    return raw


def _tickers(
    value: str, universe: Universe | None, errors: list[str], outside: list[str]
) -> tuple[str, ...]:
    out: list[str] = []
    for raw in _split(value):
        company = _company(raw, universe, errors, outside)
        if company is None:
            continue
        if company.casefold() in (c.casefold() for c in out):
            errors.append(f"{company}: listed twice")
        out.append(company)
    return tuple(out)


def _parse_labels(
    value: str, universe: Universe | None, errors: list[str], outside: list[str]
) -> dict[str, str]:
    """``AAPL:product_innovation, MSFT:other`` -> {ticker: label}; the keys are the analysed
    companies (a name outside the universe is kept as typed)."""
    labels: dict[str, str] = {}
    for item in _split(value):
        company, sep, label = item.rpartition(":")
        company, label = company.strip(), label.strip().lower()
        if not sep:
            errors.append(f"{item!r}: no label (write TICKER:label)")
            continue
        found = _company(company, universe, errors, outside)
        if found is None:
            continue
        company = found
        if company.casefold() in (c.casefold() for c in labels):
            errors.append(f"{company}: listed twice")
        if label not in COMPANY_LABELS:
            errors.append(f"{company}: label {label!r} is not a company label (D15)")
        labels[company] = label
    return labels


def _scope_errors(
    scope: str, companies: tuple[str, ...], labels: dict[str, str], sub_industry: str
) -> list[str]:
    if scope in NO_COMPANY_SCOPES:
        errors = [f"scope {scope} analyses no company"] if companies or labels else []
        if scope == "sector" and not sub_industry:
            errors.append("a sector body needs a sub_industry")
        return errors
    needed = 2 if scope == "multi" else 1
    errors = []
    if len(companies) < needed or (scope == "single" and len(companies) != 1):
        errors.append(
            "scope single needs exactly 1 labelled company (the others are mentions)"
            if scope == "single"
            else "scope multi needs 2 or more labelled companies"
        )
    return errors


def check_row(
    row: Mapping[str, str], universe: Universe | None = None
) -> tuple[Gold | None, list[str]]:
    """Validate one returned row; ``(None, errors)`` when it is not usable. With a *universe*,
    tickers must belong to it (share classes are canonicalised) and ``sub_industry`` must be one
    of its GICS sub-industries."""
    scope = (row.get("scope") or "").strip().lower()
    if scope not in GOLD_SCOPES:
        return None, [f"scope {scope!r} is not one of {', '.join(GOLD_SCOPES)}"]
    errors: list[str] = []
    outside: list[str] = []
    labels = _parse_labels(row.get("company_labels") or "", universe, errors, outside)
    companies = tuple(labels)
    mentions = _tickers(row.get("mentions") or "", universe, errors, outside)
    if {c.casefold() for c in companies} & {m.casefold() for m in mentions}:
        errors.append("a company is both analysed and a mention")
    sub_industry = (row.get("sub_industry") or "").strip()
    if universe is not None and sub_industry:
        known, why = universe.resolve_sub_industry(sub_industry)
        if known is None:
            errors.append(f"sub_industry {why}")
        else:
            sub_industry = known
    errors += _scope_errors(scope, companies, labels, sub_industry)
    if errors:
        return None, errors
    gold = Gold(
        int(row["article_id"]), scope, companies, labels, mentions, sub_industry, tuple(outside)
    )
    return gold, []


@dataclass
class Report:
    """What :func:`load_gold_v2` found in the returned file."""

    good: list[Gold]
    bad: dict[str, list[str]]  # article id as written -> errors
    untagged: int
    missing: list[int]  # expected ids absent from the file
    duplicated: list[int]
    unknown: list[str]  # ids that are not expected (or not integers)
    outside: Counter[str] = field(default_factory=Counter)  # names outside the universe -> rows


class TagFileError(ValueError):
    """The returned review file does not have the expected shape."""


def read_tag_file(path: Path) -> list[dict[str, str]]:
    """The rows of a returned review file, after checking it parses into the expected columns. A
    spreadsheet that saved it with another delimiter or code page gives a clear error, not a file
    that reads as 'untagged'."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TagFileError(
            f"{path.name} is not UTF-8 ({exc.reason}): save it as CSV UTF-8 (comma delimited)"
        ) from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    missing = sorted({"article_id", *TAG_COLUMNS} - set(reader.fieldnames or ()))
    if missing:
        raise TagFileError(
            f"{path.name} lacks the columns {missing}: it was probably saved with another "
            "delimiter or code page; save it as CSV UTF-8 (comma delimited)"
        )
    rows = list(reader)
    if any((row.get(LEGACY_COLUMN) or "").strip() for row in rows):
        raise TagFileError(
            f"{path.name} still has the old {LEGACY_COLUMN!r} column with content: run 'migrate' "
            "(the analysed companies are now the ones in company_labels, separated by commas)"
        )
    return rows


def migrate_rows(rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    """The rows of the first format (a ``companies`` column, ``;`` separators) in the current one:
    no ``companies`` column, ``company_labels`` and ``mentions`` separated by commas. Every
    company of the old column that has no entry in ``company_labels`` is added to it, as typed and
    without a label, and listed in the problems: nothing the owner wrote is dropped or guessed."""
    out: list[dict[str, str]] = []
    problems: list[str] = []
    for row in rows:
        new = {k: v for k, v in row.items() if k != LEGACY_COLUMN}
        tag = f"article {row.get('article_id', '?')}"
        items = split_list((row.get("company_labels") or "").replace(";", ","))
        named = {item.rpartition(":")[0].strip().casefold() or item.casefold() for item in items}
        for company in split_list((row.get(LEGACY_COLUMN) or "").replace(";", ",")):
            if company.casefold() not in named:
                items.append(company)
                problems.append(
                    f"{tag}: {company!r} was in companies but had no entry in company_labels"
                )
        problems += [
            f"{tag}: {item!r} has no label (write name:label)" for item in items if ":" not in item
        ]
        new["company_labels"] = ", ".join(items)
        new["mentions"] = ", ".join(split_list((row.get("mentions") or "").replace(";", ",")))
        out.append(new)
    return out, problems


def load_gold_v2(
    path: Path, expected_ids: Sequence[int] | None = None, universe: Universe | None = None
) -> Report:
    """Read the returned review file and check it row by row and as a whole: every expected
    article id present exactly once, no unknown or non-integer ids."""
    report = Report([], {}, 0, [], [], [])
    seen: Counter[int] = Counter()
    expected = set(expected_ids or ())
    for row in read_tag_file(path):
        raw_id = (row.get("article_id") or "").strip()
        if not raw_id.isdigit():
            report.unknown.append(raw_id or "<blank>")
            continue
        article_id = int(raw_id)
        seen[article_id] += 1
        if expected_ids is not None and article_id not in expected:
            report.unknown.append(raw_id)
        if not (row.get("scope") or "").strip():
            report.untagged += 1
            continue
        gold, errors = check_row(row, universe)
        if gold:
            report.good.append(gold)
            report.outside.update(set(gold.outside))
        else:
            report.bad[raw_id] = errors
    report.duplicated = sorted(i for i, n in seen.items() if n > 1)
    report.missing = sorted(expected - set(seen))
    return report


#: One line per scope for the owner's guide file (`docs/labelling-spec-v2.md` has the rules).
SCOPE_GUIDE = {
    "unrelated": "Not about business, finance or the economy; a company only as the setting.",
    "market_economy": "The market as a whole or the economy; a company event that moves the market follows its headline event.",
    "sector": "One industry is the point (swap the company and it still holds). Needs sub_industry. A company story framed by its industry is single.",
    "investing_general": "Investing or market advice about no company and no sector.",
    "multi": "Two or more companies each with a sentence of their own, also from one industry; a named peer with a sentence makes a comparison multi. Label every analysed one.",
    "single": "One company is the subject, even if framed by its industry or moving the market; a comparison with its sector or peers is single + peer_comparison. Context companies go in mentions.",
}
NEW_LABEL_GUIDE = {
    "reputational_harm": "Damages or repairs public standing: apologies, outrages, boycotts, pledges after criticism.",
    "competitive_actions": "A company's own move or statement against a competitor: share fights, rival facilities, pricing pressure.",
    "investor_activity": "A stake moves: fund or famous investor buys or sells, activist position, 13F. A view alone is market_analyst_sentiment.",
    "peer_comparison": "A writer or analyst weighs companies against each other; every compared company gets it (a single company against its sector too).",
    "other": "The event is clear, about the company, but fits none of the 13 labels.",
    "uncertain": "Two labels fit equally and the headline event does not decide, or the event is not in the first 6,000 characters.",
}


COLUMN_GUIDE = {
    "company_labels": "The analysed companies, each with its label, separated by commas: 'Apple:product_innovation, Google:other'. Tickers, or names and nicknames as the article writes them: 'normalise' converts them to tickers (and lists what it could not or inferred). Single: one entry; multi: two or more; the other scopes: none.",
    "mentions": "Companies named but not analysed, separated by commas, in every scope (names or tickers).",
    "sub_industry": "A GICS sub-industry, as its 8-digit code (see gics_sub_industries.csv) or its name. Required for a sector body; optional on every other scope.",
}


def guide_rows() -> list[dict[str, str]]:
    """The scopes and company labels of the spec as rows of a CSV guide."""
    rows = [{"kind": "scope", "name": k, "meaning": v} for k, v in SCOPE_GUIDE.items()]
    rows += [
        {"kind": "label", "name": slug, "meaning": display} for slug, display, _ in CATEGORY_LABELS
    ]
    rows += [{"kind": "label", "name": k, "meaning": v} for k, v in NEW_LABEL_GUIDE.items()]
    rows += [{"kind": "column", "name": k, "meaning": v} for k, v in COLUMN_GUIDE.items()]
    return rows


def sub_industry_rows(companies: Sequence[Company]) -> list[dict[str, str]]:
    """The distinct (GICS sector, sub-industry) pairs of the universe, with the 8-digit GICS code
    of each (empty when the name is not in the table), for the owner to copy."""
    code_of = {_sub_key(name): code for code, name in SUB_INDUSTRY_CODES.items()}
    pairs = sorted({(c.gics_sector, c.gics_sub_industry) for c in companies if c.gics_sub_industry})
    return [
        {"gics_sector": a, "gics_sub_industry": b, "gics_code": code_of.get(_sub_key(b), "")}
        for a, b in pairs
    ]
