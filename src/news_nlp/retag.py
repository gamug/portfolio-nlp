"""The owner's re-tag of the 720 articles under the new taxonomy: gold v2 (Work item 18, T-146b).

The first tagging ("tags v1") has a label and a note per article. Gold v2 has, per body: the
**scope** (one of :data:`~news_nlp.triage.GOLD_SCOPES`), the **company_labels** (``TICKER:label``
entries separated by commas: single: one company; multi: two or more; a company may carry several
different labels in one article, the same label twice is an error; the analysed companies are
exactly the labelled ones, so there is no separate companies column; sector, market_economy and
investing_general bodies may label the companies they name, unrelated may not), the **mentions**
(companies named but not analysed, D23) and the GICS **sub_industry** (one or more codes of any
level, 2/4/6/8 digits, or names; required for a sector body, allowed on any scope). :func:`retag_rows` builds
the file the owner fills in (the v1 tag shown as guidance, never the LLM's label), and
:func:`check_row` / :func:`load_gold_v2` validate it when it comes back. The rules follow
``docs/labelling-spec-v2.md`` (T-146a): D15's labels in :data:`D15_LABELS`, the per-scope shape in
:func:`_scope_errors`; a test keeps the spec and this module in step. The checker expects tickers;
names the owner typed are converted first by :mod:`news_nlp.company_names` (``normalise``).
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

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
NO_COMPANY_SCOPES = ("unrelated",)
#: Scopes whose subject is not a company but where the owner may still label the companies the
#: article names (2026-11-05: kept and scored as a bonus, never required).
OPTIONAL_COMPANY_SCOPES = ("market_economy", "investing_general", "sector")
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


#: The one format of every file the owner tags (the review file, the gate file, any later draw):
#: the Spanish reading aid right after ``text``, the v1 guidance, then the columns to tag.
TAGGING_COLUMNS = (
    *CARRIED,
    "spanish_title",
    "spanish_text",
    "v1_label",
    "v1_notes",
    "v1_scope_derived",
    *TAG_COLUMNS,
)


#: The format of a file the owner tags for the sentiment of pipeline texts: the same carried columns
#: and the Spanish pair right after ``text``, then ``kind`` (``summary`` or ``company_text``, read
#: only) and the columns to tag.
SENTIMENT_TAG_COLUMNS = ("sentiment", "notes")
SENTIMENT_TAGGING_COLUMNS = (
    *CARRIED,
    "spanish_title",
    "spanish_text",
    "kind",
    *SENTIMENT_TAG_COLUMNS,
)


#: The format of the file the owner reads to check the automatic labels of the step-4 training set:
#: the company the parser chose is in ``ticker`` / ``company``, ``kind`` is ``lead`` or
#: ``continuation`` (read only), ``previous_segment`` is the paragraph before the one to judge (with its Spanish version), and
#: the owner marks ``correct`` (yes / no).
SEGMENT_CHECK_TAG_COLUMNS = ("correct", "notes")
SEGMENT_CHECK_COLUMNS = (
    *CARRIED,
    "spanish_title",
    "spanish_text",
    "kind",
    "previous_segment",
    "spanish_previous_segment",
    *SEGMENT_CHECK_TAG_COLUMNS,
)


#: The format of the file the owner reads to mark the texts an extractor wrote for each company of a
#: narrative multi body (D27, D32): ``text`` is the extracted text of the company in ``ticker``,
#: ``body`` the article it was taken from (each with its Spanish version), and the owner marks
#: ``faithful`` (yes / no). The extractor is not in the file: the rows are shuffled and blind.
FAITHFULNESS_TAG_COLUMNS = ("faithful", "notes")
FAITHFULNESS_COLUMNS = (
    *CARRIED,
    "spanish_title",
    "spanish_text",
    "body",
    "spanish_body",
    *FAITHFULNESS_TAG_COLUMNS,
)


TAG_COLUMNS_ALL = tuple(
    dict.fromkeys(
        (
            *TAG_COLUMNS,
            *SENTIMENT_TAG_COLUMNS,
            *SEGMENT_CHECK_TAG_COLUMNS,
            *FAITHFULNESS_TAG_COLUMNS,
        )
    )
)
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def _is_formula(value: str) -> bool:
    return value.lstrip().startswith(_FORMULA_STARTS) or value.startswith(("\t", "\r"))


def spreadsheet_safe(
    rows: Sequence[Mapping[str, str]], tag_columns: Sequence[str] = TAG_COLUMNS_ALL
) -> list[dict[str, str]]:
    """Copies of *rows* for a file the owner opens in a spreadsheet: a value that starts with ``=``,
    ``+``, ``-``, ``@``, a tab or a return gets a leading apostrophe, so it is shown as text and never
    run as a formula (leading spaces do not hide it). The columns the owner marks (*tag_columns*) are
    left as they are. Only the marks and the id columns (``article_id``, ``review_order``,
    ``ticker``, ``kind``) are read back from these files; the escaped text columns never are, so
    the apostrophe cannot reach a figure. Do not read ``title`` or ``text`` back from them."""
    return [
        {k: ("'" + v if k not in tag_columns and _is_formula(v) else v) for k, v in row.items()}
        for row in rows
    ]


def tagging_format_problems(
    rows: Sequence[Mapping[str, str]], columns: Sequence[str] = TAGGING_COLUMNS
) -> list[str]:
    """What keeps *rows* from being a tagging file of the owner's format: other columns or another
    order (the Spanish columns must follow ``text``) and any row without its Spanish title and text.
    Empty when the file is right."""
    if not rows:
        return ["no rows"]
    problems: list[str] = []
    present = tuple(rows[0])
    if present != tuple(columns):
        missing = [c for c in columns if c not in present]
        extra = [c for c in present if c not in columns]
        problems.append(
            f"columns are not the tagging format (missing {missing}, extra {extra}, "
            f"expected order {', '.join(columns)})"
        )
    for column in ("spanish_title", "spanish_text"):
        empty = [r.get("article_id", "?") for r in rows if not (r.get(column) or "").strip()]
        if empty:
            problems.append(f"{column} is empty for {len(empty)} rows (first: {empty[:5]})")
    return problems


def require_tagging_format(
    rows: Sequence[Mapping[str, str]], name: str, columns: Sequence[str] = TAGGING_COLUMNS
) -> None:
    """Refuse to write a file for the owner to tag unless it has the tagging format."""
    problems = tagging_format_problems(rows, columns)
    if problems:
        raise TagFileError(f"{name} is not written: " + "; ".join(problems))


@dataclass(frozen=True)
class Universe:
    """The SOURCE universe: ticker -> canonical ticker (share classes of one company resolve to
    one, like :class:`~news_nlp.triage.CompanyResolver`), the GICS sub-industry names and the
    sub-industry of each company."""

    tickers: dict[str, str]
    sub_industries: dict[str, str]  # lower case -> source spelling
    company_sub: dict[str, str] = field(default_factory=dict)  # canonical ticker -> sub-industry

    def _by_key(self) -> dict[str, str]:
        return {_sub_key(name): name for name in self.sub_industries.values()}

    def resolve_sub_industry(self, text: str) -> tuple[str | None, str | None]:
        """The GICS code of a node given as a code of **any level** (2 sector, 4 industry group,
        6 industry, 8 sub-industry digits), as a sub-industry name (case and ``&`` / ``and`` do not
        matter) or as the ticker of a company (its own sub-industry), or ``(None, why not)``. A
        code must be a prefix of a code of :data:`~news_nlp.gics_codes.SUB_INDUSTRY_CODES`; an
        8-digit one whose name no company of the universe has is refused, not guessed."""
        raw = text.strip()
        if raw.isdigit():
            return self._resolve_digits(raw)
        by_key = self._by_key()
        name = self.sub_industries.get(raw.lower()) or by_key.get(_sub_key(raw))
        if name is None and raw.upper() in self.tickers:
            name = self.company_sub.get(self.tickers[raw.upper()])
            if name is None:
                return None, f"{raw!r} has no GICS sub-industry in the source"
        if name is None:
            return None, f"{raw!r} is not a GICS code, sub-industry name or ticker of the universe"
        code = _CODE_OF.get(_sub_key(name))
        return (code, None) if code else (None, f"no GICS code known for {name!r}")

    def _resolve_digits(self, raw: str) -> tuple[str | None, str | None]:
        if len(raw) not in _CODE_LENGTHS:
            return None, f"{raw!r} is not a GICS code (2, 4, 6 or 8 digits)"
        if not any(code.startswith(raw) for code in SUB_INDUSTRY_CODES):
            return None, f"{raw!r} is not a GICS code"
        name = SUB_INDUSTRY_CODES.get(raw)
        if name and _sub_key(name) not in self._by_key():
            return None, f"code {raw} is {name!r}, which no company of the universe has"
        return raw, None

    def resolve_sub_industries(self, text: str) -> tuple[tuple[str, ...], list[str]]:
        """Several nodes separated by commas (an article may speak of several related sectors, and
        of a level above the sub-industry): ``("20106015", "1510")``. A name may itself hold a comma
        (``Metal, Glass & Plastic Containers``), so adjacent pieces are tried joined first."""
        pieces = split_list(text)
        codes: list[str] = []
        errors: list[str] = []
        i = 0
        while i < len(pieces):
            for j in range(min(len(pieces), i + 4), i, -1):
                code, _ = self.resolve_sub_industry(", ".join(pieces[i:j]))
                if code is not None:
                    break
            if code is None:
                errors.append(f"sub_industry {self.resolve_sub_industry(pieces[i])[1]}")
                i += 1
                continue
            if code in codes:
                errors.append(f"sub_industry {code}: listed twice")
            codes.append(code)
            i = j
        return tuple(codes), errors

    @classmethod
    def of(cls, companies: Sequence[Company]) -> Universe:
        by_ticker = CompanyResolver(companies).by_ticker
        # keyed by canonical ticker: a share class that lacks a sub-industry takes its family's
        company_sub: dict[str, str] = {}
        for c in companies:
            if c.gics_sub_industry:
                company_sub.setdefault(by_ticker[c.ticker.upper()], c.gics_sub_industry)
        return cls(
            by_ticker,
            {
                c.gics_sub_industry.lower(): c.gics_sub_industry
                for c in companies
                if c.gics_sub_industry
            },
            company_sub,
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


_CODE_LENGTHS = (2, 4, 6, 8)  # sector, industry group, industry, sub-industry
_CODE_OF = {_sub_key(name): code for code, name in SUB_INDUSTRY_CODES.items()}


_split = split_list


@dataclass(frozen=True)
class Gold:
    """One validated gold-v2 article."""

    article_id: int
    scope: str
    companies: tuple[str, ...]
    labels: dict[
        str, tuple[str, ...]
    ]  # company -> its D15 labels (one article may give a company several)
    mentions: tuple[str, ...]
    sub_industry: tuple[str, ...]  # GICS codes of any level (2, 4, 6 or 8 digits), possibly several
    outside: tuple[
        str, ...
    ] = ()  # companies (analysed or mentioned) outside the universe, as typed


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
    if ";" in raw or ":" in raw:
        errors.append(f"{raw!r}: looks like several entries; separate them with commas")
        return None
    outside.append(raw)  # a delisted or foreign company written as a ticker (USG) lands here too
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
) -> dict[str, tuple[str, ...]]:
    """``AAPL:product_innovation, MSFT:other`` -> {ticker: (labels,)}; the keys are the analysed
    companies (a name outside the universe is kept as typed). A company may appear again with
    another label (one article can say two things about it); the same label twice is an error."""
    labels: dict[str, tuple[str, ...]] = {}
    for item in _split(value):
        company, sep, label = item.rpartition(":")
        company, label = company.strip(), label.strip().lower()
        if not sep:
            errors.append(f"{item!r}: no label (write TICKER:label)")
            continue
        found = _company(company, universe, errors, outside)
        if found is None:
            continue
        company = next((c for c in labels if c.casefold() == found.casefold()), found)
        if label not in COMPANY_LABELS:
            errors.append(f"{company}: label {label!r} is not a company label (D15)")
        elif label in labels.get(company, ()):
            errors.append(f"{company}: label {label} listed twice")
        labels[company] = (*labels.get(company, ()), label)
    return labels


def _scope_errors(
    scope: str,
    companies: tuple[str, ...],
    labels: dict[str, tuple[str, ...]],
    sub_industry: tuple[str, ...],
) -> list[str]:
    if scope in OPTIONAL_COMPANY_SCOPES or scope in NO_COMPANY_SCOPES:
        errors = (
            [f"scope {scope} analyses no company"]
            if scope in NO_COMPANY_SCOPES and (companies or labels)
            else []
        )
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
    tickers must belong to it (share classes are canonicalised). ``company_labels`` may give a company
    several different labels (the same label twice is an error); ``sub_industry`` takes one or more
    GICS codes of any level or names, or companies standing for their own sub-industry, and the
    returned :class:`Gold` carries the canonical codes."""
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
    sub_text = (row.get("sub_industry") or "").strip()
    sub_industry: tuple[str, ...] = ()
    if universe is not None and sub_text:
        sub_industry, sub_errors = universe.resolve_sub_industries(sub_text)
        errors += sub_errors
    elif sub_text:
        sub_industry = tuple(_split(sub_text))
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
    "sector": "One industry is the point (swap the company and it still holds). Needs sub_industry (a code of any level). Companies it names may be labelled. A company story framed by its industry is single.",
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
    "company_labels": "The analysed companies, each with its label, separated by commas: 'Apple:product_innovation, Google:other'. A company may appear again with another label. Tickers, or names and nicknames as the article writes them: 'normalise' converts them to tickers (and lists what it could not or inferred). Single: one company; multi: two or more; sector, market_economy and investing_general: optional; unrelated: none.",
    "mentions": "Companies named but not analysed, separated by commas, in every scope (names or tickers).",
    "sub_industry": "GICS codes of any level (2 sector, 4 industry group, 6 industry, 8 sub-industry digits; see gics_sub_industries.csv), several separated by commas when the article speaks of several. A sub-industry name or a company (its own sub-industry) also works. Required for a sector body; optional on every other scope. Comments go in notes.",
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


#: The fixed 240-body set of the prompt gate (T-146c), by the scope the v1 notes imply. Gold v2
#: scopes are not known before the owner has tagged, so the stratification uses ``v1_scope_derived``;
#: the rare scopes are taken whole, which is what the gate is least sure about.
GATE_QUOTAS = {
    "sector": 9,
    "market_economy": 9,
    "unrelated": 31,
    "investing_general": 40,
    "multi": 60,
    "single": 91,
}


def gate_rows(
    ids: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    """The review rows of the gate bodies, in file order and unchanged: the same columns as the
    review file (the Spanish aid after ``text``, the v1 columns, the columns to tag) and the tags
    the owner has already made. It is a tagging file of its own, 240 rows. Raises ``ValueError`` when an id is not in *rows*."""
    wanted = {i["article_id"] for i in ids}
    out = [
        dict(r)
        for r in sorted(rows, key=lambda r: int(r["review_order"]))
        if r["article_id"] in wanted
    ]
    missing = sorted(wanted - {r["article_id"] for r in out})
    if missing:
        raise ValueError(
            f"{len(missing)} gate ids are not in the review file (first: {', '.join(missing[:5])}): "
            "the file is not the one the set was drawn from"
        )
    return out


def tagged(row: Mapping[str, str]) -> bool:
    """A row holds owner work when any tag column has content, not only ``scope``: notes or labels
    typed on a body whose scope is still empty are work too."""
    return any((row.get(c) or "").strip() for c in TAG_COLUMNS)


def _tagged_by_id(rows: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    """The tagged rows by ``article_id``; raises :class:`TagFileError` on a repeated tagged id
    (which copy would win is not knowable)."""
    by_id: dict[str, Mapping[str, str]] = {}
    for r in rows:
        if not tagged(r):
            continue
        if r["article_id"] in by_id:
            raise TagFileError(f"article_id {r['article_id']} is tagged twice in the gate file")
        by_id[r["article_id"]] = r
    return by_id


def _tags(row: Mapping[str, str]) -> dict[str, str]:
    return {c: (row.get(c) or "").strip() for c in TAG_COLUMNS}


#: The tracked gold file: the tags only. The article text (and its Spanish aid) stays out of git:
#: it is news text under its publishers' rights and the repository is public; it is read back from
#: the SOURCE store by ``article_id``.
GOLD_COLUMNS = (
    "article_id",
    "session",
    "ticker",
    "scope",
    "company_labels",
    "mentions",
    "sub_industry",
    "notes",
)


#: The held-out gold also carries the weight of its draw (``stratum``, and ``frame``: the size of that
#: stratum in the whole corpus sample, D30), so a rate can be re-weighted from ``gold/`` alone. It is
#: not ``population`` of ``heldout_ids.csv``, which counts the bodies left to draw from.
TEST_GOLD_COLUMNS = (*GOLD_COLUMNS, "stratum", "frame")


def code_revision(*modules: ModuleType) -> str:
    """A version of code that is not pinned like a model: 12 hex digits of the SHA-256 of the source
    of the given modules (by module name, so the order does not matter). It changes whenever one of
    them is edited, so a mark made on a text written by one version is not read as made on another.
    It covers the code of those modules only, not the company universe or the data they read."""
    digest = hashlib.sha256()
    for module in sorted(modules, key=lambda m: m.__name__):
        digest.update(module.__name__.encode() + b"\0")
        digest.update(Path(str(module.__file__)).read_bytes())
    return digest.hexdigest()[:12]


@functools.cache
def parser_revision() -> str:
    """The revision of the one-segment-per-company parser (``company_texts``) and what it uses."""
    from news_nlp import company_texts, triage  # noqa: PLC0415 -- only a draw needs the modules

    return f"company_texts@{code_revision(company_texts, triage)}"


@functools.cache
def extractor_revision(name: str) -> str:
    """The revision of the extractor *name* that writes the company texts of narrative bodies. Only
    the baseline exists; the trained model (T-148b3) will be ``name@<checkpoint revision>``."""
    if name != "baseline":
        raise ValueError(f"no extractor {name!r}: only 'baseline' exists until T-148b3")
    import chunking  # noqa: PLC0415 -- only a draw needs the modules
    from news_nlp import attribution_baseline, triage  # noqa: PLC0415

    return f"extractor:baseline@{code_revision(attribution_baseline, triage, chunking)}"


def text_hash(text: str) -> str:
    """The identity of a text a mark belongs to: SHA-256 (hex) of the text with the outer
    whitespace stripped, UTF-8. Computed once by the draw and written to its ids file; a gold
    builder copies it from there and never recomputes it (the tagging file may carry the text
    with a spreadsheet-safe apostrophe)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


#: The tracked files of the sets that are not scope tags (``docs/gold_datasets.md``): the owner's
#: marks and the ids of the draw, never the text. The draw's weights travel with each row, so the
#: figures of D32 can be computed from ``gold/`` alone.
SENTIMENT_GOLD_COLUMNS = (
    "key",
    "article_id",
    "ticker",
    "kind",
    "part",
    "text_hash",
    "revision",
    "stratum",
    "extractor",
    "predicted",
    "population",
    "pub_date",
    "sentiment",
    "notes",
)
FAITHFULNESS_GOLD_COLUMNS = (
    "key",
    "article_id",
    "ticker",
    "text_hash",
    "extractors",
    "revisions",
    "pub_date",
    "faithful",
    "notes",
)
SEGMENT_CHECK_GOLD_COLUMNS = (
    "key",
    "article_id",
    "segment_index",
    "ticker",
    "kind",
    "year",
    "correct",
    "notes",
)


def gold_file_rows(
    sessions: Mapping[str, Sequence[Mapping[str, str]]], golds: Mapping[str, Sequence[Gold]]
) -> list[dict[str, str]]:
    """One row per tagged article of every session, in the validated form (tickers, GICS codes),
    sorted by ``article_id``. An article tagged in two sessions is an error: the sessions are
    disjoint by construction."""
    out: dict[int, dict[str, str]] = {}
    for session, rows in sessions.items():
        by_id = {int(r["article_id"]): r for r in rows}
        for gold in golds[session]:
            if gold.article_id in out:
                raise TagFileError(f"article {gold.article_id} is tagged in two sessions")
            src = by_id[gold.article_id]
            out[gold.article_id] = {
                "article_id": str(gold.article_id),
                "session": session,
                "ticker": src.get("ticker", ""),
                "scope": gold.scope,
                "company_labels": ", ".join(
                    f"{c}:{label}" for c, labels in gold.labels.items() for label in labels
                ),
                "mentions": ", ".join(gold.mentions),
                "sub_industry": ", ".join(gold.sub_industry),
                "notes": (src.get("notes") or "").strip(),
            }
    return [out[i] for i in sorted(out)]


def merge_tags(
    target: Sequence[Mapping[str, str]],
    source: Sequence[Mapping[str, str]],
    *,
    source_wins: bool = False,
) -> tuple[list[dict[str, str]], int, list[str], list[str]]:
    """*target* with the tags of the tagged rows of *source* copied in (same ``article_id``):
    ``(rows, copied, conflicts, replaced)``. A row counts as tagged when any tag column has content.
    A target row that already holds different tags is left as it is and listed in *conflicts* as
    ``"<id> (<columns that differ>)"``: nothing the owner typed is overwritten. With *source_wins*
    (the owner's later corrections are in *source*) those rows take the source tags and are listed
    the same way in *replaced*. Raises :class:`TagFileError` when *source* repeats a tagged
    ``article_id``."""
    by_id = _tagged_by_id(source)
    merged: list[dict[str, str]] = []
    copied = 0
    conflicts: list[str] = []
    replaced: list[str] = []
    for row in target:
        new = dict(row)
        other = by_id.get(row["article_id"])
        if other is not None:
            mine, theirs = _tags(row), _tags(other)
            if not any(mine.values()):
                new.update({c: other.get(c) or "" for c in TAG_COLUMNS})
                copied += 1
            elif mine != theirs:
                label = f"{row['article_id']} ({', '.join(c for c in TAG_COLUMNS if mine[c] != theirs[c])})"
                if source_wins:
                    new.update({c: other.get(c) or "" for c in TAG_COLUMNS})
                    replaced.append(label)
                else:
                    conflicts.append(label)
        merged.append(new)
    return merged, copied, conflicts, replaced


def tag_differences(
    target: Sequence[Mapping[str, str]], source: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    """Every tag column where a tagged *target* row and the same article in *source* hold different
    values: ``article_id``, ``column``, ``review_value``, ``gate_value`` (an audit of what a merge
    kept or replaced)."""
    by_id = _tagged_by_id(source)
    out: list[dict[str, str]] = []
    for row in target:
        other = by_id.get(row["article_id"])
        if other is None or not tagged(row):
            continue
        mine, theirs = _tags(row), _tags(other)
        out += [
            {
                "article_id": row["article_id"],
                "column": c,
                "review_value": mine[c],
                "gate_value": theirs[c],
            }
            for c in TAG_COLUMNS
            if mine[c] != theirs[c]
        ]
    return out


def gate_shortfalls(
    rows: Sequence[Mapping[str, str]],
    chosen: Sequence[Mapping[str, str]],
    quotas: Mapping[str, int] = GATE_QUOTAS,
) -> tuple[dict[str, tuple[int, int]], int]:
    """What a draw did not deliver: ``({stratum: (got, quota)} for the strata below quota, number of
    rows whose ``v1_scope_derived`` is not a quota key and so can never be drawn)``."""
    got = Counter(r["stratum"] for r in chosen)
    short = {s: (got[s], q) for s, q in quotas.items() if got[s] < q}
    unknown = sum(1 for r in rows if r.get("v1_scope_derived") not in quotas)
    return short, unknown


def draw_gate(
    rows: Sequence[Mapping[str, str]],
    quotas: Mapping[str, int] = GATE_QUOTAS,
    *,
    seed: int,
) -> list[dict[str, str]]:
    """The gate set: per ``v1_scope_derived`` stratum, the rows the owner has already tagged first
    (in file order), then a seeded random draw of untagged rows up to the quota (the whole stratum
    if it is smaller; see :func:`gate_shortfalls`). Each result row has ``stratum`` and
    ``already_tagged``; the list is in file order. The draw depends on which rows are tagged, so it
    is reproducible only for the same review-file state: the script records the seed and a hash of
    the file. Raises ``ValueError`` when a stratum holds more tagged rows than its quota: dropping
    work the owner has done would be silent."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible sample, not security
    chosen: list[dict[str, str]] = []
    for stratum, quota in quotas.items():
        members = [r for r in rows if r.get("v1_scope_derived") == stratum]
        done = [r for r in members if (r.get("scope") or "").strip()]
        todo = [r for r in members if not (r.get("scope") or "").strip()]
        if len(done) > quota:
            raise ValueError(
                f"stratum {stratum!r}: {len(done)} rows already tagged, quota {quota}; "
                "raise the quota rather than drop tagged rows"
            )
        rng.shuffle(todo)
        picked = done + todo[: quota - len(done)]
        chosen += [
            {
                "review_order": r["review_order"],
                "article_id": r["article_id"],
                "stratum": stratum,
                "already_tagged": "yes" if (r.get("scope") or "").strip() else "no",
            }
            for r in picked
        ]
    return sorted(chosen, key=lambda r: int(r["review_order"]))
