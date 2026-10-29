"""The owner's re-tag of the 720 articles under the new taxonomy: gold v2 (Work item 18, T-146b).

The first tagging ("tags v1") has a label and a note per article. Gold v2 has, per body: the
**scope** (one of :data:`~news_nlp.triage.GOLD_SCOPES`), the **companies** the body is about, a D15
**label per company** and, for a sector body, the GICS **sub-industry**. :func:`retag_rows` builds
the file the owner fills in (the v1 tag shown as guidance, never the LLM's label), and
:func:`check_row` / :func:`load_gold_v2` validate it when it comes back. The label list is D15's;
labelling spec v2 (T-146a) may amend it, in which place only :data:`D15_LABELS` changes.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from news_nlp.taxonomy import CATEGORY_SLUGS
from news_nlp.triage import GOLD_SCOPES, gold_scope

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
TAG_COLUMNS = ("scope", "companies", "company_labels", "sub_industry", "notes")
#: Scopes that name no company: no company, label or sub-industry is expected.
NO_COMPANY_SCOPES = ("market_economy", "investing_general", "unrelated")
REVIEW_NAME = "category_gold_v2_review.csv"
KEY_NAME = "category_gold_v2_key.csv"


def retag_rows(
    v1_review: Sequence[Mapping[str, str]], v1_key: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The v2 review and key rows from the v1 review and key rows (same order). The review rows
    show the v1 label and note and the scope they imply, with EMPTY tag columns; the LLM's label
    stays in the key only."""
    review: list[dict[str, object]] = []
    for row in v1_review:
        v1_label, v1_notes = row["human_label"], row["human_notes"]
        review.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "review_order",
                        "article_id",
                        "ticker",
                        "company",
                        "pub_date",
                        "source_domain",
                        "title",
                        "text",
                    )
                },
                "v1_label": v1_label,
                "v1_notes": v1_notes,
                "v1_scope_hint": gold_scope(v1_label, v1_notes),
                **dict.fromkeys(TAG_COLUMNS, ""),
            }
        )
    return review, [dict(k) for k in v1_key]


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.replace("\n", ";").split(";") if part.strip()]


@dataclass(frozen=True)
class Gold:
    """One validated gold-v2 article."""

    article_id: int
    scope: str
    companies: tuple[str, ...]
    labels: dict[str, str]  # company -> D15 label
    sub_industry: str


def _parse_labels(value: str, companies: tuple[str, ...]) -> tuple[dict[str, str], list[str]]:
    labels: dict[str, str] = {}
    errors: list[str] = []
    for item in _split(value):
        company, _, label = item.partition(":")
        company, label = company.strip().upper(), label.strip().lower()
        if label not in D15_LABELS:
            errors.append(f"{company}: label {label!r} is not a D15 label")
        elif company not in companies:
            errors.append(f"{company}: labelled but not in companies")
        labels[company] = label
    return labels, errors


def _scope_errors(
    scope: str, companies: tuple[str, ...], labels: dict[str, str], sub_industry: str
) -> list[str]:
    if scope in NO_COMPANY_SCOPES:
        return [f"scope {scope} names no company"] if companies or labels else []
    if scope == "sector":
        return [] if sub_industry else ["a sector body needs a sub_industry"]
    needed = 2 if scope == "multi" else 1
    if len(companies) < needed:
        return [f"scope {scope} needs {needed} or more companies"]
    missing = [c for c in companies if c not in labels]
    if scope == "single" and missing:
        return ["the company needs a label"]
    if scope == "multi" and len(missing) == len(companies):
        return ["a multi-company body needs a label for at least one company"]
    return []


def check_row(row: Mapping[str, str]) -> tuple[Gold | None, list[str]]:
    """Validate one returned row; ``(None, errors)`` when it is not usable."""
    scope = (row.get("scope") or "").strip().lower()
    if scope not in GOLD_SCOPES:
        return None, [f"scope {scope!r} is not one of {', '.join(GOLD_SCOPES)}"]
    companies = tuple(c.upper() for c in _split(row.get("companies") or ""))
    labels, errors = _parse_labels(row.get("company_labels") or "", companies)
    sub_industry = (row.get("sub_industry") or "").strip()
    errors += _scope_errors(scope, companies, labels, sub_industry)
    if errors:
        return None, errors
    return Gold(int(row["article_id"]), scope, companies, labels, sub_industry), []


def load_gold_v2(path: Path) -> tuple[list[Gold], dict[int, list[str]], int]:
    """Read the returned review file: the valid rows, the errors by article id, and how many rows
    are still untagged (empty ``scope``)."""
    good: list[Gold] = []
    bad: dict[int, list[str]] = {}
    untagged = 0
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("scope") or "").strip():
                untagged += 1
                continue
            gold, errors = check_row(row)
            if gold:
                good.append(gold)
            else:
                bad[int(row["article_id"])] = errors
    return good, bad, untagged
