"""The revision guard (T-220, PR A): the code each step hashes into its revision, pinned.

A revision is a hash of the code a step runs, so editing that code (even a comment-free refactor, even deleting a v1 class from a shared module) gives
the step a new revision and makes every finished item pending again: a rerun of hours, and rows the delivery note names (``snapshow_delivery.md``)
stop being the current ones. The depuration removes a lot of v1 code; this test is what fails if it touches a line a v2 step reads.

If a pin fails on purpose (a step really changes): the revision is new, the upstream team gets a new delivery note first (T-220, "Before the prune"),
and the pin is updated in the same PR with the reason. The hashes are of the code *as parsed* (``normalized_code``), so a comment or a docstring does
not move them. They depend on the Python that parses it (``ast.unparse``): CI's 3.12.

**The cut is stricter** (found by this guard, 2026-11-18): ``retag.code_revision`` hashes the **bytes** of five modules (``attribution_model``, ``chunking``,
``company_texts``, ``triage``, ``attribution_units``), so a comment, a blank line or a line ending in any of them moves the cut revision (``units5-...``).
The depuration must not reformat those five files at all."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable

import pytest

import news_category_stage as category
import news_company_stage as companies
import news_period_stage as period
import news_sentiment_stage as sentiment
import news_snapshot_stage as snapshot
import news_summary_stage as summary
import sector_stage as sector
from news_nlp import attribution_model as am
from news_nlp import retag

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] != (3, 12), reason="the pins are hashes of code parsed by Python 3.12"
)

WHY = "this changes the revision of the step: finished items become pending and the delivered rows are no longer the current ones (T-220 guard)"

#: What the delivery note names as the snapshot revision. The snapshot revision *is* the hash of its code, so it is checkable with no database.
DELIVERED_SNAPSHOT = "snapshot-c3a753affa5d"


def _of(parts: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def test_the_snapshot_revision_is_the_delivered_one() -> None:
    assert snapshot.revision() == DELIVERED_SNAPSHOT, WHY


@pytest.mark.parametrize(
    ("name", "pin", "actual"),
    [
        ("summary (step 2)", "92262d1e2520", lambda: _of(summary.hashed_sources())),
        ("companies (step 4)", "96028ecfcfd8", lambda: _of(companies.hashed_sources())),
        ("category (step 7)", "1a8b34b4b6cf", category._scoring_code),
        ("sentiment (step 8)", "59b617019127", sentiment._scoring_code),
        ("period (step 9)", "ea2a5428975f", period._code),
        ("sector (step 6)", "c0a9cd68027b", lambda: sector.code_hash(sector._module_source())),
        ("cut (step 1.2)", "be358e64f554", lambda: retag.code_revision(*am.UNITS_CODE_MODULES)),
    ],
)
def test_the_code_a_step_hashes_is_the_pinned_one(
    name: str, pin: str, actual: Callable[[], str]
) -> None:
    assert actual() == pin, f"{name}: {WHY}"


def test_the_period_fold_is_the_pinned_one() -> None:
    """Part of the period revision (``FOLDED_INTO_OTHER``, T-213): another set is another revision and a new delivery note."""
    assert sorted(period.FOLDED_INTO_OTHER) == [
        "capital_shareholder_returns",
        "leadership_governance",
        "product_innovation",
        "reputational_harm",
    ], WHY


def test_the_shared_modules_of_the_v1_chain_are_read_by_the_v2_steps() -> None:
    """Why ``summary_stage``, ``ner_stage``, ``chunking`` and ``sentiment_pair`` stay byte-identical in the depuration: the v2 revisions hash them."""
    assert "chunking.Chunk" in summary.hashed_sources()
    assert any(key.startswith("ner_stage.") for key in companies.hashed_sources())
    assert any(key.startswith("chunking.") for key in companies.hashed_sources())


def test_the_cut_reads_the_bytes_of_five_modules() -> None:
    """The five files the depuration must not touch, not even a comment (see the module docstring): ``units5-`` hashes their bytes."""
    assert sorted(m.__name__ for m in am.UNITS_CODE_MODULES) == [
        "chunking",
        "news_nlp.attribution_model",
        "news_nlp.attribution_units",
        "news_nlp.company_texts",
        "news_nlp.triage",
    ]
