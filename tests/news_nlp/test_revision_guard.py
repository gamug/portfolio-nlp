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
import os
import sys
from collections.abc import Callable
from pathlib import Path

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

ON_312 = sys.version_info[:2] == (3, 12)
# Off CI a different Python skips with a reason; **on CI it does not skip** (a skip would look like a pass): `test_ci_runs_the_guard_on_python_3_12` fails there instead.
pytestmark = pytest.mark.skipif(
    not ON_312 and not os.environ.get("CI"),
    reason="the pins are hashes of code parsed by Python 3.12 (`ast.unparse`); run `uv run pytest` with the project's Python",
)

LINE_ENDINGS = " (the cut hashes the BYTES of five modules: check line endings first, `.gitattributes` keeps them LF; a CRLF checkout gives another revision)"
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
    assert actual() == pin, f"{name}: {WHY}" + (LINE_ENDINGS if name.startswith("cut") else "")


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


# The full revisions, with the pins they also contain: the code hashes above do not see an edit of a pin file.
def test_the_full_category_and_sentiment_revisions_are_the_pinned_ones() -> None:
    """``code_revision`` is what ``--status`` prints: the code **and** the pin (``config/category_step.json``, ``config/sentiment_step.json``, ``config/sentiment_stack.json``)
    for fixed dummy inputs. An edit of a pin moves it while ``_scoring_code`` does not."""
    assert category.code_revision("S", "C") == "category-ece30360e9ce", WHY
    assert sentiment.code_revision("S", "C", "N") == "sentiment-41a13b80709e", WHY


#: SHA-256 (first 16 hex) of the bytes of each file of ``config/``. The revisions read the pins, the prompt and the model names from them, so a byte changed here is a new
#: revision of a step. A new file in ``config/`` fails the test on purpose: decide whether a revision reads it, then add it.
CONFIG_PINS = {
    "attribution_model.json": "384a80b5bb0fd688",
    "category_model.json": "f4c0b17c6415da17",
    "category_step.json": "c784d55575cf79d6",
    "pipeline_models.json": "e3cc80334dfa8dcb",
    "published_models.json": "ef80a6289d108929",
    "scope_model.json": "3a595038cf475dae",
    "sector_step.json": "0852b547d363f13c",
    "sentiment_stack.json": "f647b025cb1887ce",
    "sentiment_step.json": "c81efc93b91e5d25",
    "summary_prompt.md": "ce66b1385bb6c1d6",
}


def test_the_config_files_the_revisions_read_are_the_pinned_bytes() -> None:
    config = Path(__file__).resolve().parents[2] / "config"
    found = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in config.iterdir()
        if p.is_file()
    }
    assert set(found) == set(CONFIG_PINS), (
        f"config/ changed its file list: {sorted(set(found) ^ set(CONFIG_PINS))}"
    )
    moved = sorted(name for name in CONFIG_PINS if found[name] != CONFIG_PINS[name])
    assert not moved, (
        f"{moved}: {WHY}; if the bytes differ only by line endings, `.gitattributes` keeps config/ LF"
    )


def test_ci_runs_the_guard_on_python_3_12() -> None:
    if os.environ.get("CI"):
        assert ON_312, (
            "the guard pins hashes of code parsed by Python 3.12; CI must run it on 3.12, not skip it"
        )
