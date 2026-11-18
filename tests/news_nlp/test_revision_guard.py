"""The revision guard (T-220, PR A): the code each step hashes into its revision, **and the values it reads from `config/`**, pinned.

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

LINE_ENDINGS = " (the cut hashes the BYTES of five modules: check line endings first, `.gitattributes` keeps them LF; a CRLF checkout gives another revision. The pin is of today's code: the delivered units5-d3e2cbfa830e is not reproducible, see docs/db_depuration_plan.md)"
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


#: What the revisions read from ``config/``, found by **changing every key of every file one at a time** and recomputing the revisions (T-220, the manager's review
#: of #211): only these values move a revision, so only these are pinned. Any other key (the evidence figures, the batch sizes, the sentiment/category/scope/attribution
#: entries of the shared files, ``category_model.json``, ``attribution_model.json``, ``scope_model.json``) moved none, and an edit of them must not fail this test.
#: The first value of each pair is the revision it moves.
INPUTS: dict[str, dict[str, object]] = {
    "category_step.json": {"category": ["name", "identity"]},
    "pipeline_models.json": {
        "companies": ["ner.model", "ner.revision"],
        "summary_local": ["c_summary.model", "c_summary.revision"],
    },
    "published_models.json": {"companies": ["relevance.repo", "relevance.revision"]},
    "sentiment_step.json": {"sentiment": ["identity"]},
}
#: Files that a revision reads **whole**: every key moves it. Pinned as the SHA-256 of the canonical JSON (parsed, so line endings do not matter).
WHOLE_FILES = {"sector_step.json": "7ce25b9a97b9556c", "sentiment_stack.json": "a4329069a13c7084"}
#: The pinned values of ``INPUTS``, in the order listed there.
VALUES = {
    "category_step.json": ["category-entailment-single-12", "4ac093410757bc50"],
    "pipeline_models.json": [
        "gamug/sec-bert-finer-ord-ner",
        "ba7b9e43e4aa023ec5691f955b276dc58158354c",
        "sshleifer/distilbart-cnn-12-6",
        "a4f8f3ea906ed274767e9906dbaede7531d660ff",
    ],
    "published_models.json": [
        "gamug/deberta-v3-company-relevance",
        "8812c8bc551158f5aa6a9fc3f4ae1895e904e972",
    ],
    "sentiment_step.json": ["685a5f64f427a4ff"],
}
#: The summary prompt (``summary-llm-`` revision) is text, read as bytes.
PROMPT_SHA = "ce66b1385bb6c1d6"
#: Files no revision reads (status lines, ``--fetch``, research records). A new file in ``config/`` is in none of the lists, so it fails: decide which it is.
NOT_READ = {"attribution_model.json", "category_model.json", "scope_model.json"}

CONFIG = Path(__file__).resolve().parents[2] / "config"


def _dig(data: object, dotted: str) -> object:
    for key in dotted.split("."):
        data = data[key]  # type: ignore[index]
    return data


def test_every_file_of_config_is_classified() -> None:
    files = {p.name for p in CONFIG.iterdir() if p.is_file()}
    known = set(INPUTS) | set(WHOLE_FILES) | NOT_READ | {"summary_prompt.md"}
    assert files == known, (
        f"config/ changed its file list: {sorted(files ^ known)}; say whether a revision reads it"
    )


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_values_the_revisions_read_from_config_are_the_pinned_ones(name: str) -> None:
    data = json.loads((CONFIG / name).read_text(encoding="utf-8"))
    keys = [key for keys in INPUTS[name].values() for key in keys]  # type: ignore[attr-defined]
    found = [_dig(data, key) for key in keys]
    assert found == VALUES[name], f"config/{name}: {dict(zip(keys, found, strict=True))}: {WHY}"


@pytest.mark.parametrize("name", sorted(WHOLE_FILES))
def test_the_config_files_that_a_revision_reads_whole_are_the_pinned_ones(name: str) -> None:
    canonical = json.dumps(json.loads((CONFIG / name).read_text(encoding="utf-8")), sort_keys=True)
    assert hashlib.sha256(canonical.encode()).hexdigest()[:16] == WHOLE_FILES[name], (
        f"config/{name}: {WHY}"
    )


def test_the_summary_prompt_is_the_pinned_one() -> None:
    assert (
        hashlib.sha256((CONFIG / "summary_prompt.md").read_bytes()).hexdigest()[:16] == PROMPT_SHA
    ), f"config/summary_prompt.md: {WHY} (line endings count: `.gitattributes`)"


def test_ci_runs_the_guard_on_python_3_12() -> None:
    if os.environ.get("CI"):
        assert ON_312, (
            "the guard pins hashes of code parsed by Python 3.12; CI must run it on 3.12, not skip it"
        )
