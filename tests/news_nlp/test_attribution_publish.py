"""The publication of the attribution model of the cut (step 1.2; owner's word, 2026-11-14): the card, the record and the checks. No network, no GPU."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from news_nlp.attribution_model import directory_identity

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
RECORD = ROOT / "docs" / "attribution_training_record.json"
PUBLISHED = ROOT / "config" / "published_models.json"
LOCAL = ROOT / "data" / "models" / "attribution_2026_11_08"
IDENTITY = "8399ccf003c60143"


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


publish = _load("publish_attribution_model_2026_11_14")


def _metrics() -> dict[str, Any]:
    record: dict[str, Any] = json.loads(RECORD.read_text(encoding="utf-8"))
    return record


def test_the_card_says_what_the_model_is_where_it_comes_from_and_its_licence() -> None:
    text = publish.card(_metrics(), IDENTITY)
    assert text.startswith(
        "---\nlicense: cc-by-nc-sa-4.0\nbase_model: gamug/FinBERT-financial-news\n"
    )  # share-alike of the base
    assert (
        IDENTITY in text and "[COMPANY]" in text and "is this sentence about this company?" in text
    )
    metrics = _metrics()
    assert (
        f"{metrics['train']['bodies']:,} bodies" in text
        and f"{metrics['pairs_used']:,} pairs" in text
    )
    assert (
        f"{metrics['report']['all model']['ranking']:.3f}" in text
    )  # the figures come from the record, none is typed
    assert "private" in text and "not pinned" in text.replace("**", "")


def test_the_card_is_the_one_that_was_published() -> None:
    """The sha256 recorded at the upload is that of the card the script writes from the tracked training record: a change of the card or the record fails here."""
    entry = json.loads(PUBLISHED.read_text(encoding="utf-8"))["attribution"]
    assert (
        hashlib.sha256(publish.card(_metrics(), entry["identity"]).encode("utf-8")).hexdigest()
        == entry["card_sha256"]
    )


def test_the_record_names_a_private_repository_a_commit_and_the_identity_of_the_local_weights() -> (
    None
):
    entry = json.loads(PUBLISHED.read_text(encoding="utf-8"))["attribution"]
    assert entry["repo"] == publish.REPO == "gamug/FinBERT-news-attribution"
    assert entry["private"] is True
    assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]) and re.fullmatch(
        r"[0-9a-f]{64}", entry["weights_sha256"]
    )
    assert (
        entry["identity"] == IDENTITY
    )  # the identity the cut's revision names (`local:8399ccf003c60143`)


def test_a_record_is_written_once_and_an_upload_without_a_commit_records_nothing(
    tmp_path: Path,
) -> None:
    published = tmp_path / "published_models.json"
    published.write_text(
        json.dumps({"what": "the models.", "scope": {"repo": "x"}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="no commit SHA"):
        publish.record_upload(published, "dry-run", "card", "w" * 64, IDENTITY)
    assert "attribution" not in json.loads(published.read_text(encoding="utf-8"))
    for _ in range(2):  # a repeat does not add the sentence twice
        publish.record_upload(published, "a" * 40, "card", "w" * 64, IDENTITY)
    saved = json.loads(published.read_text(encoding="utf-8"))
    assert saved["what"].count("attribution: private, not pinned") == 1 and saved["scope"] == {
        "repo": "x"
    }  # the others are kept
    assert saved["attribution"]["card_sha256"] == hashlib.sha256(b"card").hexdigest()


@pytest.mark.skipif(
    not (LOCAL / "model.safetensors").exists(),
    reason="the local checkpoint is git-ignored: only where it exists",
)
def test_the_local_folder_is_the_recorded_model() -> None:
    assert directory_identity(LOCAL) == IDENTITY
    assert (
        json.loads((LOCAL / "metrics.json").read_text(encoding="utf-8")) == _metrics()
    )  # the tracked copy is the training record
    entry = json.loads(PUBLISHED.read_text(encoding="utf-8"))["attribution"]
    assert publish.sha256_file(LOCAL / "model.safetensors") == entry["weights_sha256"]
