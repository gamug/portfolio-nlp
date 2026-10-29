"""docs/labelling-spec-v2.md names every scope and label the re-tag checker accepts."""

from __future__ import annotations

from pathlib import Path

from news_nlp.retag import COMPANY_LABELS, TAG_COLUMNS
from news_nlp.triage import GOLD_SCOPES

SPEC = (Path(__file__).parents[2] / "docs" / "labelling-spec-v2.md").read_text(encoding="utf-8")


def test_the_spec_names_every_scope_label_and_column() -> None:
    for scope in GOLD_SCOPES:
        assert f"`{scope}`" in SPEC, scope
    for label in COMPANY_LABELS:
        assert f"`{label}`" in SPEC, label
    for column in TAG_COLUMNS:
        assert f"`{column}`" in SPEC, column
