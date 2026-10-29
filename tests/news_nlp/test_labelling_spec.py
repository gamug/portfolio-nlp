"""docs/labelling-spec-v2.md names every scope and label the re-tag checker accepts."""

from __future__ import annotations

from pathlib import Path

from news_nlp.retag import COMPANY_LABELS, NEW_LABEL_GUIDE, SCOPE_GUIDE, TAG_COLUMNS
from news_nlp.triage import GOLD_SCOPES

SPEC = (Path(__file__).parents[2] / "docs" / "labelling-spec-v2.md").read_text(encoding="utf-8")


def test_the_spec_names_every_scope_label_and_column() -> None:
    for scope in GOLD_SCOPES:
        assert f"`{scope}`" in SPEC, scope
    for label in COMPANY_LABELS:
        assert f"`{label}`" in SPEC, label
    for column in TAG_COLUMNS:
        assert f"`{column}`" in SPEC, column


def test_the_guide_carries_the_rules_the_spec_added() -> None:
    """The owner's guide CSV is written from code: its key rules must not lag behind the spec."""
    assert "one industry" in SPEC and "from one industry" in SCOPE_GUIDE["multi"]
    assert "single + peer_comparison" in SCOPE_GUIDE["single"] and "peer_comparison" in SPEC
    assert "headline event" in SPEC and "headline event" in SCOPE_GUIDE["market_economy"]
    assert "own move" in NEW_LABEL_GUIDE["competitive_actions"]
    assert "view alone" in NEW_LABEL_GUIDE["investor_activity"]
    assert set(SCOPE_GUIDE) == set(GOLD_SCOPES)
