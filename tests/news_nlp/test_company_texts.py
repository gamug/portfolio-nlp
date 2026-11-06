"""news_nlp.company_texts: the text of each company in a one-segment-per-company body (WI18 step 4)."""

from __future__ import annotations

import csv
from pathlib import Path

from news_nlp import company_texts
from news_nlp.triage import Company, CompanyResolver

UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
    Company("ORCL", "Oracle Corporation", "Information Technology", "Systems Software"),
]
RESOLVER = CompanyResolver(UNIVERSE)
EXPECTED_SHAPE = (2834, 33624)  # (bodies, labelled segments) of the tracked training set


def test_structured_body_gives_one_text_per_company_in_order() -> None:
    body = (
        "Stocks making the biggest moves midday:\n"
        "Apple - Shares rose 3% after the company raised its dividend.\n"
        "It also announced a larger buyback than analysts expected.\n"
        "Microsoft - Shares fell 2% as cloud growth slowed in the quarter.\n"
        "Oracle - Shares jumped 5% on a new contract with the Pentagon."
    )
    found = company_texts.split_company_texts(body, RESOLVER)
    assert found is not None
    assert list(found) == ["AAPL", "MSFT", "ORCL"]
    assert "larger buyback" in found["AAPL"]  # the continuation line stays with Apple
    assert "cloud growth" in found["MSFT"] and "Apple" not in found["MSFT"]
    assert "Stocks making" not in found["AAPL"]  # the introduction belongs to no company


def test_label_segments_marks_leads_and_continuations_and_skips_the_introduction() -> None:
    body = (
        "Stocks making the biggest moves midday:\n"
        "Apple - Shares rose 3% after the company raised its dividend.\n"
        "It also announced a larger buyback than analysts expected.\n"
        "Microsoft - Shares fell 2% as cloud growth slowed in the quarter.\n"
        "Oracle - Shares jumped 5% on a new contract with the Pentagon."
    )
    labels = company_texts.label_segments(body, RESOLVER)
    assert labels == [
        company_texts.SegmentLabel(1, "AAPL", "lead"),
        company_texts.SegmentLabel(2, "AAPL", "continuation"),
        company_texts.SegmentLabel(3, "MSFT", "lead"),
        company_texts.SegmentLabel(4, "ORCL", "lead"),
    ]


def test_narrative_body_is_left_to_the_exception_path() -> None:
    body = (
        "Apple and Microsoft are fighting over the same customers in the enterprise market.\n"
        "Analysts say the rivalry will intensify over the next year as budgets tighten.\n"
        "Neither company commented on the report published on Tuesday morning."
    )
    assert company_texts.split_company_texts(body, RESOLVER) is None


def test_one_company_is_not_a_multi_body() -> None:
    body = "Apple - Shares rose 3% after the dividend news.\nApple - More gains followed later."
    assert company_texts.split_company_texts(body, RESOLVER) is None


def test_segment_text_rebuilds_the_text_of_a_label_from_its_position() -> None:
    body = (
        "Intro line of the round-up goes here today:\n"
        "short\n"
        "Apple - Shares rose 3% after the company raised its dividend.\n"
        "Microsoft - Shares fell 2% as cloud growth slowed in the quarter."
    )
    assert company_texts.segment_text(body, 1).startswith("Apple - Shares rose")


def test_an_ambiguous_segment_has_no_label_and_nothing_is_inherited_past_it() -> None:
    body = (
        "Apple - Shares rose 3% after the company raised its dividend.\n"
        "It also announced a larger buyback than analysts expected.\n"
        "Microsoft and Oracle - Both fell as cloud growth slowed this quarter.\n"
        "Cloud spending is expected to slow further next year, analysts said.\n"
        "Oracle - Shares jumped 5% on a new contract with the Pentagon.\n"
        "Microsoft - Shares fell 2% as cloud growth slowed in the quarter."
    )
    labels = company_texts.label_segments(body, RESOLVER)
    assert labels == [
        company_texts.SegmentLabel(0, "AAPL", "lead"),
        company_texts.SegmentLabel(1, "AAPL", "continuation"),
        company_texts.SegmentLabel(4, "ORCL", "lead"),
        company_texts.SegmentLabel(5, "MSFT", "lead"),
    ]


def test_parser_constants_are_pinned_because_the_tracked_positions_depend_on_them() -> None:
    """``gold/company_segments_train.csv`` stores positions in ``segments(body)``. Changing any of
    these (or the paragraph rule) moves every position: rebuild the file with
    ``scripts/build_company_segments_2026_11_05.py`` and update the counts below and in PLAN D27."""
    assert (
        company_texts.LEAD_CHARS,
        company_texts.MIN_SEGMENT,
        company_texts.MIN_SEGMENTS,
        company_texts.MIN_LED_SHARE,
        company_texts.MIN_COMPANIES,
    ) == (50, 20, 4, 0.5, 3)
    body = "short\nA paragraph of exactly twenty.\n\n  Another long enough paragraph.  \nx"
    assert company_texts.segments(body) == [
        "A paragraph of exactly twenty.",
        "Another long enough paragraph.",
    ]


def test_the_tracked_training_set_has_the_recorded_shape() -> None:
    path = Path(__file__).resolve().parents[2] / "gold" / "company_segments_train.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == [
        "article_id",
        "body_hash",
        "year",
        "split",
        "segment_index",
        "ticker",
        "kind",
    ]
    assert {r["kind"] for r in rows} == {"lead", "continuation"}  # ambiguous ones are not rows
    assert {r["split"] for r in rows} == {"train", "val"}
    bodies = {r["body_hash"] for r in rows}
    assert (len(bodies), len(rows)) == EXPECTED_SHAPE
    val = {r["body_hash"] for r in rows if r["split"] == "val"}
    assert 0.08 < len(val) / len(bodies) < 0.12  # one body in ten, by hash
