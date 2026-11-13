"""The marks of the cut sample move to ``gold/`` (T-183): the owner's free-text notes become a stored kind, the marks stay tied to their text."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_cut_sample_gold_2026_11_12 as gold
import cut_sample_metrics_2026_11_12 as metrics

REV = "units5-aaaaaaaaaaaa"


def _owner_row(
    order: int, article: int, f: str, o: str, c: str, notes: str = "", text: str = ""
) -> dict[str, str]:
    return {
        "review_order": str(order),
        "article_id": str(article),
        "segment_index": "0",
        "ticker": "T",
        "text": text or f"piece {order}",
        "faithful": f,
        "one_story": o,
        "complete": c,
        "notes": notes,
    }


def _ids(rows: list[dict[str, str]], revision: str = REV) -> list[dict[str, str]]:
    return [
        {
            "review_order": r["review_order"],
            "article_id": r["article_id"],
            "segment_index": "0",
            "text_hash": gold.text_hash(r["text"]),
            "extractor_revision": revision,
        }
        for r in rows
    ]


def test_the_notes_become_four_kinds_and_the_scope_wins_over_the_company_over_the_mix() -> None:
    assert gold.kind_of("") == "kept"
    assert gold.kind_of("The initial part comes from the previous news") == "kept"
    assert gold.kind_of("This should be investment_general") == "wrong_scope"
    assert gold.kind_of("This is a misslabeled news") == "wrong_scope"
    assert gold.kind_of("Company missunderstood: It was tagged from the title") == "wrong_company"
    assert gold.kind_of("Company is lost. BoA is part of the title") == "wrong_company"
    assert gold.kind_of("There a huge mix of companies there") == "ambiguous"
    assert gold.kind_of("Two companies there") == "ambiguous"
    assert gold.kind_of("COMPANY MISSUNDERSTOOD") == "wrong_company"
    # two causes in one note: the scope, then the company, then the mix
    assert gold.kind_of("investment_general, a mix of companies") == "wrong_scope"
    assert gold.kind_of("company missunderstood in a mix of companies") == "wrong_company"


def test_the_gold_rows_carry_the_marks_the_kind_the_hash_and_the_cut_revision() -> None:
    rows = [
        _owner_row(1, 10, " Yes", "NO", "yes", "Company missunderstood"),
        _owner_row(2, 10, "yes", "yes", "yes"),
    ]
    marks, texts = gold.build(rows, _ids(rows))
    assert [m["faithful"] for m in marks] == ["yes", "yes"] and marks[0]["one_story"] == "no"
    assert marks[0]["kind"] == "wrong_company" and marks[1]["kind"] == "kept"
    assert (
        marks[0]["text_hash"] == gold.text_hash("piece 1") and marks[0]["extractor_revision"] == REV
    )
    assert set(marks[0]) == set(gold.MARK_COLUMNS)
    assert texts == [
        {"review_order": "1", "text": "piece 1"},
        {"review_order": "2", "text": "piece 2"},
    ]


def test_a_blank_mark_or_a_text_not_tied_to_its_row_stops_the_build() -> None:
    rows = [_owner_row(1, 1, "yes", "", "yes")]
    with pytest.raises(SystemExit, match="review_order 1"):
        gold.build(rows, _ids(rows))
    rows = [_owner_row(1, 1, "yes", "yes", "yes", text="the piece")]
    ids = _ids(rows)
    rows[0]["text"] = "an edited piece"
    with pytest.raises(SystemExit, match="review_order 1"):
        gold.build(rows, ids)


def test_two_cut_revisions_in_one_ids_file_stop_the_build() -> None:
    rows = [_owner_row(1, 1, "yes", "yes", "yes"), _owner_row(2, 1, "yes", "yes", "yes")]
    ids = _ids(rows)
    ids[1]["extractor_revision"] = "units-bbbbbbbbbbbb"
    with pytest.raises(SystemExit, match="revisions"):
        gold.build(rows, ids)


def test_the_files_are_written_and_never_overwritten_and_the_metrics_read_them_back(
    tmp_path: Path,
) -> None:
    rows = [_owner_row(i, i // 2, "yes", "yes", "yes") for i in range(1, 5)]
    marks, texts = gold.build(rows, _ids(rows))
    marks_path, texts_path = tmp_path / "marks.csv", tmp_path / "texts.csv"
    gold.write(marks, texts, marks_path, texts_path)
    with pytest.raises(SystemExit, match="exists"):
        gold.write(marks, texts, marks_path, texts_path)
    loaded, revision = metrics.load(marks_path, texts_path)
    assert revision == REV and len(loaded) == 4 and loaded[0]["text"] == "piece 1"
    with texts_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:  # the text was edited after the build
        w = csv.DictWriter(f, ["review_order", "text"])
        w.writeheader()
        w.writerows({"review_order": t["review_order"], "text": t["text"] + "!"} for t in texts)
    with pytest.raises(SystemExit, match="review_order 1"):
        metrics.load(marks_path, texts_path)


def test_a_note_about_the_news_scopes_the_whole_body_and_a_note_about_a_piece_only_that_piece() -> (
    None
):
    rows = [
        _owner_row(1, 10, "yes", "yes", "yes"),
        _owner_row(
            2, 10, "no", "no", "no", "This is a misslabeled news. This should be investment_general"
        ),
        _owner_row(3, 10, "yes", "yes", "yes", "Company missunderstood"),
        _owner_row(4, 20, "yes", "yes", "yes"),
        _owner_row(
            5, 20, "no", "no", "no", "To be fair, this fragment is more like investment_general"
        ),
        _owner_row(6, 20, "yes", "yes", "yes"),
    ]
    marks, _ = gold.build(rows, _ids(rows))
    assert [m["kind"] for m in marks] == [
        "wrong_scope",
        "wrong_scope",
        "wrong_scope",
        "kept",
        "wrong_scope",
        "kept",
    ]
    assert (
        marks[0]["notes"] == "" and marks[2]["notes"] == "Company missunderstood"
    )  # the notes stay the owner's


def test_a_macro_section_or_unreadable_text_is_not_ambiguity_and_stays_against_the_cut() -> None:
    assert gold.kind_of("There are a large news about market_economy at the beginning") == "kept"
    assert gold.kind_of("I can understand anything :S") == "kept"
    assert gold.kind_of("This is a very broken story") == "kept"
    assert gold.kind_of("There a huge mix of companies there") == "ambiguous"
    assert (
        gold.kind_of(
            "This coming from the news beign a mix of investment, company facts and market_economy"
        )
        == "ambiguous"
    )
