"""The figures of the cut sample the owner marked (T-183): shares of yes per question with an interval, on the whole sample and without the rows the owner's notes call wrongly tagged."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import cut_sample_metrics_2026_11_12 as metrics

Row = dict[str, str]


def _row(order: int, article: int, f: str, o: str, c: str, notes: str = "", text: str = "") -> Row:
    return {
        "review_order": str(order),
        "article_id": str(article),
        "segment_index": "0",
        "text": text or f"piece {order}",
        "faithful": f,
        "one_story": o,
        "complete": c,
        "notes": notes,
    }


def test_the_notes_are_sorted_into_three_kinds_and_the_wrong_tag_wins() -> None:
    assert metrics.kind_of("") == "kept"
    assert metrics.kind_of("The initial part comes from the previous news") == "kept"
    assert metrics.kind_of("Company missunderstood: It was tagged from the title") == "wrong_tag"
    assert metrics.kind_of("Company is lost. BoA is part of the title") == "wrong_tag"
    assert metrics.kind_of("This should be investment_general") == "wrong_tag"
    assert metrics.kind_of("This is a misslabeled news") == "wrong_tag"
    assert metrics.kind_of("There a huge mix of companies there") == "ambiguous"
    assert metrics.kind_of("Two companies there") == "ambiguous"
    # both words in one note: the wrong tag is the cause, so it is that
    assert metrics.kind_of("fragment is investment_general, a mix of companies") == "wrong_tag"
    assert metrics.kind_of("COMPANY MISSUNDERSTOOD") == "wrong_tag"


def test_the_share_has_a_wilson_interval_inside_zero_and_one() -> None:
    share, low, high = metrics.wilson(8, 10)
    assert share == 0.8
    assert 0.49 < low < 0.50 and 0.94 < high < 0.95
    assert metrics.wilson(0, 10)[1] == 0.0 and metrics.wilson(10, 10)[2] == 1.0
    assert metrics.wilson(0, 0) == (0.0, 0.0, 0.0)


def test_the_levels_drop_first_the_wrong_tag_then_the_ambiguous() -> None:
    rows = [
        _row(1, 1, "yes", "yes", "yes"),
        _row(2, 1, "no", "no", "no", "mix of companies"),
        _row(3, 2, "no", "no", "no", "This is a misslabeled news"),
        _row(4, 2, "yes", "yes", "yes", "Company missunderstood"),
    ]
    levels = metrics.levels(rows)
    assert [lv.name for lv in levels] == ["all", "right_tag", "right_tag_unambiguous"]
    assert [lv.pieces for lv in levels] == [4, 2, 1]
    assert [lv.bodies for lv in levels] == [2, 1, 1]
    assert [lv.yes["faithful"] for lv in levels] == [2, 1, 1]
    assert [lv.yes["one_story"] for lv in levels] == [2, 1, 1]


def test_a_clean_piece_is_yes_on_all_three_and_a_clean_body_has_only_clean_pieces() -> None:
    rows = [
        _row(1, 1, "yes", "yes", "yes"),
        _row(2, 1, "yes", "yes", "no"),
        _row(3, 2, "yes", "yes", "yes"),
    ]
    lv = metrics.levels(rows)[0]
    assert lv.clean_pieces == 2
    assert lv.clean_bodies == 1 and lv.bodies == 2


def test_the_body_bootstrap_is_seeded_and_resamples_whole_bodies() -> None:
    rows = [_row(i, i // 3, "yes" if i % 4 else "no", "yes", "yes") for i in range(40)]
    a = metrics.body_bootstrap(rows, "faithful", seed=7, draws=300)
    assert a == metrics.body_bootstrap(rows, "faithful", seed=7, draws=300)
    assert 0.0 <= a[0] <= a[1] <= 1.0
    same = [_row(i, i // 3, "yes", "yes", "yes") for i in range(30)]
    assert metrics.body_bootstrap(same, "faithful", seed=1, draws=100) == (1.0, 1.0)


def test_a_blank_or_odd_mark_stops_the_run(tmp_path: Path) -> None:
    bad = [_row(1, 1, "yes", "", "yes")]
    with pytest.raises(SystemExit, match="review_order 1"):
        metrics.check_marks(bad)
    with pytest.raises(SystemExit, match="maybe"):
        metrics.check_marks([_row(2, 1, "maybe", "yes", "yes")])
    metrics.check_marks(
        [_row(3, 1, " Yes", "NO", "yes")]
    )  # case and spaces are the owner's, not an error


def test_marks_that_do_not_belong_to_the_text_stop_the_run(tmp_path: Path) -> None:
    rows = [_row(1, 1, "yes", "yes", "yes", text="the piece")]
    ids = tmp_path / "ids.csv"
    with ids.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, ["review_order", "article_id", "segment_index", "text_hash", "extractor_revision"]
        )
        w.writeheader()
        w.writerow(
            {
                "review_order": "1",
                "article_id": "1",
                "segment_index": "0",
                "text_hash": metrics.text_hash("the piece"),
                "extractor_revision": "units5-x",
            }
        )
    assert metrics.check_ids(rows, ids) == "units5-x"
    rows[0]["text"] = "an edited piece"
    with pytest.raises(SystemExit, match="review_order 1"):
        metrics.check_ids(rows, ids)


def test_the_report_names_every_excluded_row_and_the_bar() -> None:
    rows = [
        _row(1, 1, "yes", "yes", "yes"),
        _row(2, 2, "no", "no", "no", "This is a misslabeled news"),
        _row(3, 3, "yes", "no", "yes", "Two companies there"),
    ]
    text = metrics.report(rows, revision="units5-x", seed=1, draws=100)
    assert "units5-x" in text and "misslabeled" in text and "Two companies" in text
    assert "80%" in text
    assert "wrong_tag: 1" in text and "ambiguous: 1" in text
