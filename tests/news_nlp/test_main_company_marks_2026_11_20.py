"""The owner's re-tag of the main-company marks (2026-11-20): the tracked file, the scorer's piece-level figures, the exclusion scenarios, the builder's comparison and the figures the docs quote."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_main_company_marks_2026_11_20 as builder  # noqa: E402  (scripts/ is not a package)
import score_main_company_marks_2026_11_20 as score  # noqa: E402

csv.field_size_limit(sys.maxsize)
GOLD = ROOT / "gold"
MARKS = GOLD / "main_company_marks_2026_11_20.csv"
TEXTS = GOLD / "texts" / "main_company_marks_texts_2026_11_20.csv"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows: list[dict[str, str]] = list(csv.DictReader(fh))
    return rows


def _hash(text: str) -> str:  # retag.text_hash, as in test_val_gold_2026_11_16
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def rows() -> list[score.Mark]:
    loaded: list[score.Mark] = score.load(MARKS)
    return loaded


def test_the_tracked_marks_are_the_doc_counts_and_keep_their_text() -> None:
    rows = _read(MARKS)
    texts = {r["key"]: r for r in _read(TEXTS)}
    assert len(rows) == 185
    assert sum(r["es_principal"] == "yes" for r in rows) == 145
    assert sum(r["es_principal"] == "no" for r in rows) == 40
    assert len({r["item_id"] for r in rows}) == 104
    assert set(texts) == {r["key"] for r in rows}
    for r in rows:
        assert r["text_hash"] == texts[r["key"]]["text_hash"] == _hash(texts[r["key"]]["text"])


def test_the_earlier_version_is_not_touched() -> None:
    old = _read(GOLD / "main_company_marks.csv")
    assert len(old) == 119
    assert (
        hashlib.sha256((GOLD / "main_company_marks.csv").read_bytes()).hexdigest()
        == "413a4bae73be2a297ab58e84b2f5adfa5a2145c0509d7c51ab12dfb8601667ad"
    )
    assert builder.compare(_read(MARKS), old) == {
        "kept": 80,
        "changed": 15,
        "rows_gone": 39,
        "pieces_gone": 15,
        "new_rows": 105,
        "new_pieces": 60,
    }


def test_the_scored_figures_are_the_doc_figures(rows: list[score.Mark]) -> None:
    c = score.at_threshold(rows, 0.5)
    assert (c["tp_main"], c["called_main"], c["yes"]) == (71, 77, 145)
    assert (c["tp_not"], c["called_not"], c["no"]) == (34, 108, 40)
    assert [score.at_threshold(rows, t)["tp_main"] for t in (0.1, 0.3, 0.7)] == [123, 93, 55]
    assert score.by_item(rows) == {
        "items": 104,
        "with_main": 88,
        "found": 61,
        "called": 67,
        "agreed": 61,
    }
    assert [score.by_item(rows, t)["found"] for t in (0.1, 0.2, 0.3, 0.7)] == [86, 78, 73, 47]
    assert round(score.auc(rows), 3) == 0.829
    assert score.by_owner_mains(rows) == {"one": (35, 50), "several": (26, 38)}
    w = score.weighted(rows, 0.5)
    assert (round(100 * w["recall"], 1), round(100 * w["precision"], 1)) == (88.0, 97.7)
    assert round(100 * w["with_main_share"], 1) == 90.2
    lo, hi = score.interval(rows, score.auc)
    assert (round(lo, 3), round(hi, 3)) == (0.730, 0.912)


def test_the_piece_flags_are_the_owner_notes_and_falta(rows: list[score.Mark]) -> None:
    flags = score.piece_flags(rows)
    no_company = sorted(k for k, v in flags.items() if "no_company" in v)
    missed = sorted(k for k, v in flags.items() if "missed" in v)
    assert no_company == ["322112", "335253"]
    assert len(missed) == 28
    # a `falta` yes with no note (402174), and a note with `falta` no (Exxon, 250692), both count
    assert "402174" in missed and "250692" in missed
    # the note that names a company of the universe is not a missing main company
    assert "252646" not in missed
    assert not set(no_company) & set(missed)


def test_the_scenarios_leave_out_the_flagged_pieces(rows: list[score.Mark]) -> None:
    sc = score.scenarios(rows)
    assert {k: len(score.group(v)) for k, v in sc.items()} == {
        "all pieces": 104,
        "without no-company pieces": 102,
        "without missed-main pieces": 76,
        "without both": 74,
    }
    assert {k: len(v) for k, v in sc.items()} == {
        "all pieces": 185,
        "without no-company pieces": 183,
        "without missed-main pieces": 138,
        "without both": 136,
    }
    tp = {k: score.at_threshold(v, 0.5)["tp_main"] for k, v in sc.items()}
    called = {k: score.at_threshold(v, 0.5)["called_main"] for k, v in sc.items()}
    assert (tp["all pieces"], called["all pieces"]) == (71, 77)
    assert (tp["without no-company pieces"], called["without no-company pieces"]) == (71, 76)
    assert (tp["without missed-main pieces"], called["without missed-main pieces"]) == (64, 65)
    assert (tp["without both"], called["without both"]) == (64, 64)
    assert [round(score.auc(v), 3) for v in sc.values()] == [0.829, 0.845, 0.910, 0.978]


def test_the_flags_on_a_small_fixture() -> None:
    def mark(item: str, note: str = "", missing: bool = False) -> score.Mark:
        return score.Mark(item, "X", False, 0.1, 0, note, missing)

    rows = [
        mark("a", "Not about any company, I propose to exclude this text"),
        mark("b", "Macy's is missing, It does not belongs to S&P500"),
        mark("c", "", missing=True),
        mark("d", "Multiple companies and the only one recognized is in S&P500"),
        mark("e", "Company not in text"),
    ]
    flags = score.piece_flags(rows)
    assert flags["a"] == {"no_company"}
    assert flags["b"] == {"missed"}
    assert flags["c"] == {"missed"}
    assert "d" not in flags
    assert flags["e"] == {"missed"}


def test_the_builder_does_not_overwrite_the_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / builder.MARKS).write_text("key\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["build", "--gold", str(tmp_path), "--data", str(tmp_path)])
    with pytest.raises(SystemExit, match="exists"):
        builder.main()


def test_the_doc_quotes_the_scored_figures() -> None:
    doc = (ROOT / "docs" / "main_company_training.md").read_text(encoding="utf-8")
    for pattern in (
        "185 marked rows of 104 articles",
        "main precision 92.2% (71/77)",
        "49.0% [41.0, 57.0] (71/145)",
        "69.3% [59.0, 78.0] (61/88)",
        "97.7% [92.1, 99.4] (86/88)",
        "91.0% [81.8, 95.8] (61/67)",
        "0.766 [0.662, 0.873]",
        "0.829 [0.730, 0.912]",
        "88.0%",
        "1205a302477ef651bac099ac1470a913b737be34c793f0dc84d770bae5fb5589",
        "97386a486e766fe848288e66087f95eb66fbce0658d98e19f2ee7cc0a8fa139d",
    ):
        assert pattern in doc, pattern
    for name in ("pipeline_metrics_report.md", "pipeline_metrics_report_es.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for pattern in ("(61/88)", "(71/145)", "0.829", "(71/77)"):
            assert pattern in text, f"{name}: {pattern}"
