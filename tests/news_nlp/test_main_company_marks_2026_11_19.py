"""The owner's second version of the main-company marks (2026-11-19): the tracked file, the scorer's piece-level figures, the builder's comparison and the figures the docs quote."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_main_company_marks_2026_11_19 as builder  # noqa: E402  (scripts/ is not a package)
import score_main_company_marks_2026_11_16 as score  # noqa: E402

csv.field_size_limit(sys.maxsize)
GOLD = ROOT / "gold"
MARKS = GOLD / "main_company_marks_2026_11_19.csv"
TEXTS = GOLD / "texts" / "main_company_marks_texts_2026_11_19.csv"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hash(text: str) -> str:  # retag.text_hash, as in test_val_gold_2026_11_16
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def test_the_tracked_marks_are_the_doc_counts_and_keep_their_text() -> None:
    rows = _read(MARKS)
    texts = {r["key"]: r for r in _read(TEXTS)}
    assert len(rows) == 185
    assert sum(r["es_principal"] == "yes" for r in rows) == 166
    assert sum(r["es_principal"] == "no" for r in rows) == 19
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
    cmp = builder.compare(_read(MARKS), old)
    assert cmp == {
        "kept": 80,
        "changed": 2,
        "rows_gone": 39,
        "pieces_gone": 15,
        "new_rows": 105,
        "new_pieces": 60,
    }


def test_the_scored_figures_are_the_doc_figures() -> None:
    rows = score.load(MARKS)
    c = score.at_threshold(rows, 0.5)
    assert (c["tp_main"], c["called_main"], c["yes"]) == (74, 77, 166)
    assert (c["tp_not"], c["called_not"], c["no"]) == (16, 108, 19)
    assert [score.at_threshold(rows, t)["tp_main"] for t in (0.1, 0.3, 0.7)] == [129, 98, 56]
    assert score.by_item(rows) == {
        "items": 104,
        "with_main": 99,
        "found": 65,
        "called": 67,
        "agreed": 65,
    }
    assert [score.by_item(rows, t)["found"] for t in (0.1, 0.2, 0.3, 0.7)] == [92, 84, 79, 49]
    assert round(score.auc(rows), 3) == 0.789
    assert score.by_owner_mains(rows) == {"one": (38, 57), "several": (27, 42)}
    w = score.weighted(rows, 0.5)
    assert (round(100 * w["recall"], 1), round(100 * w["precision"], 1)) == (82.6, 99.1)
    assert round(100 * w["with_main_share"], 1) == 97.5
    assert [round(100 * score.weighted(rows, t)["recall"], 1) for t in (0.3, 0.7)] == [85.7, 79.2]
    lo, hi = score.interval(rows, score.auc)
    assert (round(lo, 3), round(hi, 3)) == (0.635, 0.909)


def test_population_is_the_corpus_of_the_cut() -> None:
    assert sum(score.POPULATION.values()) == 151620


def test_the_piece_level_scorers_on_a_small_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score, "POPULATION", {0: 100, 1: 300})
    rows = [
        score.Mark("a", "X", True, 0.9, 1),
        score.Mark("a", "Y", True, 0.1, 1),
        score.Mark("b", "Z", True, 0.1, 0),
        score.Mark("c", "W", False, 0.2, 0),
    ]
    assert score.by_owner_mains(rows) == {"one": (0, 1), "several": (1, 1)}
    # band 1 (weight 300): its piece is found; band 0 (weight 100, two pieces): one with a main, none found
    w = score.weighted(rows, 0.5)
    assert w["recall"] == pytest.approx(300 / (300 + 50))
    assert w["precision"] == 1.0
    assert score.hit_rate(rows) == pytest.approx(1 / 2)
    assert score.recall_rows(rows) == pytest.approx(1 / 3)


def test_the_comparison_counts_gone_new_and_changed_marks() -> None:
    def row(key: str, mark: str) -> dict[str, str]:
        return {"key": key, "es_principal": mark}

    old = [row("1#A", "no"), row("1#B", "yes"), row("2#A", "yes")]
    new = [row("1#A", "yes"), row("3#A", "no")]
    assert builder.compare(new, old) == {
        "kept": 1,
        "changed": 1,
        "rows_gone": 2,
        "pieces_gone": 1,
        "new_rows": 1,
        "new_pieces": 1,
    }


def test_the_builder_does_not_overwrite_the_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / builder.MARKS).write_text("key\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["build", "--gold", str(tmp_path), "--data", str(tmp_path)])
    with pytest.raises(SystemExit, match="exists"):
        builder.main()


def test_the_doc_and_the_reports_quote_the_scored_figures() -> None:
    doc = (ROOT / "docs" / "main_company_training.md").read_text(encoding="utf-8")
    for pattern in (
        "185 marked rows of 104 pieces",
        "96.1% [89.2, 98.7] (74/77)",
        "44.6% [37.2, 52.2] (74/166)",
        "65 (65.7% [55.9, 74.3])",
        "92 (92.9% [86.1, 96.5])",
        "0.789 [0.635, 0.909]",
        "82.6%",
        "2e7e368d82c93c79802e8d542cf5d1a227a34a65440664f557b775efd1321687",
    ):
        assert pattern in doc, pattern
    for name in ("pipeline_metrics_report.md", "pipeline_metrics_report_es.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for pattern in ("(65/99)", "82.6%", "(129/166)", "0.789"):
            assert pattern in text, f"{name}: {pattern}"
