"""The owner's marks for step 3 (T-170): the draw by relevance band, the blind sheet, the ignored spans and the score."""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from news_nlp.translate import Translation

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "main_company_marks", ROOT / "scripts" / "main_company_marks_2026_11_13.py"
)
assert spec and spec.loader
mm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mm
spec.loader.exec_module(mm)

SUM, CUT, COMP = "summary-t", "units5-t", "companies-t"


def _db(items: dict[int, tuple[str, list[tuple[str, float, int]]]]) -> sqlite3.Connection:
    """items: id -> (state, [(ticker, score, is_main)])."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE sub_news (id INTEGER PRIMARY KEY, extractor_revision TEXT);
        CREATE TABLE news_summary (item_kind TEXT, item_id TEXT, model_revision TEXT, summary_text TEXT);
        CREATE TABLE news_company (item_kind TEXT, item_id TEXT, ticker TEXT, model_revision TEXT,
            relevance_score REAL, is_main INTEGER, mention_count INTEGER);
        CREATE TABLE pipeline_step_state (item_kind TEXT, item_id TEXT, step TEXT, model_revision TEXT, status TEXT);
        """
    )
    for item_id, (state, cands) in items.items():
        conn.execute("INSERT INTO sub_news VALUES (?, ?)", (item_id, CUT))
        conn.execute(
            "INSERT INTO news_summary VALUES ('sub_news', ?, ?, ?)",
            (str(item_id), SUM, f"text {item_id}"),
        )
        conn.execute(
            "INSERT INTO pipeline_step_state VALUES ('sub_news', ?, 'companies', ?, ?)",
            (str(item_id), COMP, state),
        )
        for ticker, score, main in cands:
            conn.execute(
                "INSERT INTO news_company VALUES ('sub_news', ?, ?, ?, ?, ?, 1)",
                (str(item_id), ticker, COMP, score, main),
            )
    return conn


def test_a_score_falls_in_one_band_and_the_last_includes_one() -> None:
    assert [mm.band_of(s) for s in (0.0, 0.099, 0.1, 0.5, 0.899, 0.9, 1.0)] == [0, 0, 1, 3, 4, 5, 5]
    with pytest.raises(ValueError):
        mm.band_of(1.2)


def test_the_items_are_read_with_their_state_text_and_candidates() -> None:
    conn = _db({1: ("done", [("AAA", 0.9, 1), ("BBB", 0.2, 0)]), 2: ("review", [])})
    items = mm.fetch_sub_news(conn, SUM, CUT, COMP)
    assert [(i.item_id, i.state, i.text, i.best) for i in items] == [
        ("1", "done", "text 1", 0.9),
        ("2", "review", "text 2", None),
    ]
    assert [c.ticker for c in items[0].candidates] == ["AAA", "BBB"]
    assert mm.fetch_sub_news(conn, SUM, "units5-other", COMP) == []


def test_the_draw_takes_the_same_number_from_each_band_and_not_the_ones_with_no_candidate() -> None:
    data: dict[int, Any] = {i: ("done", [("AAA", 0.95, 1)]) for i in range(1, 11)}
    data.update({i: ("review", [("BBB", 0.05, 0)]) for i in range(11, 14)})
    data[14] = ("review", [])
    items = mm.fetch_sub_news(_db(data), SUM, CUT, COMP)
    sample, population = mm.draw_items(items, per_band=4, seed=1)
    assert population == {5: 10, 0: 3}
    assert sorted(population) == [0, 5]
    bands = [mm.band_of(i.best) for i in sample]
    assert (
        bands.count(5) == 4 and bands.count(0) == 3
    )  # a band with fewer than asked gives all it has
    assert "14" not in {i.item_id for i in sample}
    assert mm.draw_items(items, 4, 1)[0] == sample  # the same seed, the same draw


def test_the_sheet_has_one_row_per_candidate_and_no_score_nor_band() -> None:
    items = mm.fetch_sub_news(
        _db({1: ("done", [("BBB", 0.9, 1), ("AAA", 0.3, 0)])}), SUM, CUT, COMP
    )
    rows = mm.sheet_rows(items, {"AAA": "Alpha", "BBB": "Beta"})
    assert [(r["ticker"], r["empresa"], r["falta_la_principal"]) for r in rows] == [
        ("AAA", "Alpha", ""),
        ("BBB", "Beta", "-"),
    ]
    assert set(rows[0]) == set(mm.TAG_COLUMNS)
    assert not any("0.9" in v or "0.3" in v for r in rows for v in r.values())


def test_a_sheet_that_exists_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "to_tag.csv"
    mm.write_csv(path, mm.TAG_COLUMNS, [])
    with pytest.raises(SystemExit):
        mm.write_csv(path, mm.TAG_COLUMNS, [])


def test_the_ignored_spans_are_the_organisations_with_no_ticker() -> None:
    texts = {"1": "Acme said that Goldman Sachs and Apple agreed."}

    def spans_of(batch: Any) -> list[list[dict[str, Any]]]:
        return [
            [
                {"entity_type": "ORG", "text": "Goldman Sachs", "start": 20, "end": 33},
                {"entity_type": "ORG", "text": "Apple", "start": 38, "end": 43},
                {"entity_type": "PER", "text": "Jane", "start": 0, "end": 4},
            ]
            for _ in batch
        ]

    found = mm.ignored_spans(texts, spans_of, lambda n: "AAPL" if n == "Apple" else None)
    assert [(f["item_id"], f["span"]) for f in found] == [("1", "Goldman Sachs")]
    assert "Goldman Sachs and Apple" in found[0]["contexto"]


def _marked(tmp_path: Path, marks: list[tuple[str, str, str, float, int]]) -> Path:
    """marks: (item, ticker, owner mark, score, band)."""
    items: dict[str, Any] = {}
    for item, ticker, _, score, band in marks:
        items.setdefault(item, {"band": band, "scores": {}})["scores"][ticker] = score
    meta = {"population": {"0": 100, "5": 100}, "items": items}
    path = tmp_path / "to_tag.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mm.TAG_COLUMNS))
        w.writeheader()
        for item, ticker, mark, _, _ in marks:
            w.writerow(
                {
                    "item_id": item,
                    "texto": "t",
                    "ticker": ticker,
                    "empresa": "",
                    "es_principal": mark,
                    "falta_la_principal": "-",
                    "notas": "",
                }
            )
    path.with_suffix(".strata.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_the_score_of_a_model_that_agrees_with_the_owner(tmp_path: Path) -> None:
    path = _marked(
        tmp_path,
        [
            ("1", "A", "si", 0.95, 5),
            ("2", "B", "si", 0.8, 5),
            ("3", "C", "no", 0.05, 0),
            ("4", "D", "no", 0.2, 0),
        ],
    )
    meta = json.loads(path.with_suffix(".strata.json").read_text())
    rows, missing, unmarked = mm.read_marks(path, meta)
    w = mm.weights(meta, rows)
    assert (len(rows), missing, unmarked) == (4, {}, 0)
    assert mm.weighted_auc([r["score"] for r in rows], [r["owner"] for r in rows], w) == 1.0
    row = next(r for r in mm.by_threshold(rows, w) if r["threshold"] == 0.5)
    assert (row["precision"], row["recall"]) == (1.0, 1.0)


def test_an_unmarked_row_is_counted_and_left_out(tmp_path: Path) -> None:
    path = _marked(tmp_path, [("1", "A", "si", 0.9, 5), ("2", "B", "", 0.1, 0)])
    rows, _, unmarked = mm.read_marks(
        path, json.loads(path.with_suffix(".strata.json").read_text())
    )
    assert (len(rows), unmarked) == (1, 1)


def test_the_weights_bring_each_band_back_to_its_population(tmp_path: Path) -> None:
    path = _marked(
        tmp_path, [("1", "A", "si", 0.9, 5), ("2", "B", "si", 0.8, 5), ("3", "C", "no", 0.1, 0)]
    )
    meta = json.loads(path.with_suffix(".strata.json").read_text())
    rows, _, _ = mm.read_marks(path, meta)
    assert mm.weights(meta, rows) == [50.0, 50.0, 100.0]


def test_auc_is_half_for_ties_and_none_for_one_class() -> None:
    assert mm.weighted_auc([0.5, 0.5], [True, False], [1, 1]) == 0.5
    assert mm.weighted_auc([0.5, 0.6], [True, True], [1, 1]) is None


def test_the_share_with_a_main_company_is_by_item() -> None:
    rows = [
        {"item": "1", "owner": False, "score": 0.9, "band": 5},
        {"item": "1", "owner": True, "score": 0.2, "band": 5},
        {"item": "2", "owner": False, "score": 0.1, "band": 5},
    ]
    assert mm.share_with_main(rows, [1, 1, 1], None) == 0.5
    assert mm.share_with_main(rows, [1, 1, 1], 0.5) == 0.5
    assert mm.share_with_main(rows, [1, 1, 1], 0.05) == 1.0


def test_the_rule_gives_sub_news_their_own_threshold_when_auc_is_low(
    tmp_path: Path, capsys: Any
) -> None:
    path = _marked(
        tmp_path,
        [
            ("1", "A", "no", 0.95, 5),
            ("2", "B", "si", 0.1, 0),
            ("3", "C", "si", 0.9, 5),
            ("4", "D", "no", 0.2, 0),
        ],
    )
    mm.cmd_score(SimpleNamespace(tags=path))
    out = capsys.readouterr().out
    assert "own" in out and "AUC" in out
    assert path.with_suffix(".report.md").exists()


def test_a_sheet_of_another_draw_is_refused(tmp_path: Path) -> None:
    path = _marked(tmp_path, [("1", "A", "si", 0.9, 5)])
    meta = json.loads(path.with_suffix(".strata.json").read_text())
    meta["items"] = {}
    with pytest.raises(SystemExit):
        mm.read_marks(path, meta)


def test_the_spanish_column_goes_right_after_its_english_one() -> None:
    spec2 = importlib.util.spec_from_file_location(
        "translate_marks", ROOT / "scripts" / "translate_marks_sheets_2026_11_13.py"
    )
    assert spec2 and spec2.loader
    tm = importlib.util.module_from_spec(spec2)
    sys.modules[spec2.name] = tm
    spec2.loader.exec_module(tm)
    done = {tm.key_of("hello"): Translation("-", "hola")}
    rows = [{"a": "1", "texto": "hello", "b": "2"}, {"a": "3", "texto": "other", "b": "4"}]
    out = tm.with_columns(rows, (("texto", "texto_es"),), done)
    assert list(out[0]) == ["a", "texto", "texto_es", "b"]
    assert [r["texto_es"] for r in out] == [
        "hola",
        "",
    ]  # an untranslated text is an empty cell, not a guess
