"""The draw of the second reader of the category review (T-200): the key says what each option is, the sample is reproducible and the changed rows do not
move when the unchanged ones are added. A swapped option in the key would invalidate the audit silently, so it is tested on synthetic rows."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "draw_audit", ROOT / "scripts" / "draw_category_review_audit_2026_11_15.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n: int = 60) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows, texts = [], {}
    for i in range(1, n + 1):
        changed = i % 3 == 0
        rows.append(
            {
                "review_order": str(i),
                "article_id": str(1000 + i),
                "company": f"Co{i}",
                "label": "market_analyst_sentiment" if changed else "earnings_performance",
                "label_owner": "investor_activity,other" if changed else "earnings_performance",
            }
        )
        texts[str(i)] = {"title": f"t{i}", "text": f"x{i}"}
    return rows, texts


def _alt(_row: dict[str, str]) -> str:
    return "product_innovation"


def test_the_committed_gold_has_226_changed_rows() -> None:
    mod = _module()
    with (ROOT / "gold" / "category_single_gold.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(mod.changed_rows(rows)) == 226
    assert len(mod.unchanged_rows(rows)) == len(rows) - 226


def test_the_key_names_the_file_tag_and_the_owner_tag_of_every_row() -> None:
    mod = _module()
    rows, texts = _rows()
    sheet, key = mod.build(rows, texts, _alt, 10, 5, 1)
    assert len(sheet) == len(key) == 15
    by_order = {r["review_order"]: r for r in rows}
    for s, k in zip(sheet, key, strict=True):
        row = by_order[s[0]]
        assert (s[5], s[6]) == (k[2], k[3])  # the sheet and the key show the same options
        assert sorted((s[5], s[6])) == sorted(
            [row["label"], k[5] if k[1] == "changed" else "product_innovation"]
        )
        assert (k[2], k[3])[int(k[4]) - 1] == row["label"]  # file_option points at the file's tag
        if k[1] == "changed":
            assert k[5] == "investor_activity"  # the owner's first tag
        assert s[7] == s[8] == ""  # no pick yet


def test_the_same_seed_gives_the_same_files_and_the_changed_rows_do_not_move() -> None:
    mod = _module()
    rows, texts = _rows()
    a = mod.build(rows, texts, _alt, 10, 5, 1)
    assert a == mod.build(rows, texts, _alt, 10, 5, 1)
    only_changed = mod.build(rows, texts, _alt, 10, 0, 1)
    with_unchanged = mod.build(rows, texts, _alt, 10, 12, 1)
    kept = [k for k in with_unchanged[1] if k[1] == "changed"]
    assert kept == only_changed[1]  # the same changed rows, the same option order
    assert mod.build(rows, texts, _alt, 10, 5, 2) != a
