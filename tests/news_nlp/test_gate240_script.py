"""``scripts/draw_gate240_2026_11_04.py``: draw once, never overwrite the owner's file, merge chain."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp import retag

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "draw_gate240_2026_11_04.py"
COLUMNS = [
    "review_order",
    "article_id",
    "title",
    "text",
    "spanish_title",
    "spanish_text",
    "v1_scope_derived",
    *retag.TAG_COLUMNS,
]
QUOTAS = {"sector": 2, "single": 3}


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("draw_gate240", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review(path: Path, *, tagged: dict[int, dict[str, str]] | None = None, n: int = 12) -> None:
    tagged = tagged or {}
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(1, n + 1):
            row = dict.fromkeys(COLUMNS, "")
            row.update(
                review_order=str(i),
                article_id=str(100 + i),
                title=f"t{i}",
                text="b",
                v1_scope_derived="sector" if i <= 4 else "single",
            )
            row.update(tagged.get(i, {}))
            writer.writerow(row)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def work(tmp_path: Path) -> tuple[Path, Path]:
    out = tmp_path / "data"
    out.mkdir()
    _review(out / retag.REVIEW_NAME, tagged={5: {"scope": "single"}})
    return out, tmp_path / "gold"


def test_draw_writes_the_set_the_tagging_file_and_the_tracked_record(
    work: tuple[Path, Path],
) -> None:
    out, gold = work
    _script().cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)
    ids = _rows(out / "gate240_ids.csv")
    assert len(ids) == 5 and len(_rows(out / "gate240_to_tag.csv")) == 5
    assert "105" in {i["article_id"] for i in ids}  # the tagged row is in
    tracked = _rows(gold / "gate240_ids.csv")
    assert list(tracked[0]) == ["review_order", "article_id", "stratum"]  # no news text
    record = json.loads((gold / "gate240_draw.json").read_text("utf-8"))
    assert record["seed"] == 7 and len(record["review_sha256"]) == 64


def test_a_second_draw_keeps_the_set_and_never_overwrites_the_owners_file(
    work: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out, gold = work
    script = _script()
    script.cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)
    ids_before = (out / "gate240_ids.csv").read_text("utf-8")
    owner = _rows(out / "gate240_to_tag.csv")
    owner[0]["scope"] = "multi"
    script._write(out / "gate240_to_tag.csv", owner)
    with pytest.raises(SystemExit, match="not overwritten"):
        script.cmd_draw(out, retag.REVIEW_NAME, 8, gold, QUOTAS)
    assert (out / "gate240_ids.csv").read_text("utf-8") == ids_before
    assert _rows(out / "gate240_to_tag.csv")[0]["scope"] == "multi"
    assert "--seed 8 ignored" in capsys.readouterr().out


def test_an_empty_gate_file_is_rewritten_not_a_crash(work: tuple[Path, Path]) -> None:
    out, gold = work
    (out / "gate240_to_tag.csv").write_text("", encoding="utf-8")
    _script().cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)
    assert len(_rows(out / "gate240_to_tag.csv")) == 5


def test_a_short_draw_writes_nothing(work: tuple[Path, Path]) -> None:
    out, gold = work
    with pytest.raises(SystemExit, match="Nothing was written"):
        _script().cmd_draw(out, retag.REVIEW_NAME, 7, gold, {"sector": 9, "single": 3})
    assert not (out / "gate240_ids.csv").exists() and not gold.exists()


def test_ids_missing_from_the_review_file_stop_the_run(work: tuple[Path, Path]) -> None:
    out, gold = work
    script = _script()
    script.cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)
    (out / "gate240_to_tag.csv").unlink()
    _review(out / retag.REVIEW_NAME, n=3)  # a different, smaller file
    with pytest.raises(SystemExit, match="not in the review file"):
        script.cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)


def test_merges_chain_from_the_newest_file_and_two_in_one_second_do_not_collide(
    work: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out, gold = work
    script = _script()
    script.cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)
    gate = _rows(out / "gate240_to_tag.csv")
    first = next(g for g in gate if not g["scope"])
    first["scope"] = "multi"
    script._write(out / "gate240_to_tag.csv", gate)
    script.cmd_merge(out, retag.REVIEW_NAME)
    second = next(g for g in gate if not g["scope"])
    second["scope"] = "sector"
    script._write(out / "gate240_to_tag.csv", gate)
    script.cmd_merge(out, retag.REVIEW_NAME)
    merged = sorted(out.glob("*_merged_*.csv"))
    assert len(merged) == 2
    latest = {r["article_id"]: r["scope"] for r in _rows(merged[-1])}
    assert latest[first["article_id"]] == "multi" and latest[second["article_id"]] == "sector"
    original = {r["article_id"]: r["scope"] for r in _rows(out / retag.REVIEW_NAME)}
    assert original[first["article_id"]] == ""  # the review file itself is never touched
    assert "newest" in capsys.readouterr().out


def _conflicting(work: tuple[Path, Path], script: ModuleType) -> tuple[Path, Path]:
    out, gold = work
    script.cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)
    gate = _rows(out / "gate240_to_tag.csv")
    row = next(g for g in gate if g["article_id"] == "105")  # "single" in the review file
    row["scope"] = "multi"
    script._write(out / "gate240_to_tag.csv", gate)
    return out, gold


def test_merge_with_a_conflict_lists_it_writes_the_audit_and_exits_1(
    work: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    out, _ = _conflicting(work, script)
    with pytest.raises(SystemExit, match="NOT complete"):
        script.cmd_merge(out, retag.REVIEW_NAME)
    assert "105 (scope)" in capsys.readouterr().out
    (audit,) = out.glob("*_conflicts_*.csv")
    assert _rows(audit) == [
        {
            "article_id": "105",
            "column": "scope",
            "review_value": "single",
            "gate_value": "multi",
            "resolution": "kept_review",
        }
    ]
    merged = _rows(sorted(out.glob("*_merged_*.csv"))[-1])
    assert next(r for r in merged if r["article_id"] == "105")["scope"] == "single"


def test_merge_gate_wins_takes_the_gate_tags_and_the_audit_says_so(
    work: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    out, _ = _conflicting(work, script)
    script.cmd_merge(out, retag.REVIEW_NAME, gate_wins=True)  # no exit: nothing left in conflict
    assert "replaced by the gate file's tags" in capsys.readouterr().out
    merged = _rows(sorted(out.glob("*_merged_*.csv"))[-1])
    assert next(r for r in merged if r["article_id"] == "105")["scope"] == "multi"
    (audit,) = out.glob("*_conflicts_*.csv")
    assert _rows(audit)[0]["resolution"] == "took_gate"
    assert len(list(out.glob("*_merged_*.csv"))) == 1  # the audit file is not read as a review file


def test_the_local_set_must_equal_the_tracked_one(work: tuple[Path, Path]) -> None:
    out, gold = work
    script = _script()
    script.cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)
    tracked = _rows(gold / "gate240_ids.csv")
    tracked[0]["stratum"] = "multi"
    script._write(gold / "gate240_ids.csv", tracked)
    (out / "gate240_to_tag.csv").unlink()
    with pytest.raises(SystemExit, match="differ"):
        script.cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)


def test_the_recorded_hash_is_of_the_file_the_rows_came_from(work: tuple[Path, Path]) -> None:
    out, gold = work
    script = _script()
    merged = out / "category_gold_v2_review_merged_2026-11-05_000000_000000.csv"
    merged.write_bytes((out / retag.REVIEW_NAME).read_bytes() + b"\n")  # a later current file
    script.cmd_draw(out, retag.REVIEW_NAME, 7, gold, QUOTAS)
    record = json.loads((gold / "gate240_draw.json").read_text("utf-8"))
    assert record["review_file"] == merged.name
    assert record["review_sha256"] == hashlib.sha256(merged.read_bytes()).hexdigest()
