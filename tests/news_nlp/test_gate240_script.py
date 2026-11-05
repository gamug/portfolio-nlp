"""``scripts/draw_gate240_2026_11_04.py``: draw once, never overwrite the owner's file, merge chain."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp import retag

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "draw_gate240_2026_11_04.py"
COLUMNS = list(retag.TAGGING_COLUMNS)  # the one format of every file the owner tags
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
                spanish_title="es",
                spanish_text="es b",
                v1_scope_derived="sector" if i <= 4 else "single",
            )
            row.update(tagged.get(i, {}))
            writer.writerow(row)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def work(tmp_path: Path) -> Path:
    out = tmp_path / "data"
    out.mkdir()
    _review(out / retag.REVIEW_NAME, tagged={5: {"scope": "single"}})
    return out


def test_draw_writes_the_set_and_the_tagging_file(
    work: Path,
) -> None:
    out = work
    _script().cmd_draw(out, retag.REVIEW_NAME, 7, QUOTAS)
    ids = _rows(out / "gate240_ids.csv")
    assert len(ids) == 5 and len(_rows(out / "gate240_to_tag.csv")) == 5
    assert "105" in {i["article_id"] for i in ids}  # the tagged row is in


def test_a_second_draw_keeps_the_set_and_never_overwrites_the_owners_file(work: Path) -> None:
    out = work
    script = _script()
    script.cmd_draw(out, retag.REVIEW_NAME, 7, QUOTAS)
    ids_before = (out / "gate240_ids.csv").read_text("utf-8")
    owner = _rows(out / "gate240_to_tag.csv")
    owner[0]["scope"] = "multi"
    script._write(out / "gate240_to_tag.csv", owner)
    with pytest.raises(SystemExit, match="not overwritten"):
        script.cmd_draw(out, retag.REVIEW_NAME, 8, QUOTAS)
    assert (out / "gate240_ids.csv").read_text("utf-8") == ids_before
    assert _rows(out / "gate240_to_tag.csv")[0]["scope"] == "multi"


def test_an_empty_gate_file_is_rewritten_not_a_crash(work: Path) -> None:
    out = work
    (out / "gate240_to_tag.csv").write_text("", encoding="utf-8")
    _script().cmd_draw(out, retag.REVIEW_NAME, 7, QUOTAS)
    assert len(_rows(out / "gate240_to_tag.csv")) == 5


def test_a_short_draw_writes_nothing(work: Path) -> None:
    out = work
    with pytest.raises(SystemExit, match="Nothing was written"):
        _script().cmd_draw(out, retag.REVIEW_NAME, 7, {"sector": 9, "single": 3})
    assert not (out / "gate240_ids.csv").exists()


def test_ids_missing_from_the_review_file_stop_the_run(work: Path) -> None:
    out = work
    script = _script()
    script.cmd_draw(out, retag.REVIEW_NAME, 7, QUOTAS)
    (out / "gate240_to_tag.csv").unlink()
    _review(out / retag.REVIEW_NAME, n=3)  # a different, smaller file
    with pytest.raises(SystemExit, match="not in the review file"):
        script.cmd_draw(out, retag.REVIEW_NAME, 7, QUOTAS)


def test_merges_chain_from_the_newest_file_and_two_in_one_second_do_not_collide(
    work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = work
    script = _script()
    script.cmd_draw(out, retag.REVIEW_NAME, 7, QUOTAS)
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


def _conflicting(work: Path, script: ModuleType) -> Path:
    out = work
    script.cmd_draw(out, retag.REVIEW_NAME, 7, QUOTAS)
    gate = _rows(out / "gate240_to_tag.csv")
    row = next(g for g in gate if g["article_id"] == "105")  # "single" in the review file
    row["scope"] = "multi"
    script._write(out / "gate240_to_tag.csv", gate)
    return out


def test_merge_with_a_conflict_lists_it_writes_the_audit_and_exits_1(
    work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    out = _conflicting(work, script)
    with pytest.raises(SystemExit, match="NOT complete"):
        script.cmd_merge(out, retag.REVIEW_NAME)
    shown = capsys.readouterr().out
    assert "105 (scope)" in shown and "now the current review file" not in shown
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
    assert not list(out.glob("*_merged_*.csv"))  # an incomplete merge is not a merged file
    (incomplete,) = out.glob("*_incomplete_*.csv")
    assert next(r for r in _rows(incomplete) if r["article_id"] == "105")["scope"] == "single"
    assert script._current_review(out, retag.REVIEW_NAME) == out / retag.REVIEW_NAME


def test_gate_wins_after_an_incomplete_merge_starts_from_the_review_file(work: Path) -> None:
    script = _script()
    out = _conflicting(work, script)
    with pytest.raises(SystemExit):
        script.cmd_merge(out, retag.REVIEW_NAME)
    script.cmd_merge(out, retag.REVIEW_NAME, gate_wins=True)
    (merged,) = out.glob("*_merged_*.csv")  # one current file, the incomplete one is not read
    assert next(r for r in _rows(merged) if r["article_id"] == "105")["scope"] == "multi"
    assert script._current_review(out, retag.REVIEW_NAME) == merged


def test_merge_gate_wins_takes_the_gate_tags_and_the_audit_says_so(
    work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    out = _conflicting(work, script)
    script.cmd_merge(out, retag.REVIEW_NAME, gate_wins=True)  # no exit: nothing left in conflict
    assert "replaced by the gate file's tags" in capsys.readouterr().out
    merged = _rows(sorted(out.glob("*_merged_*.csv"))[-1])
    assert next(r for r in merged if r["article_id"] == "105")["scope"] == "multi"
    (audit,) = out.glob("*_conflicts_*.csv")
    assert _rows(audit)[0]["resolution"] == "took_gate"
    assert len(list(out.glob("*_merged_*.csv"))) == 1  # the audit file is not read as a review file


def test_write_refuses_an_empty_list(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no rows"):
        _script()._write(tmp_path / "x.csv", [])
    assert not list(tmp_path.iterdir())


def test_a_file_open_in_excel_gives_a_clear_message_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def locked(*_a: object, **_k: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("os.replace", locked)
    with pytest.raises(SystemExit, match="close it"):
        _script()._write(tmp_path / "x.csv", [{"a": "1"}])
    assert not list(tmp_path.iterdir())  # neither x.csv nor x.csv.tmp


def test_any_other_write_error_also_removes_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", broken)
    with pytest.raises(OSError, match="disk full"):
        _script()._write(tmp_path / "x.csv", [{"a": "1"}])
    assert not list(tmp_path.iterdir())
