"""The builder of the training marks (T-148c1, D35): pairing, hashes, the design of the sample."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp import gold_metrics, retag

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _script("build_sentiment_train_2026_11_07")
DRAW = _script("draw_sentiment_train_2026_11_07")


def _ids(per_class: int = 30) -> list[dict[str, str]]:
    """A drawn set as the draw script writes it: blocks, batches, slices, texts and their hashes."""
    drawn = []
    for n, label in enumerate(("neutral", "positive", "negative")):
        for i in range(per_class):
            number = i + 1000 * (n + 1)
            text = f"A summary number {number}."
            drawn.append(
                {
                    "key": f"{number}#summary",
                    "article_id": str(number),
                    "body_hash": f"h{number}",
                    "kind": "summary",
                    "rule_scope": "single",
                    "revision": "model@rev",
                    "predicted": label,
                    "pool_class_size": "99",
                    "text_hash": retag.text_hash(text),
                    "title": f"Title {number}",
                    "ticker": "",
                    "company": "",
                    "text": text,
                }
            )
    ordered: list[dict[str, str]] = DRAW.in_blocks(drawn, seed=2)
    return ordered


def _marks(ids: list[dict[str, str]]) -> list[dict[str, str]]:
    """The owner's file: the drawn texts with a mark and a publication date."""
    cycle = ("positive", "negative", "neutral")
    return [
        {
            "article_id": r["article_id"],
            "kind": r["kind"],
            "text": r["text"],
            "pub_date": "2023-05-06T10:00:00+0000",
            "sentiment": cycle[i % 3],
            "notes": "",
        }
        for i, r in enumerate(ids)
    ]


def test_the_rows_are_the_marks_and_the_draw_without_the_text() -> None:
    ids = _ids()
    rows = BUILD.build(_marks(ids), ids)
    assert len(rows) == 90
    assert list(rows[0]) == list(BUILD.TRAIN_COLUMNS)
    assert "text" not in rows[0] and "title" not in rows[0]
    assert [r["key"] for r in rows] == [r["key"] for r in ids]  # the order of the draw
    assert rows[0]["sentiment"] == "positive"
    assert rows[0]["slice"] in {"train", "validation"}
    assert rows[0]["pub_date"] == "2023-05-06T10:00:00+0000"


def test_a_text_edited_after_the_draw_stops_the_build() -> None:
    ids = _ids()
    marks = _marks(ids)
    marks[4]["text"] += " edited"
    with pytest.raises(gold_metrics.MarkedFileError, match="not the text the draw recorded"):
        BUILD.build(marks, ids)


def test_one_apostrophe_of_a_spreadsheet_safe_text_is_forgiven_and_not_kept() -> None:
    ids = _ids()
    ids[2]["text"] = "-5% on the day, the company said."
    ids[2]["text_hash"] = retag.text_hash(ids[2]["text"])
    marks = _marks(ids)
    marks[2]["text"] = "'" + ids[2]["text"]  # what a tagging file holds
    rows = BUILD.build(marks, ids)
    assert "text" not in rows[2]
    marks[2]["text"] = "''" + ids[2]["text"]  # two is an edit
    with pytest.raises(gold_metrics.MarkedFileError, match="not the text the draw recorded"):
        BUILD.build(marks, ids)


def test_the_marked_file_must_hold_exactly_the_drawn_texts() -> None:
    ids = _ids()
    with pytest.raises(gold_metrics.MarkedFileError):
        BUILD.build(_marks(ids)[1:], ids)  # a text is missing
    marks = _marks(ids)
    marks[0]["article_id"] = "424242"
    with pytest.raises(gold_metrics.MarkedFileError):
        BUILD.build(marks, ids)
    marks = _marks(ids)
    marks[5]["kind"] = "company_text"
    with pytest.raises(gold_metrics.MarkedFileError):
        BUILD.build(marks, ids)


def test_a_pub_date_a_spreadsheet_rewrote_stops_the_build() -> None:
    ids = _ids()
    marks = _marks(ids)
    marks[1]["pub_date"] = "7/5/23 10:00"
    with pytest.raises(gold_metrics.MarkedFileError, match="not an ISO date"):
        BUILD.build(marks, ids)


def test_the_draw_must_have_what_the_training_needs() -> None:
    ids = _ids()
    ids[7]["text_hash"] = ""
    with pytest.raises(gold_metrics.MarkedFileError, match="has no"):
        BUILD.build(_marks(ids), ids)


def test_the_design_of_the_sample_is_checked() -> None:
    ids = _ids()
    rows = BUILD.build(_marks(ids), ids)
    BUILD.check_design(rows)
    short = [r for r in rows if r["key"] != rows[0]["key"]]
    with pytest.raises(gold_metrics.MarkedFileError, match="predicted classes are not equal"):
        BUILD.check_design(short)
    moved = [dict(r) for r in rows]
    moved[0]["slice"] = "validation" if moved[0]["slice"] == "train" else "train"
    with pytest.raises(gold_metrics.MarkedFileError, match="the file says"):
        BUILD.check_design(moved)
    unbalanced = [dict(r) for r in rows]
    other = next(
        r
        for r in unbalanced
        if r["predicted"] != unbalanced[0]["predicted"] and r["block"] != unbalanced[0]["block"]
    )
    for column in ("block", "batch", "slice"):  # swap two rows of different classes between blocks
        unbalanced[0][column], other[column] = other[column], unbalanced[0][column]
    with pytest.raises(gold_metrics.MarkedFileError, match="blocks without"):
        BUILD.check_design(unbalanced)


def test_two_rows_with_the_same_body_stop_the_build() -> None:
    ids = _ids()
    ids[1]["body_hash"] = ids[0]["body_hash"]
    with pytest.raises(gold_metrics.MarkedFileError, match="share a body"):
        BUILD.build(_marks(ids), ids)


def test_an_existing_file_is_the_reference_for_marks_and_blocks() -> None:
    ids = _ids()
    before = BUILD.build(_marks(ids), ids)
    assert BUILD.changed_marks(before, before) == []
    after = [dict(r) for r in before]
    after[3]["sentiment"] = "negative" if after[3]["sentiment"] != "negative" else "neutral"
    assert BUILD.changed_marks(before, after) == [before[3]["key"]]
    assert BUILD.changed_marks(before, after[1:]) == [before[0]["key"], before[3]["key"]]


def test_main_writes_the_file_and_refuses_to_move_a_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = _ids()
    out = tmp_path / "data"
    out.mkdir()
    with (out / BUILD.IDS_NAME).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ids[0]))
        writer.writeheader()
        writer.writerows(ids)
    marks = _marks(ids)
    with (out / BUILD.TO_TAG_NAME).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(marks[0]))
        writer.writeheader()
        writer.writerows(marks)
    gold = tmp_path / "gold" / "sentiment_train.csv"
    argv = ["build", "--out", str(out), "--gold", str(gold)]
    monkeypatch.setattr(sys, "argv", argv)
    BUILD.main()
    assert len(list(csv.DictReader(gold.open(encoding="utf-8")))) == 90
    assert "train:" in capsys.readouterr().out
    BUILD.main()  # the same marks again: nothing moves
    marks[0]["sentiment"] = "negative" if marks[0]["sentiment"] != "negative" else "neutral"
    with (out / BUILD.TO_TAG_NAME).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(marks[0]))
        writer.writeheader()
        writer.writerows(marks)
    with pytest.raises(SystemExit, match="would change or go missing"):
        BUILD.main()
