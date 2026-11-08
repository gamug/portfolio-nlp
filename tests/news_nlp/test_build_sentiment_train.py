"""The builder of the training marks (T-148c1, D35): pairing, hashes, the design of the sample."""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from collections import Counter
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
                    "company": f"Company {number}",
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
    with pytest.raises(gold_metrics.MarkedFileError, match="pub_date that is not ISO"):
        BUILD.build(marks, ids)


def test_the_draw_must_have_what_the_training_needs() -> None:
    ids = _ids()
    ids[7]["text_hash"] = ""
    with pytest.raises(gold_metrics.MarkedFileError, match="without"):
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


def test_every_problem_is_named_at_once_with_its_count_and_where_the_text_parts() -> None:
    ids = _ids()
    marks = _marks(ids)
    for m, d in zip(marks[:20], ids[:20], strict=True):
        m["text"] = d["text"].replace("summary", "summarÃ­")  # UTF-8 read as a code page
    for m in marks[20:23]:
        m["pub_date"] = "7/5/23 10:00"
    marks[30]["sentiment"] = ""
    marks[31]["sentiment"] = "bullish"
    with pytest.raises(gold_metrics.MarkedFileError) as caught:
        BUILD.build(marks, ids)
    text = str(caught.value)
    assert "20 rows whose text is not the text the draw recorded" in text
    assert "at character 8" in text and "looks like an encoding change" in text
    assert "3 rows with a pub_date that is not ISO" in text and "'7/5/23 10:00'" in text
    assert "1 row without a mark" in text
    assert "1 row with a mark outside" in text and "'bullish'" in text
    assert len(ids[0]["key"]) > 0 and ids[0]["key"] in text  # the first keys are named


def test_the_marks_are_normalised_but_a_missing_column_is_named(tmp_path: Path) -> None:
    ids = _ids()
    marks = _marks(ids)
    marks[0]["sentiment"] = "  Positive "
    assert BUILD.build(marks, ids)[0]["sentiment"] == "positive"
    path = tmp_path / "t.csv"
    path.write_text("article_id,kind,text\n1,summary,x\n", encoding="utf-8")
    with pytest.raises(gold_metrics.MarkedFileError, match="lacks the columns"):
        BUILD.read_marks(path)


def test_an_encoding_change_and_the_first_difference_are_recognised() -> None:
    assert BUILD.looks_like_an_encoding_change("Café opens", "CafÃ© opens")
    assert BUILD.looks_like_an_encoding_change("Café opens", "Caf? opens")
    assert not BUILD.looks_like_an_encoding_change("Café opens", "Café opened")
    assert (
        BUILD.first_difference("abcdef", "abXdef") == "at character 2: drawn 'cdef', marked 'Xdef'"
    )
    assert BUILD.first_difference("abc", "abcd").startswith("at character 3")


def test_a_fill_down_and_a_missing_class_are_warned_about_not_stopped() -> None:
    ids = _ids()
    marks = _marks(ids)
    rows = BUILD.build(marks, ids)
    assert BUILD.warnings(rows) == []
    for m in marks[:25]:  # a fill-down over the first 25 rows
        m["sentiment"] = "neutral"
    filled = BUILD.build(marks, ids)  # builds all the same
    found = "\n".join(BUILD.warnings(filled))
    assert "rows in a row" in found and "fill-down" in found
    for m in marks:
        m["sentiment"] = "positive" if m["sentiment"] != "negative" else "negative"
    for m in marks[:2]:
        m["sentiment"] = "negative"
    few = BUILD.warnings(BUILD.build(marks, ids))
    assert any("no training mark is neutral" in line for line in few)


def _files(tmp_path: Path, marks: list[dict[str, str]], ids: list[dict[str, str]]) -> Path:
    out = tmp_path / "data"
    out.mkdir(exist_ok=True)
    for name, rows in ((BUILD.IDS_NAME, ids), (BUILD.TO_TAG_NAME, marks)):
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return out


def test_a_corrected_mark_gets_in_only_with_overwrite_marks_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _ids()
    marks = _marks(ids)
    out = _files(tmp_path, marks, ids)
    gold = tmp_path / "gold" / "sentiment_train.csv"
    base = ["build", "--out", str(out), "--gold", str(gold)]
    monkeypatch.setattr(sys, "argv", base)
    BUILD.main()
    before = list(csv.DictReader(gold.open(encoding="utf-8")))
    marks[3]["sentiment"] = "negative" if marks[3]["sentiment"] != "negative" else "neutral"
    _files(tmp_path, marks, ids)
    with pytest.raises(SystemExit, match="would change or go missing"):
        BUILD.main()
    assert list(csv.DictReader(gold.open(encoding="utf-8"))) == before  # untouched
    monkeypatch.setattr(sys, "argv", [*base, "--overwrite-marks"])
    BUILD.main()
    after = list(csv.DictReader(gold.open(encoding="utf-8")))
    assert [r["key"] for r in after] == [r["key"] for r in before]  # the same order and rows
    assert [r["key"] for r in after if r != before[after.index(r)]] == [before[3]["key"]]
    assert after[3]["sentiment"] == marks[3]["sentiment"]


def test_a_text_that_changed_since_the_first_build_is_never_accepted_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _ids()
    marks = _marks(ids)
    out = _files(tmp_path, marks, ids)
    gold = tmp_path / "gold" / "sentiment_train.csv"
    monkeypatch.setattr(sys, "argv", ["build", "--out", str(out), "--gold", str(gold)])
    BUILD.main()
    ids[5]["text"] = "A different summary."  # the draw was redone: another text under that key
    ids[5]["text_hash"] = retag.text_hash(ids[5]["text"])
    marks[5]["text"] = ids[5]["text"]
    _files(tmp_path, marks, ids)
    with pytest.raises(SystemExit, match="would change or go missing"):
        BUILD.main()


def test_a_file_in_a_windows_code_page_is_explained_not_a_traceback(tmp_path: Path) -> None:
    marks = _marks(_ids())
    marks[7]["notes"] = "caída fuerte"  # a non-ASCII character
    path = tmp_path / "sentiment_train_to_tag.csv"
    with path.open("w", newline="", encoding="cp1252") as fh:  # Excel's plain "CSV" on Windows
        writer = csv.DictWriter(fh, fieldnames=list(marks[0]))
        writer.writeheader()
        writer.writerows(marks)
    with pytest.raises(gold_metrics.MarkedFileError, match="is not UTF-8") as caught:
        BUILD.read_marks(path)
    assert "line 9" in str(caught.value)  # the header and 7 rows come first
    assert "CSV UTF-8" in str(caught.value)


def test_a_semicolon_file_is_called_by_its_separator_not_by_missing_columns(
    tmp_path: Path,
) -> None:
    marks = _marks(_ids())
    marks[0]["text"] = (
        "Sales rose, costs fell, and margins grew."  # commas in a text, as in the real file
    )
    path = tmp_path / "sentiment_train_to_tag.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:  # a Spanish-locale Excel
        writer = csv.DictWriter(fh, fieldnames=list(marks[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(marks)
    with pytest.raises(gold_metrics.MarkedFileError, match="separated by a semicolon"):
        BUILD.read_marks(path)
    plain = tmp_path / "plain.csv"
    plain.write_text("article_id,kind,text\n1,summary,x\n", encoding="utf-8")
    with pytest.raises(gold_metrics.MarkedFileError, match="lacks the columns"):
        BUILD.read_marks(plain)  # a comma file with columns missing is still said that way


def test_missing_repeated_and_extra_rows_are_named_together_with_the_row_problems() -> None:
    ids = _ids()
    marks = _marks(ids)
    extra_row = dict(marks[0], article_id="999999")
    marks[20]["pub_date"] = "7/5/23"
    marks[21]["pub_date"] = "7/5/23"
    marks[22]["pub_date"] = "7/5/23"
    marks = [*marks[:80], *marks[90:], marks[3], extra_row]  # 10 rows missing, one repeated
    with pytest.raises(gold_metrics.MarkedFileError) as caught:
        BUILD.build(marks, ids)
    text = str(caught.value)
    assert "10 rows of the draw missing from the tagged file" in text
    assert "1 row whose key is repeated in the tagged file" in text
    assert "1 row of the tagged file missing from the draw" in text
    assert "3 rows with a pub_date that is not ISO" in text


def test_an_empty_and_a_header_only_file_say_they_have_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "sentiment_train_to_tag.csv"
    path.write_bytes(b"")
    with pytest.raises(gold_metrics.MarkedFileError, match="has no rows"):
        BUILD.read_marks(path)
    path.write_text(",".join(BUILD.NEEDED_FROM_THE_OWNER) + "\n", encoding="utf-8")
    with pytest.raises(gold_metrics.MarkedFileError, match="has no rows"):
        BUILD.read_marks(path)


def test_a_missing_tagged_file_says_where_to_save_the_owners_copy(tmp_path: Path) -> None:
    with pytest.raises(gold_metrics.MarkedFileError) as caught:
        BUILD.read_marks(tmp_path / "sentiment_train_to_tag.csv")
    assert "save your tagged copy at" in str(caught.value)
    assert "writes only the untagged one" in str(caught.value)
    assert "draw script of that set" not in str(caught.value)


def test_marks_in_spanish_are_refused_with_the_english_words_named() -> None:
    ids = _ids()
    marks = _marks(ids)
    for m, word in zip(marks[:3], ("positivo", "negativo", "neutro"), strict=True):
        m["sentiment"] = word
    with pytest.raises(gold_metrics.MarkedFileError) as caught:
        BUILD.build(marks, ids)
    text = str(caught.value)
    assert "3 rows with a mark outside" in text and "'positivo'" in text
    assert "write the English word: positive, negative or neutral" in text
    marks[0]["sentiment"] = "bullish"  # a first refused word that is not a Spanish class: no hint
    with pytest.raises(gold_metrics.MarkedFileError) as other:
        BUILD.build(marks, ids)
    assert "write the English word" not in str(other.value)


ROOT = Path(__file__).resolve().parents[2]


def _committed_rows() -> list[dict[str, str]]:
    return list(csv.DictReader((ROOT / "gold" / "sentiment_train.csv").open(encoding="utf-8-sig")))


def test_the_committed_sentiment_train_marks_are_the_owners_450() -> None:
    """Pinned like the other committed gold files: changing a mark is a decision that moves these
    counts (D35: the builder refuses to change a mark without ``--overwrite-marks``)."""
    rows = _committed_rows()
    assert tuple(rows[0]) == retag.SENTIMENT_TRAIN_COLUMNS
    assert len(rows) == 450 and len({r["key"] for r in rows}) == 450
    assert Counter(r["slice"] for r in rows) == {"train": 360, "validation": 90}
    by_slice = {
        s: Counter(r["sentiment"] for r in rows if r["slice"] == s) for s in ("train", "validation")
    }
    assert by_slice["train"] == {"positive": 180, "negative": 96, "neutral": 84}
    assert by_slice["validation"] == {"positive": 52, "neutral": 21, "negative": 17}
    BUILD.check_design(rows)  # the sample is the one drawn: classes, blocks, batches, slices
    assert all(r["sentiment"] in gold_metrics.SENTIMENT_CLASSES for r in rows)
    assert all(len(r["text_hash"]) == 64 and len(r["body_hash"]) == 64 for r in rows)
    assert not {"text", "title", "spanish_text"} & set(rows[0])  # the marks and ids, no news text


def test_the_committed_marks_say_which_company_each_is_about() -> None:
    """The label is anchored to the company of the row (D35), so the file carries it: a company on every
    row, and a blank ticker only where the row is a sector or market text (a bracketed placeholder)."""
    rows = _committed_rows()
    assert {"ticker", "company"} <= set(rows[0])
    assert all(r["company"].strip() for r in rows)
    blank = [r for r in rows if not r["ticker"].strip()]
    assert len(blank) == 26
    assert all(r["company"].startswith("[") and r["rule_scope"] != "single" for r in blank)
    # the dates are the ones of the source, kept as tagged (a protected column): three forms, all ISO
    assert all(
        re.fullmatch(r"\d{4}-\d\d-\d\dT[\d:]{8}(\.\d{3})?(Z|\+0000)", r["pub_date"]) for r in rows
    )


def test_a_row_without_a_company_stops_the_build_and_a_blank_ticker_does_not() -> None:
    ids = _ids()
    rows = BUILD.build(_marks(ids), ids)
    assert rows[0]["ticker"] == "" and rows[0]["company"].startswith("Company ")
    ids[3]["company"] = " "
    with pytest.raises(gold_metrics.MarkedFileError, match=r"without .*company"):
        BUILD.build(_marks(ids), ids)


def test_a_note_is_read_by_its_words_into_one_kind_and_a_missing_column_is_not_an_error() -> None:
    kind = BUILD.note_kind
    assert kind("Tesla not in the news") == "absent"
    assert kind("Nothing relevant about Goldman Sachs") == "nothing relevant"
    assert (
        kind("This is not about the company but about the entire market instead") == "other scope"
    )
    assert kind("Is not about a sector") == "other scope"
    assert kind("Baxter comes from a person name, not for the company") == "namesake"
    assert kind("Investor activity") == "" and kind("") == ""
    rows = [{"notes": "Oracle not in the text"}, {"notes": ""}, {"sentiment": "neutral"}]
    assert {k: len(v) for k, v in BUILD.notes_by_kind(rows).items()} == {"absent": 1}
    lines = BUILD.summary(
        [{**r, "slice": "train", "predicted": "neutral", "sentiment": "neutral"} for r in rows]
    )
    assert any("1 rows have a note; by keyword: absent 1" in ln for ln in lines)


def test_the_committed_counts_of_the_notes_and_the_date_forms_are_the_ones_the_docs_quote() -> None:
    rows = _committed_rows()
    assert sum(1 for r in rows if r["notes"].strip()) == 49
    kinds = BUILD.notes_by_kind(rows)
    assert {k: len(v) for k, v in kinds.items()} == {
        "absent": 9,
        "nothing relevant": 4,
        "other scope": 3,
        "namesake": 3,
    }
    assert all(r["sentiment"] == "neutral" for v in kinds.values() for r in v)
    assert {r["key"] for r in kinds["namesake"]} == {
        "91090#summary",
        "479890#summary",
        "475332#summary",
    }
    forms = Counter(
        "Z.000"
        if r["pub_date"].endswith(".000Z")
        else "Z"
        if r["pub_date"].endswith("Z")
        else "+0000"
        for r in rows
    )
    assert forms == {"+0000": 355, "Z": 77, "Z.000": 18}


def test_the_company_of_a_row_is_protected_once_the_file_has_it() -> None:
    ids = _ids()
    rows = BUILD.build(_marks(ids), ids)
    before = [dict(r) for r in rows]
    assert BUILD.changed_marks(before, rows) == []
    after = [dict(r) for r in rows]
    after[1]["company"] = "Another company"
    assert BUILD.changed_marks(before, after) == [rows[1]["key"]]
    old = [{k: v for k, v in r.items() if k not in ("ticker", "company")} for r in rows]
    assert BUILD.changed_marks(old, after) == []  # the file of 2026-11-08 had neither column


def test_the_committed_training_set_shares_no_article_with_the_two_exams() -> None:
    train = {r["article_id"] for r in _committed_rows()}
    exam = {
        r["article_id"]
        for r in csv.DictReader((ROOT / "gold" / "sentiment_gold.csv").open(encoding="utf-8-sig"))
    }
    heldout = {
        r["article_id"]
        for r in csv.DictReader(
            (ROOT / "gold" / "category_gold_v2_test.csv").open(encoding="utf-8-sig")
        )
    }
    assert len(train) == 450  # one summary per article
    assert not train & exam and not train & heldout  # D35: it trains, never exams
