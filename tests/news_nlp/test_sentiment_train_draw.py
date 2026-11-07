"""The draw of the training summaries for sentiment (T-148c): who is never drawn, balance by class."""

from __future__ import annotations

import csv
import importlib.util
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

import news_nlp as db
from news_nlp import sentiment_train

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = _script("draw_sentiment_train_2026_11_07")


def _row(n: int, summary: str = "A summary long enough to be read by the owner.") -> dict[str, str]:
    return {
        "body_hash": f"h{n:04d}",
        "article_ids": str(n),
        "rule_scope": "single",
        "status": "summarised",
        "summary": summary,
    }


def test_candidates_leave_out_the_gold_and_held_out_bodies_and_unreadable_lengths() -> None:
    rows = [_row(i) for i in range(10)] + [_row(10, "short"), _row(11, "x" * 5000)]
    got = TRAIN.candidates(rows, {"h0003", "h0004"}, pool=100, seed=1)
    hashes = {r["body_hash"] for r in got}
    assert not hashes & {"h0003", "h0004", "h0010", "h0011"}
    assert len(hashes) == 8


def test_candidates_are_seeded_and_capped() -> None:
    rows = [_row(i) for i in range(50)]
    first = TRAIN.candidates(rows, set(), pool=20, seed=7)
    assert [r["body_hash"] for r in first] == [
        r["body_hash"] for r in TRAIN.candidates(rows, set(), pool=20, seed=7)
    ]
    assert len(first) == 20
    assert [r["body_hash"] for r in first] != [
        r["body_hash"] for r in TRAIN.candidates(rows, set(), pool=20, seed=8)
    ]


def test_the_draw_takes_the_same_number_of_each_predicted_class_or_all_of_a_short_one() -> None:
    scored = (
        [{**_row(i), "text": "t", "predicted": "neutral"} for i in range(30)]
        + [{**_row(100 + i), "text": "t", "predicted": "positive"} for i in range(30)]
        + [{**_row(200 + i), "text": "t", "predicted": "negative"} for i in range(4)]
    )
    chosen, population = TRAIN.draw(scored, per_class=10, seed=3)
    assert Counter(t["predicted"] for t in chosen) == {"neutral": 10, "positive": 10, "negative": 4}
    assert population == {"neutral": 30, "positive": 30, "negative": 4}
    assert {t["pool_class_size"] for t in chosen if t["predicted"] == "negative"} == {"4"}
    assert all(t["text_hash"] for t in chosen)
    assert len({t["body_hash"] for t in chosen}) == len(chosen)


def test_the_run_stops_on_a_cut_that_is_not_finished(tmp_path: Path) -> None:
    for number in (1, 2, 4):  # chunk 3 is missing
        with (tmp_path / f"chunk_{number:05d}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_row(1)))
            writer.writeheader()
            writer.writerow(_row(number))
    with pytest.raises(SystemExit, match="finish the cut first"):
        TRAIN.cut_rows(tmp_path)
    (tmp_path / "chunk_00003.csv").write_text(
        (tmp_path / "chunk_00002.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert len(TRAIN.cut_rows(tmp_path)) == 4


def test_only_summarised_rows_with_a_summary_are_read(tmp_path: Path) -> None:
    rows = [_row(1), {**_row(2), "status": "by_scope", "summary": ""}, {**_row(3), "summary": " "}]
    with (tmp_path / "chunk_00001.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert [r["body_hash"] for r in TRAIN.cut_rows(tmp_path)] == ["h0001"]


def _drawn(per_class: int = 30) -> list[dict[str, str]]:
    return [
        {**_row(i + 1000 * n), "predicted": label, "key": f"{i + 1000 * n}#summary"}
        for n, label in enumerate(("neutral", "positive", "negative"))
        for i in range(per_class)
    ]


def test_blocks_hold_the_same_number_of_each_class_and_mark_the_validation_slice() -> None:
    ordered = TRAIN.in_blocks(_drawn(150), seed=5)
    assert len(ordered) == 450
    for rows in range(15, 451, 15):  # every whole block prefix is balanced
        assert set(Counter(t["predicted"] for t in ordered[:rows]).values()) == {rows // 3}
    batch_one = [t for t in ordered if t["batch"] == "1"]
    assert len(batch_one) == 150
    assert Counter(t["predicted"] for t in batch_one) == {
        "neutral": 50,
        "positive": 50,
        "negative": 50,
    }
    assert Counter((t["batch"], t["slice"]) for t in ordered) == {
        (batch, kind): n for batch in "123" for kind, n in (("train", 120), ("validation", 30))
    }
    validation = [t for t in ordered if t["slice"] == "validation"]
    assert set(Counter(t["predicted"] for t in validation).values()) == {30}  # 10 per batch x 3
    assert {t["block"] for t in validation if t["batch"] == "1"} == {"9", "10"}


def test_the_order_is_seeded_and_does_not_reveal_the_class() -> None:
    first = TRAIN.in_blocks(_drawn(30), seed=5)
    assert [t["key"] for t in first] == [t["key"] for t in TRAIN.in_blocks(_drawn(30), seed=5)]
    assert [t["key"] for t in first] != [t["key"] for t in TRAIN.in_blocks(_drawn(30), seed=6)]
    classes = [t["predicted"] for t in first[:15]]
    assert classes != sorted(classes)  # shuffled inside the block


def test_blocks_stop_on_classes_of_different_sizes() -> None:
    with pytest.raises(SystemExit, match="same number of each"):
        TRAIN.in_blocks(_drawn(30)[:-3], seed=1)


def test_check_draw_stops_on_a_gold_body_a_gold_headline_or_a_repeat() -> None:
    chosen = [{**_row(i), "key": f"{i}#summary", "title": f"Headline number {i}"} for i in range(5)]
    TRAIN.check_draw(chosen, {"h9999"}, {"some other headline"})  # clean
    with pytest.raises(SystemExit, match="gold or held-out bodies"):
        TRAIN.check_draw(chosen, {"h0002"}, set())
    with pytest.raises(SystemExit, match="under a headline"):
        TRAIN.check_draw(chosen, set(), {"headline number 3"})
    with pytest.raises(SystemExit, match="share a key or a body"):
        TRAIN.check_draw([*chosen, chosen[0]], set(), set())


def test_title_keys_compare_words_not_punctuation_or_case() -> None:
    assert TRAIN.title_key("  Apple, Inc.: Q4 EARNINGS beat! ") == "apple inc q4 earnings beat"


def test_the_exam_keys_and_the_syndicated_copies_come_from_the_source(
    source_db_path: Path, results_db_path: Path
) -> None:
    titles = {1: "Acme raises its annual outlook", 2: "ACME raises its annual outlook!", 3: "Short"}
    raw = sqlite3.connect(source_db_path)
    raw.execute("CREATE TABLE discovered_urls (id INTEGER PRIMARY KEY, url TEXT)")
    for article_id, title in titles.items():
        raw.execute("UPDATE articles SET title = ? WHERE id = ?", (title, article_id))
    raw.commit()
    raw.close()
    conn = db.connect_pipeline(results_db=results_db_path, source_db=source_db_path)
    hashes, keys = TRAIN.exam_keys(conn, {1, 3})
    assert len(hashes) == 2
    assert keys == {"acme raises its annual outlook"}  # "Short" is too short to mean anything
    pool = [{"article_ids": "2"}, {"article_ids": "3"}]
    assert TRAIN.without_title_copies(conn, pool, keys) == [{"article_ids": "3"}]  # 2 is a copy
    db.detach_source(conn)
    conn.close()


def _ids_file(path: Path, rows: list[dict[str, str]]) -> None:
    TRAIN.extra._write(path, rows)


def test_an_ids_file_from_before_the_columns_is_renamed_and_put_in_blocks(tmp_path: Path) -> None:
    old = [{**t, "population": "99"} for t in _drawn(30)]
    _ids_file(tmp_path / "ids.csv", old)
    rows, changed = TRAIN.read_ids(tmp_path / "ids.csv", seed=3)
    assert changed
    assert {r["pool_class_size"] for r in rows} == {"99"}
    assert all("population" not in r for r in rows)
    assert {r["block"] for r in rows} == {str(n) for n in range(1, 7)}
    assert sorted(r["key"] for r in rows) == sorted(t["key"] for t in old)  # the same texts


def test_an_ids_file_with_blocks_is_read_back_unchanged(tmp_path: Path) -> None:
    ordered = TRAIN.in_blocks(_drawn(30), seed=3)
    _ids_file(tmp_path / "ids.csv", [{**t, "pool_class_size": "5"} for t in ordered])
    rows, changed = TRAIN.read_ids(tmp_path / "ids.csv", seed=99)
    assert not changed
    assert [r["key"] for r in rows] == [t["key"] for t in ordered]  # the seed does not reorder


def test_an_empty_ids_file_stops_the_run(tmp_path: Path) -> None:
    (tmp_path / "ids.csv").write_text("key,block\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no rows"):
        TRAIN.read_ids(tmp_path / "ids.csv", seed=1)


def test_the_shape_of_the_set_is_written_once() -> None:
    assert [sentiment_train.batch_of(b) for b in (1, 10, 11, 20, 21, 30)] == [1, 1, 2, 2, 3, 3]
    assert [sentiment_train.slice_of(b) for b in (1, 8, 9, 10, 11, 18, 19, 20)] == [
        "train",
        "train",
        "validation",
        "validation",
        "train",
        "train",
        "validation",
        "validation",
    ]
    ordered = TRAIN.in_blocks(_drawn(150), seed=1)  # the draw uses the same rule
    assert all(
        (t["batch"], t["slice"])
        == (
            str(sentiment_train.batch_of(int(t["block"]))),
            sentiment_train.slice_of(int(t["block"])),
        )
        for t in ordered
    )
