"""The scorer of a candidate sentiment model on the 180-text gold (T-148c2, D35)."""

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


SCORE = _script("score_sentiment_gold_2026_11_07")
REPORT = _script("report_wi18_2026_11_06")


def _gold(per_part: dict[str, int] | None = None) -> list[dict[str, str]]:
    """A gold file with the design the figures assume: 40 of each class in ``checked``, 20 in
    ``extractor``; the owner marks every text neutral, half of them from 2022 on."""
    design = per_part or gold_metrics.SENTIMENT_DESIGN
    rows: list[dict[str, str]] = []
    for part, per_class in design.items():
        for label in gold_metrics.SENTIMENT_CLASSES:
            for i in range(per_class):
                key = f"{part}-{label}-{i}#summary"
                rows.append(
                    {
                        "key": key,
                        "article_id": str(len(rows) + 1),
                        "ticker": "",
                        "kind": "summary",
                        "part": part,
                        "text_hash": retag.text_hash(f"text of {key}"),
                        "revision": "r",
                        "stratum": f"{part}/{label}",
                        "extractor": "",
                        "predicted": label,
                        "population": "500",
                        "pub_date": "2023-01-02" if i % 2 else "2019-05-06",
                        "sentiment": "neutral",
                        "notes": "",
                    }
                )
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ids(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"key": r["key"], "text": f"text of {r['key']}"} for r in rows]


def test_the_texts_come_from_the_ids_and_must_be_the_ones_the_owner_marked() -> None:
    rows = _gold()
    items = SCORE.texts_of_the_gold(rows, _ids(rows))
    assert [i["key"] for i in items] == [r["key"] for r in rows]
    edited = _ids(rows)
    edited[3]["text"] += " edited"
    with pytest.raises(gold_metrics.MarkedFileError, match="not the text the owner marked"):
        SCORE.texts_of_the_gold(rows, edited)
    with pytest.raises(gold_metrics.MarkedFileError, match="is not in the ids file"):
        SCORE.texts_of_the_gold(rows, _ids(rows)[1:])


def test_the_candidate_is_compared_with_the_owner_inside_the_strata_of_the_draw(
    tmp_path: Path,
) -> None:
    rows = _gold()
    _write(tmp_path / "sentiment_gold.csv", rows)
    everything_positive = {r["key"]: "positive" for r in rows}
    drawn_by, populations = REPORT._sentiment_rows_from_gold(tmp_path)
    candidate, same_populations = REPORT._sentiment_rows_from_gold(tmp_path, everything_positive)
    assert {r["predicted"] for r in candidate} == {"positive"}
    assert {r["predicted"] for r in drawn_by} == {"positive", "negative", "neutral"}
    assert [r["stratum"] for r in candidate] == [r["stratum"] for r in drawn_by]  # the weights stay
    assert populations == same_populations
    with pytest.raises(gold_metrics.MarkedFileError, match="no prediction"):
        REPORT._sentiment_rows_from_gold(tmp_path, {rows[0]["key"]: "neutral"})


def test_the_report_of_a_candidate_uses_its_predictions(tmp_path: Path) -> None:
    rows = _gold()
    _write(tmp_path / "sentiment_gold.csv", rows)
    agree_all = {r["key"]: "neutral" for r in rows}  # the owner marked every text neutral
    text = "\n".join(REPORT.sentiment_report(tmp_path, gold=True, predictions=agree_all))
    assert "overall agreement (weighted): 100.0%" in text
    assert "overall agreement (weighted): 33" in "\n".join(
        REPORT.sentiment_report(tmp_path, gold=True)
    )


def test_the_period_lines_split_the_agreement_at_2022() -> None:
    gold_rows = [
        {"predicted": "neutral", "gold": "neutral", "pub_date": "2023-01-02"},
        {"predicted": "neutral", "gold": "negative", "pub_date": "2024-01-02"},
        {"predicted": "neutral", "gold": "neutral", "pub_date": "2019-05-06"},
        {"predicted": "neutral", "gold": "neutral", "pub_date": ""},
    ]
    text = "\n".join(SCORE.period_lines(gold_rows))
    assert "2022 on: 2 texts, 50.0%" in text
    assert "before 2022: 1 texts, 100.0%" in text
    assert "1 texts have no usable pub_date" in text


def test_every_run_is_appended_to_the_log_with_its_number(tmp_path: Path) -> None:
    log = tmp_path / "sub" / "log.csv"
    row = {f: "x" for f in SCORE.LOG_FIELDS}
    assert SCORE.log_run(log, row) == 1
    assert SCORE.log_run(log, row) == 2
    with log.open(encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert [r["run"] for r in got] == ["1", "2"]
    assert list(got[0]) == list(SCORE.LOG_FIELDS)


def test_a_text_with_no_chunk_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SCORE.gold_draw,
        "predict",
        lambda texts, cfg=None: texts[:1] and [{**texts[0], "predicted": "neutral"}],
    )
    with pytest.raises(gold_metrics.MarkedFileError, match="no chunk to score"):
        SCORE.score([{"key": "a", "text": "x"}, {"key": "b", "text": "y"}])
