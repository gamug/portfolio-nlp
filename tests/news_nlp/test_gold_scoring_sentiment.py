"""The scorer of a candidate sentiment model on the 180-text gold (T-148c2, D35)."""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections.abc import Callable
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


def _fake_predict(
    answer: dict[str, str] | str,
) -> Callable[..., list[dict[str, str]]]:
    def predict(
        texts: list[dict[str, str]], cfg: dict[str, str] | None = None
    ) -> list[dict[str, str]]:
        return [
            {**t, "predicted": answer if isinstance(answer, str) else answer[t["key"]]}
            for t in texts
        ]

    return predict


IDENTITY = {"weights_sha256": "w" * 8, "id2label": '{"0": "positive"}', "device": "cpu"}


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: dict[str, str] | str,
    *,
    edit_text: bool = False,
) -> Path:
    """A temporary gold, ids and log, a fake model, and ``main`` run on them; returns the log."""
    rows = _gold()
    for i, r in enumerate(rows):
        r["sentiment"] = gold_metrics.SENTIMENT_CLASSES[i % 3]
    gold = tmp_path / "gold"
    gold.mkdir(exist_ok=True)
    _write(gold / "sentiment_gold.csv", rows)
    ids = _ids(rows)
    if edit_text:
        ids[2]["text"] += " edited"
    _write(tmp_path / "ids.csv", ids)
    monkeypatch.setattr(SCORE.gold_draw, "predict", _fake_predict(answer))
    monkeypatch.setattr(SCORE, "describe_model", lambda model, revision: IDENTITY)
    log = gold / "log.csv"
    argv = ["score", "--model", "m", "--gold", str(gold), "--texts", str(tmp_path / "ids.csv")]
    monkeypatch.setattr(sys, "argv", [*argv, "--log", str(log), "--label", "t"])
    SCORE.main()
    return log


def _marks_as_answers() -> dict[str, str]:
    return {r["key"]: gold_metrics.SENTIMENT_CLASSES[i % 3] for i, r in enumerate(_gold())}


def test_main_scores_prints_the_report_and_logs_the_run_with_what_identifies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _run(tmp_path, monkeypatch, _marks_as_answers())
    out = capsys.readouterr().out
    assert "run 1 on the gold" in out and "earlier runs on the gold: 0" in out
    assert "overall agreement (weighted): 100.0%" in out and "agreement by period" in out
    with log.open(encoding="utf-8", newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert list(row) == list(SCORE.LOG_FIELDS)
    assert row["weights_sha256"] == "w" * 8 and row["device"] == "cpu" and row["texts"] == "180"
    assert (
        row["gold_sha256"]
        == SCORE.hashlib.sha256((tmp_path / "gold" / "sentiment_gold.csv").read_bytes()).hexdigest()
    )
    assert (float(row["agreement_point"]), row["agreement_judgement"]) == (1.0, "pass")
    assert float(row["agreement_low"]) <= 1.0 <= float(row["agreement_high"]) + 1e-9


def test_a_second_run_lists_the_first_and_a_third_is_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for _ in range(3):
        _run(tmp_path, monkeypatch, _marks_as_answers())
    out = capsys.readouterr().out
    assert "earlier runs on the gold: 2" in out and "  run 2 t: 100.0%" in out
    assert "warning: this is run 3; D35 plans the baseline and the final run" in out
    assert out.count("warning: this is run") == 1  # runs 1 and 2 are the planned ones


def test_a_model_that_does_not_answer_with_the_three_classes_is_never_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="label_0"):
        _run(tmp_path, monkeypatch, "label_0")
    assert not (tmp_path / "gold" / "log.csv").exists()


def test_an_edited_text_stops_the_run_before_any_scoring_or_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="not the text the owner marked"):
        _run(tmp_path, monkeypatch, "neutral", edit_text=True)
    assert not (tmp_path / "gold" / "log.csv").exists()


def test_a_log_with_other_columns_is_not_appended_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "gold" / "log.csv"
    log.parent.mkdir()
    log.write_text("run,time,label\n1,x,old\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not have the columns this script writes"):
        _run(tmp_path, monkeypatch, "neutral")
    assert log.read_text(encoding="utf-8") == "run,time,label\n1,x,old\n"


def test_the_weights_of_a_local_checkpoint_are_hashed_and_a_hub_id_has_none(
    tmp_path: Path,
) -> None:
    model = tmp_path / "ckpt"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights-1")
    (model / "config.json").write_text("{}", encoding="utf-8")
    first = SCORE.weights_hash(model)
    assert len(first) == 64
    (model / "model.safetensors").write_bytes(b"weights-2")
    assert SCORE.weights_hash(model) != first  # a replaced checkpoint is a different run
    assert SCORE.weights_hash(Path("gamug/FinBERT-financial-news")) == ""
    assert SCORE.weights_hash(tmp_path / "empty-dir-without-weights") == ""
