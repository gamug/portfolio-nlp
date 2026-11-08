"""The scorer of a candidate sentiment model on the 180-text gold (T-148c2, D35)."""

from __future__ import annotations

import csv
import importlib.util
import json
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
COMMIT = "0123456789abcdef0123456789abcdef01234567"
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
        lambda texts, cfg=None, pair=False: texts[:1] and [{**texts[0], "predicted": "neutral"}],
    )
    with pytest.raises(gold_metrics.MarkedFileError, match="no chunk to score"):
        SCORE.score([{"key": "a", "text": "x"}, {"key": "b", "text": "y"}])


def _fake_predict(
    answer: dict[str, str] | str,
) -> Callable[..., list[dict[str, str]]]:
    def predict(
        texts: list[dict[str, str]], cfg: dict[str, str] | None = None, pair: bool = False
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
    extra: tuple[str, ...] = (),
    model: str = "m",
    revision: str = COMMIT,
    tickers: int = 0,
    calls: list[tuple[int, bool, str]] | None = None,
) -> Path:
    """A temporary gold, ids and log, a fake model, and ``main`` run on them; returns the log."""
    rows = _gold()
    for i, r in enumerate(rows):
        r["sentiment"] = gold_metrics.SENTIMENT_CLASSES[i % 3]
        if i < tickers:
            r["ticker"] = f"T{i}"
    gold = tmp_path / "gold"
    gold.mkdir(exist_ok=True)
    _write(gold / "sentiment_gold.csv", rows)
    ids = _ids(rows)
    if edit_text:
        ids[2]["text"] += " edited"
    _write(tmp_path / "ids.csv", ids)
    fake = _fake_predict(answer)

    def recording(
        texts: list[dict[str, str]], cfg: dict[str, str] | None = None, pair: bool = False
    ) -> list[dict[str, str]]:
        if calls is not None:
            calls.append((len(texts), pair, (cfg or {}).get("model", "pinned")))
        return fake(texts, cfg, pair)

    monkeypatch.setattr(SCORE.gold_draw, "predict", recording)
    monkeypatch.setattr(
        SCORE, "company_names", lambda: {f"T{i}": f"Company {i}" for i in range(180)}
    )
    monkeypatch.setattr(
        SCORE.gold_draw,
        "_models",
        lambda: {"sentiment": {"model": "org/pinned", "revision": "9" * 40}},
    )
    monkeypatch.setattr(SCORE, "describe_model", lambda model, revision: IDENTITY)
    monkeypatch.setattr(
        SCORE,
        "token_counts",
        lambda items, model, revision: {
            i["key"]: 200 if n % 4 == 0 else 50 for n, i in enumerate(items)
        },
    )
    log = gold / "log.csv"
    argv = ["score", "--model", model, "--gold", str(gold), "--texts", str(tmp_path / "ids.csv")]
    if revision:
        argv += ["--revision", revision]
    monkeypatch.setattr(sys, "argv", [*argv, "--log", str(log), "--label", "t", *extra])
    SCORE.main()
    return log


def _stored_answers() -> dict[str, str]:
    """What the model the sample was drawn by said (the ``predicted`` column): a baseline run of the
    pinned model reproduces it."""
    return {r["key"]: r["predicted"] for r in _gold()}


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


@pytest.mark.parametrize(
    ("roles", "warnings"),
    [
        (["baseline", "final"], 0),  # the plan: no warning
        (["baseline", "other", "final"], 1),  # the extra look warns, the planned final does not
        (["other", "other"], 2),
    ],
)
def test_only_an_extra_look_is_warned_about_not_a_planned_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    roles: list[str],
    warnings: int,
) -> None:
    for role in roles:
        answers = _stored_answers() if role == "baseline" else _marks_as_answers()
        _run(tmp_path, monkeypatch, answers, extra=("--role", role))
    out = capsys.readouterr().out
    assert out.count("warning: this is an extra look at the gold") == warnings


def test_the_warning_counts_the_looks_beyond_the_plan_so_far() -> None:
    earlier = [
        {"run": "1", "label": "b", "agreement": "x", "role": "baseline"},
        {"run": "2", "label": "t", "agreement": "x", "role": "other"},
        {"run": "3", "label": "f", "agreement": "x", "role": "final"},
        {"run": "4", "label": "f2", "agreement": "x", "role": "final"},  # a redo
    ]
    assert SCORE.looks_beyond_the_plan(earlier) == 2
    text = "\n".join(SCORE.earlier_lines(earlier, "other", False))
    assert "looks beyond the plan so far: 2, with this one 3" in text
    assert SCORE.earlier_lines(earlier[:1], "final", False) == [
        "earlier runs on the gold: 1",
        "  run 1 b: x",
    ]


def test_the_report_quotes_the_last_final_row() -> None:
    rows = [
        {"run": "1", "role": "final", "revision": "old"},
        {"run": "2", "role": "other", "revision": "x"},
        {"run": "3", "role": "final", "revision": "new"},
    ]
    last = SCORE.last_final_run(rows)
    assert last is not None and last["revision"] == "new"
    assert SCORE.last_final_run(rows[1:2]) is None


def _pin_is_the_scored_model(log: Path, config: Path) -> bool | None:
    """Whether the sentiment pin is a model the gold scored and the pin rule allows (D35): the model
    and revision of the last ``final`` row **when its paired gain is ``better``**, or those of the
    last ``baseline`` row (a candidate that was rejected leaves the baseline pinned). ``None`` when
    the log has no ``final`` row yet (nothing is being pinned)."""
    with log.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    final = SCORE.last_final_run(rows)
    if final is None:
        return None
    baseline = SCORE.last_baseline_run(rows)
    pinned = json.loads(config.read_text(encoding="utf-8"))["sentiment"]
    pin = (pinned["model"], pinned["revision"])
    if final["gain_verdict"] == "better" and (final["model"], final["revision"]) == pin:
        return True
    return baseline is not None and (baseline["model"], baseline["revision"]) == pin


def test_the_pinned_sentiment_model_is_the_one_scored_as_final_and_better() -> None:
    """The figure that counts (D35) is the last ``final`` row; the model the cut uses is the one
    pinned in ``config/pipeline_models.json``. The pin may be the candidate only if its paired gain
    over the baseline is ``better``; otherwise the baseline stays. The final run is of the
    *published* model (a Hub id and a revision), not of a local checkpoint."""
    root = Path(__file__).resolve().parents[2]
    matches = _pin_is_the_scored_model(
        root / "gold" / "sentiment_gold_scoring_log.csv", root / "config" / "pipeline_models.json"
    )
    if matches is None:
        pytest.skip("no final run on the gold yet (T-148c4 scores the published model as final)")
    assert matches, "pin the candidate only if its gain is better, else keep the baseline pinned"


def test_the_pin_rule_has_its_four_cases(tmp_path: Path) -> None:
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps({"sentiment": {"model": "org/base", "revision": "b" * 40}}), encoding="utf-8"
    )
    log = tmp_path / "log.csv"
    fields = ["run", "role", "model", "revision", "gain_verdict"]

    def write(rows: list[dict[str, str]]) -> None:
        with log.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows({f: r.get(f, "") for f in fields} for r in rows)

    base = {"run": "1", "role": "baseline", "model": "org/base", "revision": "b" * 40}
    cand = {"run": "2", "role": "final", "model": "org/cand", "revision": "c" * 40}
    write([base])
    assert _pin_is_the_scored_model(log, config) is None  # a baseline alone: nothing to pin
    # the baseline stays pinned and the candidate is rejected: passes
    write([base, {**cand, "gain_verdict": "not better"}])
    assert _pin_is_the_scored_model(log, config) is True
    # the candidate is pinned and better: passes
    config.write_text(
        json.dumps({"sentiment": {"model": "org/cand", "revision": "c" * 40}}), encoding="utf-8"
    )
    write([base, {**cand, "gain_verdict": "better"}])
    assert _pin_is_the_scored_model(log, config) is True
    # the candidate is pinned and not better, or not decided, or undefined: fails
    for verdict in ("not better", "not decided", "undefined", ""):
        write([base, {**cand, "gain_verdict": verdict}])
        assert _pin_is_the_scored_model(log, config) is False
    # a pin that is neither: fails
    config.write_text(
        json.dumps({"sentiment": {"model": "org/other", "revision": "d" * 40}}), encoding="utf-8"
    )
    write([base, {**cand, "gain_verdict": "better"}])
    assert _pin_is_the_scored_model(log, config) is False
    # a candidate that was better, then redone as a worse final row: the last final row counts
    config.write_text(
        json.dumps({"sentiment": {"model": "org/cand", "revision": "c" * 40}}), encoding="utf-8"
    )
    write(
        [
            base,
            {**cand, "gain_verdict": "better"},
            {**cand, "run": "3", "gain_verdict": "not better"},
        ]
    )
    assert _pin_is_the_scored_model(log, config) is False


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


def test_a_hub_model_without_a_revision_is_refused_and_a_local_directory_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="needs --revision"):
        _run(tmp_path, monkeypatch, "neutral", model="gamug/some-model", revision="")
    assert not (tmp_path / "gold" / "log.csv").exists()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    log = _run(tmp_path, monkeypatch, "neutral", model=str(checkpoint), revision="")  # no revision
    assert log.exists()


def test_the_row_carries_the_code_the_periods_and_every_judgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _run(tmp_path, monkeypatch, _marks_as_answers(), extra=("--role", "baseline"))
    with log.open(encoding="utf-8", newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["role"] == "baseline"
    assert len(row["code_revision"]) == 12 and row["code_revision"] == SCORE.scoring_code_revision()
    assert (row["texts_2022_on"], row["texts_before_2022"]) == ("90", "90")  # half of each, _gold
    assert float(row["agreement_2022_on"]) == float(row["agreement_before_2022"]) == 1.0
    assert row["negative_precision_judgement"] in {"pass", "fail", "not decided", "undefined"}
    assert row["negative_recall_judgement"] in {"pass", "fail", "not decided"}


def test_a_second_baseline_or_final_run_needs_redo_role_and_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = _marks_as_answers()
    _run(tmp_path, monkeypatch, _stored_answers(), extra=("--role", "baseline"))
    _run(tmp_path, monkeypatch, answers, extra=("--role", "final"))
    with pytest.raises(SystemExit, match="already has a final run"):
        _run(tmp_path, monkeypatch, answers, extra=("--role", "final"))
    _run(tmp_path, monkeypatch, answers)  # another role is not limited
    with pytest.raises(SystemExit, match="already has a baseline run"):
        _run(tmp_path, monkeypatch, _stored_answers(), extra=("--role", "baseline"))
    with pytest.raises(SystemExit, match="--redo-role needs --redo-reason"):
        _run(tmp_path, monkeypatch, answers, extra=("--role", "final", "--redo-role"))
    with pytest.raises(SystemExit, match="only for a --redo-role run"):
        _run(tmp_path, monkeypatch, answers, extra=("--role", "other", "--redo-reason", "why"))
    log = _run(
        tmp_path,
        monkeypatch,
        answers,
        extra=("--role", "final", "--redo-role", "--redo-reason", "the GPU ran out of memory"),
    )
    with log.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["role"] for r in rows] == ["baseline", "final", "other", "final"]
    assert [r["redo_reason"] for r in rows] == ["", "", "", "the GPU ran out of memory"]


def test_a_model_that_never_says_negative_has_an_undefined_precision_in_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _run(tmp_path, monkeypatch, "neutral")
    with log.open(encoding="utf-8", newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["negative_precision_judgement"] == "undefined"  # nothing predicted: no figure
    assert row["negative_recall_judgement"] == "fail"  # and none of the negatives is found
    assert row["negative_precision_point"] == "nan"


def test_a_hub_revision_must_be_a_commit_not_a_branch_or_a_tag() -> None:
    for bad in ("main", "v1.0", "abc123", COMMIT[:39], COMMIT.upper()):
        with pytest.raises(gold_metrics.MarkedFileError, match="not a commit"):
            SCORE.require_pinned("gamug/model", bad)
    SCORE.require_pinned("gamug/model", COMMIT)  # a commit passes
    SCORE.require_pinned(".", "")  # a local directory needs none


def test_the_texts_over_128_tokens_are_scored_apart_from_the_rest() -> None:
    rows = [
        {"predicted": "neutral", "gold": "neutral", "text_hash": "a"},
        {"predicted": "neutral", "gold": "negative", "text_hash": "b"},
        {"predicted": "negative", "gold": "negative", "text_hash": "c"},
    ]
    tokens = {"a": 50, "b": 300, "c": 128}  # 128 is the training length: not over
    by = SCORE.length_figures(rows, tokens)
    assert by["over 128"][0] == 1 and by["over 128"][1].point == 0.0
    assert by["up to 128"][0] == 2 and by["up to 128"][1].point == 1.0
    assert "over 128 tokens: 1 texts, 0.0%" in "\n".join(SCORE.length_lines(rows, tokens))


def _gain_rows(candidate: str, baseline: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = [
        {
            "stratum": "checked/positive",
            "gold": ("positive", "negative", "neutral")[i % 3],
            "predicted": candidate,
            "baseline": baseline,
            "text_hash": str(i),
        }
        for i in range(60)
    ]
    return rows, {"checked/positive": 600}


def test_the_gain_of_a_final_run_is_over_the_last_baseline_row_and_has_a_verdict() -> None:
    rows, populations = _gain_rows("positive", "positive")
    for r in rows:  # the candidate is right on every row; the baseline on a third
        r["predicted"] = r["gold"]
        r["baseline"] = "positive"
    again = gold_metrics.sentiment_report(
        [{**r, "predicted": r["baseline"]} for r in rows], populations
    ).agreement.point
    baseline = {"run": "1", "agreement_point": f"{again:.4f}"}
    gain = SCORE.gain_columns(rows, populations, baseline)
    assert gain["gain_verdict"] == "better"
    assert gain["gain_negative_recall_verdict"] == "better"
    assert set(gain) == {
        "gain_agreement",
        "gain_verdict",
        "gain_negative_precision",
        "gain_negative_precision_verdict",
        "gain_negative_recall",
        "gain_negative_recall_verdict",
    }


def test_a_gain_whose_interval_spans_zero_is_not_decided() -> None:
    rows, populations = _gain_rows("positive", "positive")
    for i, r in enumerate(rows):  # the candidate and the baseline each right on half, differently
        r["predicted"] = r["gold"] if i % 2 else "neutral"
        r["baseline"] = r["gold"] if i % 2 == 0 else "neutral"
    again = gold_metrics.sentiment_report(
        [{**r, "predicted": r["baseline"]} for r in rows], populations
    ).agreement.point
    gain = SCORE.gain_columns(rows, populations, {"run": "1", "agreement_point": f"{again:.4f}"})
    assert gain["gain_verdict"] == "not decided"


def test_a_gain_is_refused_when_the_baseline_row_is_not_the_stored_predictions() -> None:
    rows, populations = _gain_rows("positive", "positive")
    with pytest.raises(gold_metrics.MarkedFileError, match="would compare different things"):
        SCORE.gain_columns(rows, populations, {"run": "1", "agreement_point": "0.9000"})


def test_a_final_run_needs_a_baseline_row_and_other_roles_have_no_gain() -> None:
    assert SCORE.require_baseline("other", []) is None
    row = {"run": "1", "role": "baseline"}
    assert SCORE.require_baseline("final", [row]) == row
    assert SCORE.require_baseline("baseline", [row]) is None
    with pytest.raises(gold_metrics.MarkedFileError, match="the log has none"):
        SCORE.require_baseline("final", [{"run": "1", "role": "other"}])


def test_main_logs_the_gain_of_the_final_run_and_the_length_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(tmp_path, monkeypatch, _stored_answers(), extra=("--role", "baseline"))
    log = _run(tmp_path, monkeypatch, _marks_as_answers(), extra=("--role", "final"))
    out = capsys.readouterr().out
    assert "gain over the baseline (run 1), paired, same strata:" in out
    assert "agreement by length" in out
    with log.open(encoding="utf-8", newline="") as fh:
        baseline, final = list(csv.DictReader(fh))
    assert baseline["gain_verdict"] == "" and baseline["texts_over_128"] == "45"
    assert final["gain_verdict"] == "better"
    assert final["gain_negative_recall_verdict"] in {"better", "not decided", "not better"}
    assert final["texts_over_128"] == "45" and final["texts_up_to_128"] == "135"
    assert final["input_format"] == "text" and final["company_less_model"] == ""


def test_a_final_run_with_no_baseline_row_stops_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="the log has none"):
        _run(tmp_path, monkeypatch, _marks_as_answers(), extra=("--role", "final"))
    assert not (tmp_path / "gold" / "log.csv").exists()


def test_the_committed_baseline_row_keeps_its_values_and_has_empty_new_columns() -> None:
    root = Path(__file__).resolve().parents[2]
    with (root / "gold" / "sentiment_gold_scoring_log.csv").open(
        encoding="utf-8-sig", newline=""
    ) as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(retag.SENTIMENT_SCORING_LOG_COLUMNS)
    (baseline,) = rows
    assert (baseline["role"], baseline["agreement_point"]) == ("baseline", "0.6040")
    new = retag.SENTIMENT_SCORING_LOG_COLUMNS[
        retag.SENTIMENT_SCORING_LOG_COLUMNS.index("texts") + 1 :
    ]
    assert all(baseline[c] == "" for c in new)


def test_the_company_of_a_row_is_its_tickers_name_and_empty_without_a_ticker() -> None:
    rows = _gold()
    rows[0]["ticker"] = "AAA"
    items = SCORE.texts_of_the_gold(rows, _ids(rows), {"AAA": "Alpha Inc."})
    assert items[0]["company"] == "Alpha Inc." and items[1]["company"] == ""
    with pytest.raises(gold_metrics.MarkedFileError, match="has no company name"):
        SCORE.texts_of_the_gold(rows, _ids(rows), {"BBB": "Beta"})
    assert "company" not in SCORE.texts_of_the_gold(rows, _ids(rows))[0]  # a text-only run


def test_a_pair_run_sends_texts_with_a_company_to_the_pair_model_and_the_rest_to_the_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, bool, str]] = []

    def predict(
        texts: list[dict[str, str]], cfg: dict[str, str] | None = None, pair: bool = False
    ) -> list[dict[str, str]]:
        seen.append((len(texts), pair, (cfg or {})["model"]))
        return [{**t, "predicted": "positive" if pair else "negative"} for t in texts]

    monkeypatch.setattr(SCORE.gold_draw, "predict", predict)
    items = [
        {"key": "a", "text": "x", "company": "Alpha"},
        {"key": "b", "text": "y", "company": ""},
        {"key": "c", "text": "z", "company": "Gamma"},
    ]
    got = SCORE.score(
        items,
        {"model": "cand"},
        pair=True,
        company_less_cfg={"model": "pinned"},
    )
    assert got == {"a": "positive", "b": "negative", "c": "positive"}
    assert seen == [(2, True, "cand"), (1, False, "pinned")]


def test_main_scores_a_pair_model_and_logs_the_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, bool, str]] = []
    log = _run(
        tmp_path,
        monkeypatch,
        _marks_as_answers(),
        extra=("--pair", "--role", "other"),
        tickers=143,
        calls=calls,
    )
    assert calls == [(143, True, "m"), (37, False, "org/pinned")]  # 143 with a company, 37 without
    with log.open(encoding="utf-8", newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["input_format"] == "pair"
    assert (row["company_less_model"], row["company_less_revision"]) == ("org/pinned", "9" * 40)


def test_the_company_less_model_is_pinned_like_any_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="not a commit"):
        _run(
            tmp_path,
            monkeypatch,
            _marks_as_answers(),
            extra=(
                "--pair",
                "--company-less-model",
                "org/text-only",
                "--company-less-revision",
                "main",
            ),
            tickers=143,
        )
    assert not (tmp_path / "gold" / "log.csv").exists()
