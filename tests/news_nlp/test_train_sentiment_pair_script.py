"""The command that gives the pair recipe its data (T-148c3): the parts that run without a GPU."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import sentiment_pair_train as spt
from fti import TrainedArtifact
from news_nlp import triage

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CMD = _script("train_sentiment_pair_2026_11_09")


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as raw:
        raw.execute(
            "CREATE TABLE articles (id INTEGER, ticker TEXT, company TEXT, body_text TEXT, fetch_status TEXT, title TEXT)"
        )
        raw.executemany(
            "INSERT INTO articles VALUES (?,?,?,?,?,?)",
            [
                (1, "AAA", "Acme Corp", "Acme  rose today", "ok", "Acme raises its yearly outlook"),
                (2, "BBB", "Bravo Inc", "Bravo fell", "quarantined", "Bravo"),  # a short headline
                (3, "CCC", "Charlie", "", "ok", None),  # no body
                (4, "", "Delta", "Delta news", "ok", None),  # no ticker
                (5, "EEE", None, "Echo news", "ok", None),  # no company
            ],
        )
    return path


def test_the_replay_gets_the_body_ticker_and_company_of_each_article_that_has_all_three(
    tmp_path: Path,
) -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ? AS source", (str(_source(tmp_path)),))
    infos = CMD.article_infos(conn, {1, 2, 3, 4, 5, 99})
    assert set(infos) == {1, 2}
    assert infos[1] == spt.ArticleInfo(
        triage.body_hash("Acme rose today"),
        "AAA",
        "Acme Corp",
        CMD.headline_key("Acme raises its yearly outlook"),
    )
    assert infos[1].title_key and infos[2].title_key == ""  # too short to mean anything


def test_the_source_is_read_as_an_immutable_snapshot_when_it_cannot_be_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse() -> Any:
        raise RuntimeError("Could not ATTACH the source: disk I/O error.")

    monkeypatch.setattr(CMD.db, "connect_pipeline", refuse)
    monkeypatch.setattr(CMD.db.env, "source_db_path", lambda *_a: _source(tmp_path))
    conn = CMD.connect_source()
    assert conn.execute("SELECT COUNT(*) FROM source.articles").fetchone()[0] == 5
    assert "immutable snapshot" in capsys.readouterr().err
    assert "immutable snapshot" in CMD.SOURCE_MODE["how"]  # written into the results file
    with pytest.raises(sqlite3.OperationalError):  # read-only
        conn.execute("INSERT INTO source.articles VALUES (9, 'X', 'X', 'X', 'ok', 'X')")


def test_without_a_source_the_attach_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> Any:
        raise RuntimeError("no source")

    monkeypatch.setattr(CMD.db, "connect_pipeline", refuse)
    monkeypatch.setattr(CMD.db.env, "source_db_path", lambda *_a: None)
    with pytest.raises(RuntimeError, match="no source"):
        CMD.connect_source()


def test_the_validation_agreement_is_the_share_of_texts_read_as_the_owner_marked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = [
        spt.Example("a", "Acme", "t1", 0, "summary"),
        spt.Example("b", "Bravo", "t2", 1, "summary"),
        spt.Example("c", "Acme", "t3", 2, "summary"),
        spt.Example("d", "Bravo", "t4", 0, "summary"),
    ]
    seen: list[Any] = []

    def predict(items: list[dict[str, str]], cfg: Any, pair: bool = False) -> list[dict[str, str]]:
        seen.append((cfg, pair, [i["company"] for i in items]))
        answers = {"a": "positive", "b": "negative", "c": "positive", "d": "positive"}
        return [{**i, "predicted": answers[i["key"]]} for i in items]

    monkeypatch.setattr(CMD.gold_draw, "predict", predict)
    score = CMD.validation_scorer(validation)
    assert score("models/x") == 0.75
    assert seen == [({"model": "models/x"}, True, ["Acme", "Bravo", "Acme", "Bravo"])]  # the pair

    monkeypatch.setattr(CMD.gold_draw, "predict", lambda items, cfg, pair=False: items[:2])
    with pytest.raises(spt.MarkedFileError, match="no chunk"):
        CMD.validation_scorer(validation)("models/x")


def test_the_results_are_written_whole_and_only_the_grid_rows_choose_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CMD, "OUT", tmp_path / "out")
    assert CMD.load_results() == ({}, [])
    rows = [
        {"name": "n1", "tag": "grid", "validation_agreement": 0.7, "settings": {"epochs": 2, "learning_rate": 1e-5}},
        {"name": "n2", "tag": "grid", "validation_agreement": 0.8, "settings": {"epochs": 2, "learning_rate": 1e-5}},
        {"name": "t", "tag": "timing", "validation_agreement": 0.1, "settings": {"epochs": 4, "learning_rate": 2e-5}},
    ]  # fmt: skip
    CMD.save_results(rows, {"fingerprint": "f1", "source": "attached read-only"})
    assert CMD.load_results() == ({"fingerprint": "f1", "source": "attached read-only"}, rows)
    assert not list((tmp_path / "out").glob("*.tmp"))
    assert CMD.agreements_by_setting(rows) == {(2, 1e-5): [0.7, 0.8]}


class _Trainer:
    """Stands for the FTI trainer: writes a model directory and gives back what the real one would."""

    def __init__(self, scorer: Any, log: Any = print) -> None:
        self._scorer = scorer

    def train(self, config: spt.SentimentPairTrainConfig) -> TrainedArtifact:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(config.output_dir) / "config.json").write_text("{}")
        return TrainedArtifact(
            config.output_dir,
            {
                "validation_agreement": self._scorer(config.output_dir),
                "epoch_losses": [1.0],
                "train_seconds": 1.0,
                "settings": {"epochs": config.settings.epochs, "learning_rate": config.settings.learning_rate, "seed": config.settings.seed},
                "n_train": len(config.train),
            },
        )  # fmt: skip


def _patch_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    logged: list[str] = []
    monkeypatch.setattr(CMD, "OUT", tmp_path / "out")
    monkeypatch.setattr(CMD, "MODELS", tmp_path / "models")
    monkeypatch.setattr(CMD.spt, "SentimentPairTrainer", _Trainer)
    monkeypatch.setattr(CMD, "base_model", lambda: ("base", "rev"))
    monkeypatch.setattr(
        CMD, "log_mlflow", lambda row, settings, n, fp="": logged.append(f"{row['name']}|{fp}")
    )
    return logged


def test_a_run_deletes_its_model_unless_it_is_kept_and_logs_to_mlflow_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged = _patch_run(monkeypatch, tmp_path)
    settings = spt.RunSettings(2, 1e-5, 1)
    row = CMD.one_run(settings, [], lambda _d: 0.5)
    assert row["validation_agreement"] == 0.5 and row["tag"] == "grid" and logged == []
    assert not (tmp_path / "models" / "_run" / settings.name).exists()  # deleted after scoring
    kept = CMD.one_run(
        settings,
        [],
        lambda _d: 0.6,
        keep=tmp_path / "keep",
        mlflow_on=True,
        tag="final",
        fingerprint="fp1",
    )
    assert (
        (tmp_path / "keep" / "config.json").exists()
        and logged == [f"{settings.name}|fp1"]
        and kept["tag"] == "final"
    )


def test_the_grid_runs_what_is_not_done_and_saves_after_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_run(monkeypatch, tmp_path)
    first = spt.grid_settings()[0]
    CMD.save_results(
        [{**CMD.one_run(first, [], lambda _d: 0.7), "tag": "grid"}], {"fingerprint": "f"}
    )
    saved: list[int] = []
    real_save = CMD.save_results

    def counting_save(rows: list[dict[str, Any]], header: dict[str, Any]) -> None:
        saved.append(len(rows))
        real_save(rows, header)

    monkeypatch.setattr(CMD, "save_results", counting_save)
    CMD.run_grid([], lambda _d: 0.8, False, "f", 83)
    out = capsys.readouterr().out
    assert "1 runs already done, 26 to run" in out
    assert "/83 texts" in out and "sampling noise of 83 texts" in out
    assert saved == list(range(2, 28))  # one save per run
    header, rows = CMD.load_results()
    assert len(rows) == 27 and len({r["name"] for r in rows}) == 27 and header["fingerprint"] == "f"


def test_rows_of_other_data_are_not_resumed_unless_asked_and_then_set_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    row = {**CMD.one_run(spt.grid_settings()[0], [], lambda _d: 0.7), "tag": "grid"}
    CMD.save_results([row], {"fingerprint": "old-data"})
    with pytest.raises(SystemExit, match=r"other data or settings.*--fresh"):
        CMD.run_grid([], lambda _d: 0.8, False, "new-data")
    with pytest.raises(SystemExit, match=r"other data or settings"):
        CMD.run_final([], lambda _d: 0.8, False, "new-data")
    CMD.run_grid([], lambda _d: 0.8, False, "new-data", fresh=True)
    assert (tmp_path / "out" / "grid_results.previous.json").exists()
    header, rows = CMD.load_results()
    assert (
        header["fingerprint"] == "new-data"
        and len(rows) == 27
        and rows[0]["validation_agreement"] == 0.8
    )
    old = tmp_path / "out" / "grid_results.json"
    old.write_text("[]")  # the first format: no fingerprint, and no rows to lose
    CMD.run_grid([], lambda _d: 0.8, False, "new-data")


def test_the_final_run_needs_the_whole_grid_and_is_made_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_run(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="grid is not complete"):
        CMD.run_final([], lambda _d: 0.8, False, "f")
    CMD.save_results(
        [{**CMD.one_run(s, [], lambda _d: 0.8), "tag": "grid"} for s in spt.grid_settings()],
        {"fingerprint": "f"},
    )
    CMD.run_final([], lambda _d: 0.9, False, "f", 83)
    out = capsys.readouterr().out
    assert (
        "chosen: 4 epochs, lr 2e-05" in out
        and "the three seeds of the same setting: mean 0.800" in out
    )
    assert "not published, not scored on the gold" in out
    last = CMD.load_results()[-1]
    last = CMD.load_results()[1][-1]
    assert last["tag"] == "final" and last["settings"]["seed"] == spt.FINAL_SEED
    assert last["fingerprint"] == "f"  # T-148c4 can check what the published model was trained on
    assert (tmp_path / "models" / "final" / "config.json").exists()
    assert not (tmp_path / "models" / "final.partial").exists()
    with pytest.raises(SystemExit, match=r"a final run is made once.*exists.*a final row"):
        CMD.run_final([], lambda _d: 0.9, False, "f")  # a final run is made once


def test_a_final_run_that_fails_leaves_no_model_without_a_row_and_can_be_made_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    CMD.save_results(
        [{**CMD.one_run(s, [], lambda _d: 0.8), "tag": "grid"} for s in spt.grid_settings()],
        {"fingerprint": "f"},
    )

    def broken(_d: str) -> float:
        raise RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        CMD.run_final([], broken, False, "f")
    assert not (
        tmp_path / "models" / "final"
    ).exists()  # only the partial one, never taken for the model
    CMD.run_final([], lambda _d: 0.9, False, "f")  # the partial directory is replaced
    assert (tmp_path / "models" / "final" / "config.json").exists()
    assert CMD.load_results()[1][-1]["tag"] == "final"
    # a final row without its model directory is refused and says which of the two is missing
    shutil.rmtree(tmp_path / "models" / "final")
    with pytest.raises(SystemExit, match=r"is missing and the results file has a final row"):
        CMD.run_final([], lambda _d: 0.9, False, "f")


def test_the_baseline_scores_the_pinned_model_with_the_pair_and_without(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_run(monkeypatch, tmp_path)
    validation = [spt.Example(k, "Acme", "t", 0, "summary") for k in "abcd"]
    seen: list[Any] = []

    def predict(items: list[dict[str, str]], cfg: Any, pair: bool = False) -> list[dict[str, str]]:
        seen.append((cfg, pair))
        hits = 3 if pair else 1  # the pair reads three of the four as marked
        return [
            {**i, "predicted": "positive" if n < hits else "negative"} for n, i in enumerate(items)
        ]

    monkeypatch.setattr(CMD.gold_draw, "predict", predict)
    result = CMD.run_baseline(validation)
    assert seen == [
        ({"model": "base", "revision": "rev"}, True),
        ({"model": "base", "revision": "rev"}, False),
    ]
    assert (result["pair"], result["text_only"], result["n"]) == (3, 1, 4)
    record = json.loads((tmp_path / "out" / "baseline.json").read_text())
    assert record["pair"] == 3 and record["model"] == "base@rev" and len(record["code"]) == 12
    assert record["keys"] == CMD.keys_hash(list("abcd"))
    assert "as the pair 3/4 = 0.750; text only 1/4 = 0.250" in capsys.readouterr().out


def test_the_stage_rule_the_replay_uses_is_still_there_under_its_name() -> None:
    from sentiment_stage import _text_mentions_subject  # noqa: PLC0415

    assert CMD._text_mentions_subject is _text_mentions_subject


def test_the_replay_seed_and_the_timing_setting_are_fixed_in_the_command() -> None:
    assert CMD.REPLAY_SEED == 20261109 and spt.RunSettings(4, 2e-5, 1) == CMD.TIMING_SETTING
    models = SCRIPTS.parent / "config" / "pipeline_models.json"
    assert json.loads(models.read_text(encoding="utf-8"))["sentiment"]["revision"]


def test_a_baseline_that_leaves_a_text_unread_stops_and_the_predictions_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    validation = [spt.Example(k, "Acme", "t", i % 3, "summary") for i, k in enumerate("abcd")]
    monkeypatch.setattr(
        CMD.gold_draw, "predict", lambda items, cfg, pair=False: [{**i, "predicted": "neutral"} for i in items][:3]
    )  # fmt: skip
    with pytest.raises(spt.MarkedFileError, match="1 validation texts have no chunk"):
        CMD.run_baseline(validation)
    monkeypatch.setattr(
        CMD.gold_draw, "predict", lambda items, cfg, pair=False: [{**i, "predicted": "neutral"} for i in items]
    )  # fmt: skip
    CMD.run_baseline(validation)
    kept = CMD.read_predictions("baseline_pair")
    assert kept == [
        {"key": "a", "expected": "positive", "predicted": "neutral"},
        {"key": "b", "expected": "negative", "predicted": "neutral"},
        {"key": "c", "expected": "neutral", "predicted": "neutral"},
        {"key": "d", "expected": "positive", "predicted": "neutral"},
    ]


def test_a_run_keeps_what_each_validation_text_was_read_as_when_the_scorer_has_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    validation = [spt.Example(k, "Acme", "t", 0, "summary") for k in "ab"]
    scorer = CMD.ValidationScorer(validation)
    monkeypatch.setattr(
        CMD.gold_draw, "predict", lambda items, cfg, pair=False: [{**i, "predicted": "negative"} for i in items]
    )  # fmt: skip
    settings = spt.RunSettings(2, 1e-5, 1)
    CMD.one_run(settings, [], scorer)
    assert CMD.read_predictions(settings.name) == [
        {"key": "a", "expected": "positive", "predicted": "negative"},
        {"key": "b", "expected": "positive", "predicted": "negative"},
    ]
    CMD.one_run(
        spt.RunSettings(2, 1e-5, 2), [], lambda _d: 0.5
    )  # a plain function: nothing to keep
    assert CMD.read_predictions("e2_lr1e-05_s2") is None


def _report_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str = "c1"
) -> tuple[list[str], list[str]]:
    """Predictions of the final run and both baselines for 8 texts, the marks they come from (two strata with
    different pools), baseline.json and the final row."""
    _patch_run(monkeypatch, tmp_path)
    monkeypatch.setattr(CMD, "code_revision", lambda: "c1")
    keys = [f"k{i}" for i in range(8)]
    expected = [
        "negative",
        "negative",
        "positive",
        "positive",
        "neutral",
        "neutral",
        "negative",
        "positive",
    ]
    marks = tmp_path / "marks.csv"
    marks.write_text(
        "key,slice,predicted,pool_class_size,sentiment\n"
        + "".join(
            f"{k},validation,{'negative' if i < 4 else 'positive'},{100 if i < 4 else 900},{e}\n"
            for i, (k, e) in enumerate(zip(keys, expected, strict=True))
        )
    )
    monkeypatch.setattr(CMD, "MARKS", marks)

    def predictions(predicted: list[str]) -> list[dict[str, str]]:
        return [
            {"key": k, "expected": e, "predicted": p}
            for k, e, p in zip(keys, expected, predicted, strict=True)
        ]

    CMD.write_predictions(
        "baseline_text_only", predictions(["negative"] * 4 + ["positive"] * 4)
    )  # the strata the texts were drawn from: the pinned model's own prediction
    CMD.write_predictions("baseline_pair", predictions(["positive"] * 8))
    CMD.write_predictions("e2_lr3e-05_s4", predictions([*expected[:6], "neutral", "neutral"]))
    CMD.save_results(
        [{"name": "e2_lr3e-05_s4", "tag": "final", "code": code}], {"fingerprint": "f"}
    )
    (tmp_path / "out" / "baseline.json").write_text(
        json.dumps({"model": "base@rev", "code": "c1", "keys": CMD.keys_hash(keys), "n": 8})
    )
    return keys, expected


def test_the_report_gives_the_weighted_figures_and_gain_first_and_the_raw_description_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="needs --baseline and --final"):
        _patch_run(monkeypatch, tmp_path)
        CMD.run_report()
    capsys.readouterr()
    CMD.run_report(required=False)
    assert "run --baseline, then --report" in capsys.readouterr().out
    _report_fixture(tmp_path, monkeypatch)
    CMD.run_report()
    out = capsys.readouterr().out
    assert out.index("WEIGHTED") < out.index("RAW on the 8 texts")
    assert "{'negative': 100, 'positive': 900}" in out and "7 company-less rows" in out
    assert "final: agreement" in out and "pinned_text: agreement" in out
    assert (
        "paired weighted gain of the final run over the pinned model as text (D35's pin rule)"
        in out
    )
    assert "negative precision" in out and "-> " in out  # the gold's verdict of the weighted gain
    assert "D32's bar on the final run" in out
    assert "agreement not decided; negative precision pass" in out  # 8 texts: wide intervals
    assert "the pin rule on these figures" in out
    assert "(validation; the rule is applied once, to the gold)" in out
    assert "agrees with its stored draw prediction on 8 of 8 texts" in out
    assert "final (e2_lr3e-05_s4), RAW on the 8 texts: agreement 6/8" in out
    assert (
        "final against pinned text_only, RAW: both right 2, only the final 4, only the pinned 1, neither 1"
        in out
    )
    assert "exact McNemar p = 0.3750" in out and "no verdict" in out
    assert (
        "better" not in out.split("RAW is the description")[1]
    )  # no verdict word for the raw gain


def test_the_report_stops_if_the_baseline_is_not_of_this_model_these_texts_or_this_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    keys, _ = _report_fixture(tmp_path, monkeypatch)
    baseline = tmp_path / "out" / "baseline.json"
    good = json.loads(baseline.read_text())
    for change, message in (
        ({"model": "other@rev"}, "the pinned model is now base@rev"),
        ({"keys": "x"}, "other validation texts"),
        ({"code": "c0"}, "the code differs: baseline c0, final run c1, now c1"),
    ):
        baseline.write_text(json.dumps({**good, **change}))
        with pytest.raises(SystemExit, match=message):
            CMD.run_report()
    baseline.write_text(json.dumps(good))
    monkeypatch.setattr(CMD, "code_revision", lambda: "c2")  # the code moved on since the runs
    with pytest.raises(SystemExit, match="the code differs: baseline c1, final run c1, now c2"):
        CMD.run_report()
    capsys.readouterr()
    CMD.run_report(required=False)  # after --final: said, not fatal
    assert "the report is not made: the code differs" in capsys.readouterr().out
    baseline.unlink()
    with pytest.raises(SystemExit, match=r"no baseline\.json"):
        CMD.run_report()
    assert keys


def test_resuming_with_the_source_read_another_way_says_so_and_keeps_the_first_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    row = {**CMD.one_run(spt.grid_settings()[0], [], lambda _d: 0.7), "tag": "grid"}
    monkeypatch.setitem(CMD.SOURCE_MODE, "how", "attached read-only")
    CMD.save_results([row], CMD.open_results("f")[0])
    assert CMD.load_results()[0]["source"] == "attached read-only"
    monkeypatch.setitem(CMD.SOURCE_MODE, "how", "immutable snapshot")
    header, _ = CMD.open_results("f")
    assert header["source"] == "attached read-only, then immutable snapshot"
    CMD.save_results([row], header)
    assert CMD.open_results("f")[0]["source"] == header["source"]  # not appended again
    assert CMD.open_results("g", fresh=True)[0]["source"] == "immutable snapshot"  # a new file


def test_the_code_revision_of_the_fingerprint_is_twelve_hex_digits_of_the_modules_that_make_a_number() -> (
    None
):
    assert len(CMD.code_revision()) == 12 and int(CMD.code_revision(), 16) >= 0


def _argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["train_sentiment_pair", *args])
    monkeypatch.setattr(CMD.db.env, "load_env_file", lambda *_a: None)


def test_fresh_is_only_for_the_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("--final", "--time-one", "--baseline", "--report"):
        _argv(monkeypatch, mode, "--fresh")
        with pytest.raises(SystemExit) as stopped:
            CMD.main()
        assert stopped.value.code == 2  # argparse's own refusal


def test_the_baseline_and_the_report_never_open_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    def opened() -> Any:
        raise AssertionError("the SOURCE was opened")

    called: list[str] = []
    monkeypatch.setattr(CMD, "connect_source", opened)
    monkeypatch.setattr(CMD, "read_csv", lambda _p: [])
    monkeypatch.setattr(CMD, "run_baseline", lambda validation: called.append("baseline"))  # type: ignore[arg-type,return-value]
    monkeypatch.setattr(CMD, "run_report", lambda: called.append("report"))
    for mode in ("--baseline", "--report"):
        _argv(monkeypatch, mode)
        CMD.main()
    assert called == ["baseline", "report"]


def test_a_baseline_that_is_not_the_reading_the_strata_came_from_withholds_the_weighted_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _report_fixture(tmp_path, monkeypatch)
    keys = [f"k{i}" for i in range(8)]
    other = [{"key": k, "expected": "positive", "predicted": "neutral"} for k in keys]
    CMD.write_predictions("baseline_text_only", other)  # not the stored draw prediction
    CMD.run_report()
    out = capsys.readouterr().out
    assert "agrees with its stored draw prediction on 0 of 8 texts" in out
    assert (
        "WARNING: the baseline is not the reading the strata were drawn from (8 texts differ)"
        in out
    )
    assert "paired weighted gain" not in out and "D32's bar" not in out
    assert "RAW on the 8 texts" in out  # the raw description still follows


def _grid_with_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, good_for: tuple[int, float]
) -> None:
    """Grid rows for two settings, three seeds each, with predictions: the setting *good_for* reads all 8 texts
    as the owner marked, the other one reads none of them so."""
    keys, expected = _report_fixture(tmp_path, monkeypatch)
    rows = []
    for epochs, lr in ((2, 1e-5), (3, 2e-5)):
        for seed in (1, 2, 3):
            name = f"e{epochs}_lr{lr:g}_s{seed}"
            right = (epochs, lr) == good_for
            CMD.write_predictions(
                name,
                [
                    {"key": k, "expected": e, "predicted": e if right else "neutral"}
                    for k, e in zip(keys, expected, strict=True)
                ],
            )
            rows.append(
                {
                    "name": name,
                    "tag": "grid",
                    "validation_agreement": 1.0 if right else 0.25,
                    "settings": {"epochs": epochs, "learning_rate": lr, "seed": seed},
                }
            )
    final = {"name": "e2_lr3e-05_s4", "tag": "final", "code": "c1"}
    CMD.save_results([*rows, final], {"fingerprint": "f"})


def test_the_report_says_whether_the_weighted_agreement_would_choose_the_same_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _grid_with_predictions(tmp_path, monkeypatch, good_for=(2, 1e-5))
    CMD.run_report()
    out = capsys.readouterr().out
    assert (
        "weighted agreement by setting (mean of the seeds): 2 ep 1e-05: 1.000; 3 ep 2e-05: 0.450"
        in out
    )
    assert "the grid chose 2 epochs, lr 1e-05 (raw agreement)" in out and ": the same" in out
    # the raw agreements of the rows say one thing, the predictions on disk another: the ranking differs
    _grid_with_predictions(tmp_path, monkeypatch, good_for=(3, 2e-5))
    rows = CMD.load_results()[1]
    for row in rows:
        if row["tag"] == "grid":
            row["validation_agreement"] = 1.0 if row["settings"]["epochs"] == 2 else 0.25
    CMD.save_results(rows, {"fingerprint": "f"})
    capsys.readouterr()
    CMD.run_report()
    out = capsys.readouterr().out
    assert "the grid chose 2 epochs, lr 1e-05" in out
    assert (
        "would choose 3 epochs, lr 2e-05: DIFFERENT, a question for the owner (nothing is re-picked)"
        in out
    )
    (tmp_path / "out" / "predictions" / "e3_lr2e-05_s1.csv").unlink()
    CMD.run_report()
    assert "needs the predictions of e3_lr2e-05_s1" in capsys.readouterr().out
