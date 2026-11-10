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
from news_nlp import retag, triage

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


def test_the_results_are_written_whole_and_only_the_grid_rows_choose_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CMD, "OUT", tmp_path / "out")
    assert CMD.load_results() == ({}, [])
    rows = [
        {"name": "n1", "tag": "grid", "validation_score": 0.7, "settings": {"epochs": 2, "learning_rate": 1e-5}},
        {"name": "n2", "tag": "grid", "validation_score": 0.8, "settings": {"epochs": 2, "learning_rate": 1e-5}},
        {"name": "t", "tag": "timing", "validation_score": 0.1, "settings": {"epochs": 4, "learning_rate": 2e-5}},
    ]  # fmt: skip
    CMD.save_results(rows, {"fingerprint": "f1", "source": "attached read-only"})
    assert CMD.load_results() == ({"fingerprint": "f1", "source": "attached read-only"}, rows)
    assert not list((tmp_path / "out").glob("*.tmp"))
    assert CMD.scores_by_setting(rows) == {(2, 1e-5): [0.7, 0.8]}


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
                "validation_score": self._scorer(config.output_dir),
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
    assert row["validation_score"] == 0.5 and row["tag"] == "grid" and logged == []
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
    assert "mean validation score by setting" in out and "the validation slice has 83 texts" in out
    assert "the mean of the macro precision, recall and ROC OVR" in out
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
    assert (tmp_path / "out" / "grid_results.previous-old-data.json").exists()
    header, rows = CMD.load_results()
    assert (
        header["fingerprint"] == "new-data"
        and len(rows) == 27
        and rows[0]["validation_score"] == 0.8
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


def test_the_stage_rule_the_replay_uses_is_still_there_under_its_name() -> None:
    from sentiment_stage import _text_mentions_subject  # noqa: PLC0415

    assert CMD._text_mentions_subject is _text_mentions_subject


def test_the_replay_seed_and_the_timing_setting_are_fixed_in_the_command() -> None:
    assert CMD.REPLAY_SEED == 20261109 and spt.RunSettings(4, 2e-5, 1) == CMD.TIMING_SETTING
    models = SCRIPTS.parent / "config" / "pipeline_models.json"
    assert json.loads(models.read_text(encoding="utf-8"))["sentiment"]["revision"]


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


def test_models_is_only_for_the_all_data_model(monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("--grid", "--final", "--time-one", "--baseline", "--report"):
        _argv(monkeypatch, mode, "--models", "somewhere")
        with pytest.raises(SystemExit) as stopped:
            CMD.main()
        assert stopped.value.code == 2  # argparse's own refusal


def test_the_baseline_and_the_report_never_open_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    def opened() -> Any:
        raise AssertionError("the SOURCE was opened")

    called: list[str] = []
    monkeypatch.setattr(CMD, "connect_source", opened)
    monkeypatch.setattr(CMD, "read_csv", lambda _p: [])
    monkeypatch.setattr(CMD, "run_baseline", lambda validation, marks: called.append("baseline"))  # type: ignore[arg-type,return-value]
    monkeypatch.setattr(CMD, "run_report", lambda: called.append("report"))
    for mode in ("--baseline", "--report"):
        _argv(monkeypatch, mode)
        CMD.main()
    assert called == ["baseline", "report"]


def test_two_fresh_starts_keep_both_sets_of_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    row = {**CMD.one_run(spt.grid_settings()[0], [], lambda _d: 0.7), "tag": "grid"}
    CMD.save_results([row], {"fingerprint": "aaaa"})
    CMD.open_results("bbbb", fresh=True)
    CMD.save_results([{**row, "validation_score": 0.9}], {"fingerprint": "aaaa"})  # the same old
    CMD.open_results("cccc", fresh=True)
    kept = sorted(p.name for p in (tmp_path / "out").glob("grid_results.previous-*.json"))
    assert kept == ["grid_results.previous-aaaa-2.json", "grid_results.previous-aaaa.json"]
    # the first set-aside rows were not overwritten by the second
    first = json.loads((tmp_path / "out" / "grid_results.previous-aaaa.json").read_text())
    assert first["rows"][0]["validation_score"] == 0.7
    CMD.save_results([], {})
    (tmp_path / "out" / "grid_results.json").write_text("[]")  # the first format: nothing to lose
    assert CMD.open_results("dddd", fresh=True)[1] == []


# --- the metrics (owner, 2026-11-09; PLAN D37): precision, recall and ROC OVR, per class -----------

KEYS = [f"k{i}" for i in range(8)]
EXPECTED = [
    "negative",
    "negative",
    "positive",
    "positive",
    "neutral",
    "neutral",
    "negative",
    "positive",
]
STRATA = ["negative"] * 4 + [
    "positive"
] * 4  # the class the pinned model predicted when the texts were drawn


def _marks() -> list[dict[str, str]]:
    return [
        {
            "key": k,
            "slice": "validation",
            "predicted": STRATA[i],
            "pool_class_size": "100" if i < 4 else "900",
            "sentiment": EXPECTED[i],
        }
        for i, k in enumerate(KEYS)
    ]


def _validation() -> list[spt.Example]:
    return [
        spt.Example(k, "Acme", f"t{i}", spt.LABEL2ID[EXPECTED[i]], "summary")
        for i, k in enumerate(KEYS)
    ]


def _probs(label: str) -> dict[str, float]:
    return {c: 0.1 for c in spt.LABEL2ID} | {label: 0.8}


def _scored_items(items: list[dict[str, str]], predicted: list[str]) -> list[dict[str, Any]]:
    """What ``predict_probs`` gives back: each text with its label and class probabilities."""
    return [
        {**i, "predicted": p, "probs": _probs(p)} for i, p in zip(items, predicted, strict=True)
    ]


def _rows(predicted: list[str]) -> list[dict[str, str]]:
    """Predictions as the command keeps them (``key``, ``expected``, ``predicted``, ``p_<class>``)."""
    return [
        {
            "key": k,
            "expected": e,
            "predicted": p,
            **{f"p_{c}": repr(v) for c, v in _probs(p).items()},
        }
        for k, e, p in zip(KEYS, EXPECTED, predicted, strict=True)
    ]


def test_the_validation_score_is_the_mean_of_precision_recall_and_roc_ovr_of_the_three_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def predict_probs(
        items: list[dict[str, str]], cfg: Any, pair: bool = False
    ) -> list[dict[str, Any]]:
        seen.append((cfg, pair, [i["company"] for i in items]))
        return _scored_items(items, EXPECTED)  # every text read as the owner marked it

    monkeypatch.setattr(CMD.gold_draw, "predict_probs", predict_probs)
    scorer = CMD.validation_scorer(_validation(), _marks())
    assert scorer("models/x") == 1.0
    assert seen == [({"model": "models/x"}, True, ["Acme"] * 8)]  # the pair
    assert scorer.detail["macro"] == {"precision": 1.0, "recall": 1.0, "roc_ovr": 1.0}
    assert set(scorer.detail) == {"positive", "negative", "neutral", "macro"}
    assert scorer.predictions[0] == {
        "key": "k0", "expected": "negative", "predicted": "negative",
        "p_positive": "0.1", "p_negative": "0.8", "p_neutral": "0.1",
    }  # fmt: skip
    # a worse reading scores lower, by the same nine figures
    monkeypatch.setattr(
        CMD.gold_draw,
        "predict_probs",
        lambda items, cfg, pair=False: _scored_items(items, ["positive"] * 8),
    )
    worse = scorer("models/y")
    assert worse < 0.6 and scorer.detail["negative"]["recall"] == 0.0
    assert scorer.detail["negative"]["precision"] is None  # never predicted: undefined, not 0
    monkeypatch.setattr(CMD.gold_draw, "predict_probs", lambda items, cfg, pair=False: items[:2])
    with pytest.raises(spt.MarkedFileError, match="no chunk"):
        CMD.validation_scorer(_validation(), _marks())("models/x")


def test_the_baseline_scores_the_pinned_model_with_the_pair_and_without_and_keeps_the_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_run(monkeypatch, tmp_path)
    monkeypatch.setattr(CMD, "code_revision", lambda: "c1")
    seen: list[Any] = []

    def predict_probs(
        items: list[dict[str, str]], cfg: Any, pair: bool = False
    ) -> list[dict[str, Any]]:
        seen.append((cfg, pair))
        return _scored_items(items, EXPECTED if pair else ["positive"] * 8)

    monkeypatch.setattr(CMD.gold_draw, "predict_probs", predict_probs)
    result = CMD.run_baseline(_validation(), _marks())
    assert seen == [
        ({"model": "base", "revision": "rev"}, True),
        ({"model": "base", "revision": "rev"}, False),
    ]
    assert result["pair_score"] == 1.0 and result["text_only_score"] < result["pair_score"]
    record = json.loads((tmp_path / "out" / "baseline.json").read_text())
    assert record["model"] == "base@rev" and record["code"] == "c1" and record["n"] == 8
    assert (
        record["keys"] == CMD.keys_hash(KEYS) and record["pair_detail"]["macro"]["roc_ovr"] == 1.0
    )
    out = capsys.readouterr().out
    assert "pinned, pair: validation score 1.000 (weighted; the mean of the nine figures)" in out
    assert "negative precision 1.000, recall 1.000, ROC OVR 1.000" in out.replace(
        "  negative ", "negative "
    )
    assert CMD.read_predictions("baseline_pair") == sorted(_rows(EXPECTED), key=lambda r: r["key"])


def test_a_baseline_that_leaves_a_text_unread_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    monkeypatch.setattr(
        CMD.gold_draw,
        "predict_probs",
        lambda items, cfg, pair=False: _scored_items(items, EXPECTED)[:7],
    )
    with pytest.raises(spt.MarkedFileError, match="1 validation texts have no chunk"):
        CMD.run_baseline(_validation(), _marks())


def test_a_run_keeps_the_probabilities_and_the_per_class_figures_when_the_scorer_has_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    scorer = CMD.ValidationScorer(_validation(), _marks())
    monkeypatch.setattr(
        CMD.gold_draw,
        "predict_probs",
        lambda items, cfg, pair=False: _scored_items(items, ["positive"] * 8),
    )
    settings = spt.RunSettings(2, 1e-5, 1)
    row = CMD.one_run(settings, [], scorer)
    assert CMD.read_predictions(f"grid__{settings.name}") == sorted(
        _rows(["positive"] * 8), key=lambda r: r["key"]
    )
    # the trainer stub of these tests records only the score; the scorer carries the per-class figures
    assert row["validation_score"] == scorer("x") and scorer.detail["positive"]["recall"] == 1.0
    CMD.one_run(
        spt.RunSettings(2, 1e-5, 2), [], lambda _d: 0.5
    )  # a plain function: nothing to keep
    assert CMD.read_predictions("grid__e2_lr1e-05_s2") is None


def _report_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str = "c1"
) -> list[dict[str, str]]:
    """Predictions of the final run and both baselines for the 8 texts, their marks (two strata with different
    pools), baseline.json and the final row. Returns the marks."""
    _patch_run(monkeypatch, tmp_path)
    monkeypatch.setattr(CMD, "code_revision", lambda: "c1")
    monkeypatch.setattr(CMD, "REPORT_RESAMPLES", 30)
    marks = _marks()
    marks_file = tmp_path / "marks.csv"
    marks_file.write_text(
        "key,slice,predicted,pool_class_size,sentiment\n"
        + "".join(
            f"{m['key']},validation,{m['predicted']},{m['pool_class_size']},{m['sentiment']}\n"
            for m in marks
        )
    )
    monkeypatch.setattr(CMD, "MARKS", marks_file)
    CMD.write_predictions(
        "baseline_text_only", _rows(STRATA)
    )  # the pinned model's own draw prediction
    CMD.write_predictions("baseline_pair", _rows(["positive"] * 8))
    CMD.write_predictions("final__e2_lr3e-05_s4", _rows([*EXPECTED[:6], "neutral", "neutral"]))
    final = {"name": "e2_lr3e-05_s4", "tag": "final", "code": code}
    CMD.save_results(
        [
            {
                **final,
                "validation_score": CMD.measure(
                    _rows([*EXPECTED[:6], "neutral", "neutral"]), marks
                )[1],
            }
        ],
        {"fingerprint": "f"},
    )
    (tmp_path / "out" / "baseline.json").write_text(
        json.dumps({"model": "base@rev", "code": "c1", "keys": CMD.keys_hash(KEYS), "n": 8})
    )
    return marks


def test_the_report_gives_precision_recall_and_roc_ovr_weighted_first_and_the_raw_description_after(
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
    assert "Precision, recall and ROC OVR, per class (PLAN D37)" in out
    assert "agrees with its stored draw prediction on 8 of 8 texts" in out
    for reading in ("final", "pinned_text", "pinned_pair"):
        assert f"  {reading}: validation score " in out
    assert "positive precision" in out and "negative roc_ovr" in out and "macro roc_ovr" in out
    assert (
        "D32's bar on the final run's negative class (precision and recall 70%; ROC OVR has no bar)"
        in out
    )
    assert (
        "paired weighted gain of the final run over the pinned model as text (D35's pin rule)"
        in out
    )
    assert "score: " in out and "-> " in out  # the gold's verdict of each gain
    assert "the pin rule on these figures:" in out
    assert "(validation; the rule is applied once, to the gold)" in out
    assert "final (e2_lr3e-05_s4), RAW on the 8 texts: agreement 6/8" in out
    assert (
        "final against pinned text_only, RAW: both right 2, only the final 4, only the pinned 1, neither 1"
        in out
    )
    assert "exact McNemar p = 0.3750" in out and "no verdict" in out
    assert "WARNING" not in out  # every run's score matches its predictions


def test_the_report_stops_if_the_baseline_is_not_of_this_model_these_texts_or_this_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _report_fixture(tmp_path, monkeypatch)
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


def test_a_baseline_that_is_not_the_reading_the_strata_came_from_withholds_the_weighted_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _report_fixture(tmp_path, monkeypatch)
    CMD.write_predictions(
        "baseline_text_only", _rows(["neutral"] * 8)
    )  # not the stored draw prediction
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, right_on: dict[tuple[int, float], set[int]]
) -> list[dict[str, str]]:
    """Grid rows for the settings of *right_on*, three seeds each, with predictions: each setting reads right the
    texts (by position among the 8) it is given and wrong all the others; the rows log the score of their own
    predictions."""
    marks = _report_fixture(tmp_path, monkeypatch)
    rows = []
    for (epochs, lr), right in right_on.items():
        for seed in (1, 2, 3):
            name = f"e{epochs}_lr{lr:g}_s{seed}"
            predicted = [
                e if i in right else ("positive" if e == "neutral" else "neutral")
                for i, e in enumerate(EXPECTED)
            ]
            CMD.write_predictions(f"grid__{name}", _rows(predicted))
            rows.append(
                {
                    "name": name,
                    "tag": "grid",
                    "validation_score": CMD.measure(_rows(predicted), marks)[1],
                    "settings": {"epochs": epochs, "learning_rate": lr, "seed": seed},
                }
            )
    header, kept = CMD.load_results()
    CMD.save_results([*rows, *kept], header)
    return marks


def test_the_report_checks_every_runs_score_against_its_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _grid_with_predictions(tmp_path, monkeypatch, {(2, 1e-5): {0, 1, 2, 3}, (3, 2e-5): {0, 1, 2}})
    CMD.run_report()
    assert "WARNING" not in capsys.readouterr().out
    header, rows = CMD.load_results()
    for row in rows:
        if row["name"] == "e3_lr2e-05_s2":
            row["validation_score"] = 0.9  # the row says one thing, the file another
    CMD.save_results(rows, header)
    CMD.run_report()
    out = capsys.readouterr().out
    assert "WARNING: the predictions of e3_lr2e-05_s2 give a score of" in out
    assert "its row logged 0.9000: they are not of the same run" in out
    assert out.count("WARNING") == 1  # only that run
    (tmp_path / "out" / "predictions" / "grid__e3_lr2e-05_s1.csv").unlink()
    CMD.run_report()
    assert (
        "the predictions of e3_lr2e-05_s1 are not here: its score is not checked"
        in capsys.readouterr().out
    )


def test_a_timing_run_does_not_overwrite_the_predictions_of_the_grid_run_with_the_same_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, tmp_path)
    scorer = CMD.ValidationScorer(_validation(), _marks())
    answers = {"label": ["positive"] * 8}
    monkeypatch.setattr(
        CMD.gold_draw,
        "predict_probs",
        lambda items, cfg, pair=False: _scored_items(items, answers["label"]),
    )
    grid_run = CMD.TIMING_SETTING  # the timing setting is, by name, one of the grid's 27
    assert grid_run.name in {s.name for s in spt.grid_settings()}
    CMD.one_run(grid_run, [], scorer, tag="grid")
    kept = CMD.read_predictions(f"grid__{grid_run.name}")
    answers["label"] = ["neutral"] * 8  # another training run: another reading
    CMD.one_run(grid_run, [], scorer, tag="timing")
    assert (
        CMD.read_predictions(f"grid__{grid_run.name}") == kept
    )  # the grid run's file is untouched
    assert CMD.read_predictions(f"timing__{grid_run.name}") != kept


def test_the_summary_of_the_grid_shows_the_score_and_the_macro_precision_recall_and_roc_ovr_it_is_made_of(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def row(epochs: int, lr: float, seed: int, score: float) -> dict[str, Any]:
        macro = {"precision": score + 0.1, "recall": score, "roc_ovr": score - 0.1}
        return {
            "name": f"e{epochs}_lr{lr:g}_s{seed}",
            "tag": "grid",
            "validation_score": score,
            "validation_detail": {"macro": macro},
            "settings": {"epochs": epochs, "learning_rate": lr, "seed": seed},
        }

    rows = [row(2, 3e-5, s, 0.7) for s in (1, 2, 3)] + [row(4, 2e-5, s, 0.6) for s in (1, 2, 3)]
    choice = CMD.summarise(rows, 83)
    out = capsys.readouterr().out
    assert "2 epochs, lr 3e-05: 0.700 (precision 0.800, recall 0.700, ROC OVR 0.600)" in out
    assert "4 epochs, lr 2e-05: 0.600 (precision 0.700, recall 0.600, ROC OVR 0.500)" in out
    assert (choice.epochs, choice.learning_rate) == (2, 3e-5)
    # rows made before the per-class figures were recorded still print, without the parts
    plain = [{k: v for k, v in r.items() if k != "validation_detail"} for r in rows]
    CMD.summarise(plain)
    assert "2 epochs, lr 3e-05: 0.700 [0.7, 0.7, 0.7]" in capsys.readouterr().out


def test_the_all_data_entry_point_reads_both_gold_files_and_the_grid_keeps_the_first() -> None:
    """``--final-all`` and the cross-validation read every file of ``MARK_SETS``; the grid and ``--final``
    keep the 450-row file PR #154 recorded (``MARKS``), so their results stay reproducible."""
    assert list(CMD.MARK_SETS) == ["old", "new"]
    assert CMD.MARK_SETS["old"] == (CMD.MARKS, CMD.IDS)
    assert CMD.MARK_SETS["new"][0] == Path("gold/sentiment_train_2.csv")
    assert CMD.ALL_SETTING == (2, 3e-5)
    assert callable(CMD.build_all_data) and callable(CMD.run_final_all)


def test_build_all_data_merges_both_files_and_keeps_every_marked_body_out_of_the_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both files' rows are one set (the no-company rows counted, not trained), the replay is as large as the
    FIRST file's training rows, and a replay sentence of a body marked in the SECOND file is left out."""
    old_marks: list[dict[str, str]] = []
    old_ids: list[dict[str, str]] = []
    new_marks: list[dict[str, str]] = []
    new_ids: list[dict[str, str]] = []
    for i in range(3):
        for marks, ids, tag, slice_ in (
            (old_marks, old_ids, "o", "train"),
            (new_marks, new_ids, "n", "validation" if i == 2 else "train"),
        ):
            text = f"Acme item {tag}{i}."
            marks.append(
                {
                    "key": f"{tag}{i}#summary",
                    "text_hash": retag.text_hash(text),
                    "sentiment": ("positive", "negative", "neutral")[i],
                    "slice": slice_,
                    "ticker": "AAA",
                    "company": "Acme Corp",
                    "rule_scope": "single",
                    "body_hash": f"body-{tag}{i}",
                }
            )
            ids.append(
                {
                    "key": f"{tag}{i}#summary",
                    "text": text,
                    "body_hash": f"body-{tag}{i}",
                    "title": "",
                }
            )
    new_marks.append(
        {
            **new_marks[0],
            "key": "n9#summary",
            "ticker": "",
            "rule_scope": "sector",
            "body_hash": "body-n9",
            "text_hash": retag.text_hash("Sector news."),
        }
    )
    new_ids.append(
        {"key": "n9#summary", "text": "Sector news.", "body_hash": "body-n9", "title": ""}
    )
    files = {
        Path("o_marks"): old_marks, Path("o_ids"): old_ids, Path("n_marks"): new_marks, Path("n_ids"): new_ids,
    }  # fmt: skip
    monkeypatch.setattr(
        CMD,
        "MARK_SETS",
        {"old": (Path("o_marks"), Path("o_ids")), "new": (Path("n_marks"), Path("n_ids"))},
    )
    monkeypatch.setattr(CMD, "read_csv", lambda path: files[path])
    pool = tmp_path / "pool.jsonl"
    labels = ("positive", "negative", "neutral")
    sentences = [
        {"article_id": a, "sentence": f"Acme sentence {a}.", "label": labels[a % 3]}
        for a in range(1, 13)
    ]
    pool.write_text("\n".join(json.dumps(s) for s in sentences), encoding="utf-8")
    monkeypatch.setattr(CMD, "REPLAY_POOL", pool)
    # article 1 is the body marked in the second file: its sentence must not be in the replay
    infos = {
        a: spt.ArticleInfo("body-n0" if a == 1 else f"other-{a}", "AAA", "Acme Corp")
        for a in range(1, 13)
    }
    monkeypatch.setattr(CMD, "article_infos", lambda _conn, _ids: infos)
    monkeypatch.setattr(CMD, "exam_keys", lambda _conn, hashes, titles: (hashes, titles))
    monkeypatch.setattr(CMD, "OUT", tmp_path / "out")
    by_set, merged, replay = CMD.build_all_data(object())  # type: ignore[arg-type]
    assert {k: len(v.train) + len(v.validation) for k, v in by_set.items()} == {"old": 3, "new": 3}
    assert len(merged.train) + len(merged.validation) == 6
    assert merged.left_out == {"no company (sector)": 1}
    assert (
        len(replay) == len(by_set["old"].train) == 3
    )  # as large as the FIRST file's training rows
    assert all(not e.key.startswith("1#") for e in replay)
    assert "body-n9" in merged.body_hashes  # a no-company body is still kept out of the replay


def test_the_all_data_model_is_always_logged_with_its_path_and_is_made_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model with no MLflow run is what ``--final-all`` would otherwise leave: it logs without ``--mlflow``,
    with the number of summaries and replay sentences and the path of the model."""
    merged = spt.MarkedSummaries(
        train=[spt.Example("a#summary", "Acme", "Acme rose.", 0, "summary")],
        validation=[spt.Example("b#summary", "Acme", "Acme fell.", 1, "summary")],
    )
    replay = [spt.Example("1#x", "Acme", "Acme sentence.", 2, "sentence")]
    monkeypatch.setattr(CMD, "connect_source", object)
    monkeypatch.setattr(CMD, "build_all_data", lambda _conn: ({}, merged, replay))
    seen: dict[str, Any] = {}

    def one_run(
        settings: Any, train: Any, scorer: Any, *, keep: Path, **kwargs: Any
    ) -> dict[str, Any]:
        keep.mkdir(parents=True)
        seen["train"], seen["mlflow_on"] = list(train), kwargs.get("mlflow_on")
        return {"epoch_losses": [0.9, 0.4], "train_seconds": 1.0}

    logged: list[tuple[Any, ...]] = []
    monkeypatch.setattr(CMD, "one_run", one_run)
    monkeypatch.setattr(CMD, "log_final_all", lambda *args: logged.append(args))
    CMD.run_final_all(tmp_path)
    assert (tmp_path / "final").is_dir() and not (tmp_path / "final.partial").exists()
    assert [e.key for e in seen["train"]] == [
        "a#summary",
        "b#summary",
        "1#x",
    ]  # both slices and the replay
    assert seen["mlflow_on"] is False  # the generic per-run log is not the record: log_final_all is
    (_row, _settings, n_summaries, n_replay, target) = logged[0]
    assert (n_summaries, n_replay, target) == (2, 1, tmp_path / "final")
    with pytest.raises(SystemExit, match="exists"):
        CMD.run_final_all(tmp_path)


def test_the_all_data_record_goes_to_the_store_mlflow_resolves_and_says_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The record goes to the store MLflow resolves at that moment and the command prints which. (In the command
    that is ``MLFLOW_TRACKING_URI`` or ``./mlruns``; here the store is set the way other tests of this suite do,
    because MLflow's process-wide setting wins over the variable once something has set it.)"""
    import mlflow  # noqa: PLC0415 -- the eval group

    store = tmp_path / "store"
    before = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(store.as_uri())
    monkeypatch.setattr(CMD, "base_model", lambda: ("base", "rev"))
    row = {"epoch_losses": [0.9, 0.4], "train_seconds": 1.5}
    try:
        CMD.log_final_all(row, spt.RunSettings(2, 3e-5, seed=4), 809, 341, tmp_path / "final")
        assert store.as_uri() in capsys.readouterr().out
        runs = mlflow.search_runs(experiment_names=["sentiment_pair_final_all"])
    finally:
        mlflow.set_tracking_uri(before)
    assert len(runs) == 1
    assert runs.iloc[0]["params.output"] == str(tmp_path / "final")
    assert runs.iloc[0]["params.n_train"] == "1150"
