"""Step 6, the sentiment (T-168): the label and the three probabilities of each main company of a text, from the pair model and the stack of the
relevance score step 3 stored. What is pending, what is stored, how the step is run and what the pin refuses. The model is a stub: no network, no GPU."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sqlite3
import sys
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import news_nlp as db_module
import news_sentiment_stage as stage
from news_nlp import attribution_model as am
from news_nlp import steps
from news_nlp.steps import CompanyScore, SentimentScore
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source

SUM_REV = "summary-test"
COMP_REV = "companies-test"
REV = "sentiment-test"
ROOT = Path(__file__).resolve().parents[2]


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "news_nlp_steps_cli_sentiment", ROOT / "cli" / "news_nlp_steps.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pair(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    """A RESULTS store (with the tables of the steps) attached to a one-article SOURCE store."""
    source = tmp_path / "src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    seed_article(
        raw, id=1, title="One", body_text="Apple reported results. " * 30, fetch_status="ok"
    )
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    yield conn
    db_module.detach_source(conn)
    conn.close()


COEFFICIENTS = stage.stack_coefficients()


class StubScorer:
    """A scorer that loads no model: a text holding ``good`` reads positive, ``bad`` negative, ``empty`` has no chunk, ``broken`` raises like a lost GPU."""

    device = "cpu"

    def __init__(self, identity: str = "stubidentity") -> None:
        self._identity = identity
        self.calls: list[list[tuple[str, str]]] = []

    def identity(self) -> str:
        return self._identity

    def probabilities(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[list[float], int] | None]:
        self.calls.append(list(pairs))
        if any("broken" in t for t, _ in pairs):
            raise RuntimeError("CUDA out of memory")
        out: list[tuple[list[float], int] | None] = []
        for text, _ in pairs:
            if "empty" in text:
                out.append(None)
            elif "good" in text:
                out.append(([0.9, 0.05, 0.05], 1))
            elif "bad" in text:
                out.append(([0.05, 0.9, 0.05], 2))
            else:
                out.append(([0.1, 0.1, 0.8], 1))
        return out


class StubFinder:
    revision = REV
    model_name = "stub/sentiment@r1"
    model_identity = "stubidentity"

    def __init__(self, scorer: StubScorer | None = None) -> None:
        self.scorer = scorer or StubScorer()
        self.warm_ups = 0

    def warm_up(self) -> None:
        self.warm_ups += 1

    def classify(self, triples: Sequence[tuple[str, str, float]]) -> list[SentimentScore | None]:
        reads = self.scorer.probabilities([(t, c) for t, c, _ in triples])
        return [
            None if r is None else stage.score_of(COEFFICIENTS, r[0], rel)
            for r, (_, _, rel) in zip(reads, triples, strict=True)
        ]


# --- the stack: the arithmetic of the script that fitted it ------------------------------------------------------------------


def _script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "final_models_for_the_test", ROOT / "scripts" / "final_models_2026_11_10.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relevance", [0.0, 0.03, 0.5, 0.97, 1.0])
@pytest.mark.parametrize("p", [[0.9, 0.05, 0.05], [0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
def test_the_stack_is_the_arithmetic_of_the_script_that_fitted_it(
    p: list[float], relevance: float
) -> None:
    expected = _script().apply_stack(COEFFICIENTS, p, relevance)
    got = stage.stacked(COEFFICIENTS, p, relevance)
    assert got.keys() == expected.keys()
    for name in expected:
        assert got[name] == pytest.approx(expected[name], abs=1e-9)
    assert sum(got.values()) == pytest.approx(1.0)


def test_the_classes_are_the_models_order_and_the_stack_file_says_so() -> None:
    assert stage.CLASSES == ("positive", "negative", "neutral")
    assert COEFFICIENTS["classes"] == list(stage.CLASSES)
    assert COEFFICIENTS["class_order_of_coef"] == list(stage.CLASSES)
    assert len(COEFFICIENTS["coef"]) == 3 and all(len(row) == 4 for row in COEFFICIENTS["coef"])


def test_a_stack_in_another_order_is_refused(tmp_path: Path) -> None:
    other = dict(COEFFICIENTS, classes=["negative", "positive", "neutral"])
    path = tmp_path / "stack.json"
    path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="classes"):
        stage.stack_coefficients(path)


def test_the_label_is_the_largest_stacked_probability_and_a_low_relevance_pulls_to_neutral() -> (
    None
):
    relevant = stage.score_of(COEFFICIENTS, [0.97, 0.01, 0.02], 0.95)
    assert relevant.label == "positive" and relevant.score == pytest.approx(relevant.positive)
    # a positive text about another company: the stack turns it neutral for the company asked (docs/sentiment_training.md, "Published")
    elsewhere = stage.score_of(COEFFICIENTS, [0.97, 0.01, 0.02], 0.05)
    assert elsewhere.label == "neutral"
    assert relevant.positive + relevant.negative + relevant.neutral == pytest.approx(1.0)


def test_the_finder_scores_each_triple_and_a_text_with_no_chunk_has_no_score() -> None:
    finder = stage.SentimentFinder(SUM_REV, COMP_REV, StubScorer(), COEFFICIENTS)
    out = finder.classify(
        [("good news", "Apple Inc.", 0.95), ("bad news", "Apple Inc.", 0.95), ("empty", "A", 0.9)]
    )
    assert [None if r is None else r.label for r in out] == ["positive", "negative", None]
    assert finder.revision.startswith("sentiment-")
    assert (
        finder.model_name
        == "gamug/FinBERT-company-sentiment@9a7b1eb34c3f2e03c841173e18812f0d3d6f2d3e"
    )


# --- the revision ---------------------------------------------------------------------------------------------------------


def test_the_revision_follows_the_model_the_stack_and_the_inputs() -> None:
    base = stage.scoring_revision("m1", "s1", SUM_REV, COMP_REV)
    assert base == stage.scoring_revision("m1", "s1", SUM_REV, COMP_REV)
    assert base != stage.scoring_revision("m2", "s1", SUM_REV, COMP_REV)
    assert base != stage.scoring_revision("m1", "s2", SUM_REV, COMP_REV)
    assert base != stage.scoring_revision("m1", "s1", "summary-other", COMP_REV)
    assert base != stage.scoring_revision("m1", "s1", SUM_REV, "companies-other")


def _with_source(monkeypatch: pytest.MonkeyPatch, function: Any, source: str) -> None:
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda obj: source if obj is function else real(obj))


@pytest.mark.parametrize("function_name", ["stacked", "score_of", "_pair_probabilities"])
def test_a_comment_a_docstring_or_the_formatting_of_the_scoring_code_is_not_a_new_revision(
    function_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = getattr(stage, function_name)
    base = stage.scoring_revision("m", "s", SUM_REV, COMP_REV)
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    node.body = [
        ast.Expr(ast.Constant("another docstring")),
        *node.body[1 if ast.get_docstring(node) else 0 :],
    ]
    reformatted = "# a comment before\n\n\n" + ast.unparse(tree) + "\n\n# and one after\n"
    _with_source(monkeypatch, function, reformatted)
    assert stage.scoring_revision("m", "s", SUM_REV, COMP_REV) == base


@pytest.mark.parametrize(
    ("function_name", "old", "new"),
    [
        ("stacked", "max(", "min("),
        ("score_of", "max(", "min("),
        ("_pair_probabilities", "sum(", "max("),
    ],
)
def test_a_statement_of_the_scoring_code_is_a_new_revision(
    function_name: str, old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = getattr(stage, function_name)
    base = stage.scoring_revision("m", "s", SUM_REV, COMP_REV)
    source = inspect.getsource(function)
    assert old in source
    _with_source(monkeypatch, function, source.replace(old, new, 1))
    assert stage.scoring_revision("m", "s", SUM_REV, COMP_REV) != base


def test_the_bookkeeping_of_the_scorer_is_not_part_of_the_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = stage.scoring_revision("m", "s", SUM_REV, COMP_REV)
    source = inspect.getsource(stage.SentimentScorer.probabilities)
    _with_source(
        monkeypatch,
        stage.SentimentScorer.probabilities,
        source.replace("out.append(None)", "out.append(None)  # edited"),
    )
    assert stage.scoring_revision("m", "s", SUM_REV, COMP_REV) == base


def test_the_status_revision_comes_from_the_pin_and_loads_nothing() -> None:
    assert stage.code_revision(SUM_REV, COMP_REV) == stage.scoring_revision(
        stage.pin()["identity"], stage.stack_identity(), SUM_REV, COMP_REV
    )
    models = stage.models_of_the_step()
    assert "gamug/FinBERT-company-sentiment@9a7b1eb3" in models["published_as"]


# --- the items and the rows -----------------------------------------------------------------------------------------------


def _seed(conn: db_module.NewsNlpDatabase) -> None:
    """Two bodies and a sub-news with a text at SUM_REV; b1 has two main companies, b2 only a mention (review), 7 one main."""
    for kind, item_id, text in [
        (steps.ITEM_BODY, "b1", "good: the results"),
        (steps.ITEM_BODY, "b2", "bad: a mention"),
        (steps.ITEM_SUB_NEWS, "7", "bad: a recall"),
    ]:
        steps.record_summary(conn, kind, item_id, SUM_REV, text, "original", 0)
    steps.record_summary(conn, steps.ITEM_BODY, "b1", "summary-other", "elsewhere", "original", 0)
    steps.record_companies(
        conn,
        steps.ITEM_BODY,
        "b1",
        COMP_REV,
        [CompanyScore("AAPL", 2, 0.9), CompanyScore("MSFT", 1, 0.7)],
        0.5,
    )
    steps.record_companies(
        conn, steps.ITEM_BODY, "b2", COMP_REV, [CompanyScore("AAPL", 1, 0.2)], 0.5
    )  # no main: the review list
    steps.record_companies(
        conn, steps.ITEM_SUB_NEWS, "7", COMP_REV, [CompanyScore("MSFT", 1, 0.8)], 0.5
    )


def _items(conn: db_module.NewsNlpDatabase, **kw: Any) -> list[steps.SentimentItem]:
    return list(steps.pending_sentiment_items(conn, SUM_REV, COMP_REV, REV, **kw))


def test_the_items_read_carry_the_relevance_score_step_3_stored_for_each_pair(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    items = _items(pair)
    assert [(i.kind, i.item_id, i.tickers) for i in items] == [
        ("body", "b1", (("AAPL", 0.9), ("MSFT", 0.7))),
        ("sub_news", "7", (("MSFT", 0.8),)),
    ]  # b2 is on the review list: no main company, no sentiment
    assert _items(pair, limit=1)[0].item_id == "b1"
    assert list(steps.pending_sentiment_items(pair, "summary-none", COMP_REV, REV)) == []
    assert list(steps.pending_sentiment_items(pair, SUM_REV, "companies-none", REV)) == []


def test_a_row_is_stored_per_main_company_and_a_repeat_adds_nothing(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    score = stage.score_of(COEFFICIENTS, [0.9, 0.05, 0.05], 0.9)
    found = {"AAPL": score, "MSFT": score}
    assert steps.record_sentiment(pair, "body", "b1", REV, found, "stub") == "done"
    steps.record_sentiment(pair, "body", "b1", REV, found, "stub")
    rows = pair.execute(
        "SELECT ticker, label, score, positive, negative, neutral, model_name, sentiment_checked "
        "FROM news_sentiment ORDER BY ticker"
    ).fetchall()
    assert len(rows) == 2 and rows[0][1] == "positive" and rows[0][6] == "stub" and rows[0][7] == 1
    assert rows[0][3] + rows[0][4] + rows[0][5] == pytest.approx(1.0)
    assert [i.item_id for i in _items(pair)] == ["7"]  # b1 is finished at this revision
    other = list(steps.pending_sentiment_items(pair, SUM_REV, COMP_REV, "sentiment-other"))
    assert [i.item_id for i in other] == ["b1", "7"]  # another revision starts again


def test_the_status_counts_the_rows_by_label(pair: db_module.NewsNlpDatabase) -> None:
    _seed(pair)
    steps.record_sentiment(
        pair, "body", "b1", REV, {"AAPL": stage.score_of(COEFFICIENTS, [0.9, 0.05, 0.05], 0.9)}, "s"
    )
    report = steps.status(pair, steps.STEP_SENTIMENT, REV)
    assert report["news_sentiment"] == {"rows": 1, "items": 1}
    assert report["by_label"] == {"positive": 1}
    assert report["by_state"] == {"done": 1}


# --- the run -------------------------------------------------------------------------------------------------------------


NAMES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}


def test_a_run_scores_each_main_company_and_returns_the_counts(
    pair: db_module.NewsNlpDatabase,
) -> None:
    cli = _load_cli()
    _seed(pair)
    finder = StubFinder()
    counts = cli.run_sentiment(pair, finder, NAMES, SUM_REV, COMP_REV, None)
    assert counts == {"done": 2, "failed": 0, "skipped": 0, "sentiment_rows": 3, "name_missing": 0}
    assert finder.warm_ups == 1
    again = cli.run_sentiment(pair, finder, NAMES, SUM_REV, COMP_REV, None)
    assert again["done"] == 0  # a rerun adds nothing
    assert pair.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0] == 3


def test_the_pair_is_read_for_the_company_name_of_the_universe_with_its_relevance(
    pair: db_module.NewsNlpDatabase,
) -> None:
    cli = _load_cli()
    _seed(pair)
    finder = StubFinder()
    cli.run_sentiment(pair, finder, NAMES, SUM_REV, COMP_REV, 1)
    assert finder.scorer.calls[0] == [
        ("good: the results", "Apple Inc."),
        ("good: the results", "Microsoft Corporation"),
    ]
    counts = cli.run_sentiment(pair, finder, {}, SUM_REV, COMP_REV, None)
    assert (
        counts["name_missing"] == 1
    )  # MSFT of the sub-news: not in the universe given, read as its ticker


def test_a_blank_text_or_one_with_no_sentence_is_skipped_with_a_reason_and_not_retried(
    pair: db_module.NewsNlpDatabase,
) -> None:
    cli = _load_cli()
    steps.record_summary(pair, "body", "blank", SUM_REV, "   ", "original", 0)
    steps.record_summary(pair, "body", "nothing", SUM_REV, "empty text", "original", 0)
    for item_id in ("blank", "nothing"):
        steps.record_companies(pair, "body", item_id, COMP_REV, [CompanyScore("AAPL", 1, 0.9)], 0.5)
    counts = cli.run_sentiment(pair, StubFinder(), NAMES, SUM_REV, COMP_REV, None)
    assert counts["skipped"] == 2 and counts["done"] == 0
    reasons = dict(
        pair.execute(
            "SELECT item_id, error FROM pipeline_step_state WHERE step = 'sentiment' AND status = 'skipped'"
        ).fetchall()
    )
    assert reasons == {"blank": "blank text", "nothing": "no sentence to read"}
    assert list(steps.pending_sentiment_items(pair, SUM_REV, COMP_REV, REV)) == []


def test_a_batch_that_fails_is_redone_item_by_item_so_one_bad_text_fails_alone(
    pair: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    _seed(pair)
    steps.record_summary(pair, "body", "bad1", SUM_REV, "broken text", "original", 0)
    steps.record_companies(pair, "body", "bad1", COMP_REV, [CompanyScore("AAPL", 1, 0.9)], 0.5)
    monkeypatch.setattr(
        cli, "_gpu_lost", lambda exc: False
    )  # an out-of-memory error, not a lost GPU
    counts = cli.run_sentiment(pair, StubFinder(), NAMES, SUM_REV, COMP_REV, None)
    assert counts["done"] == 2 and counts["failed"] == 1
    failed = pair.execute(
        "SELECT item_id, status FROM pipeline_step_state WHERE step = 'sentiment' AND status = 'failed'"
    ).fetchall()
    assert [tuple(r) for r in failed] == [("bad1", "failed")]
    assert [i.item_id for i in _items(pair)] == ["bad1"]  # a failed item is retried


def test_a_gpu_fault_stops_the_run_at_once_and_marks_no_item_failed(
    pair: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    steps.record_summary(pair, "body", "x", SUM_REV, "broken text", "original", 0)
    steps.record_companies(pair, "body", "x", COMP_REV, [CompanyScore("AAPL", 1, 0.9)], 0.5)
    monkeypatch.setattr(cli, "_gpu_lost", lambda exc: True)
    with pytest.raises(cli.RunAborted, match="GPU failed"):
        cli.run_sentiment(pair, StubFinder(), NAMES, SUM_REV, COMP_REV, None)
    assert (
        pair.execute("SELECT COUNT(*) FROM pipeline_step_state WHERE status = 'failed'").fetchone()[
            0
        ]
        == 0
    )


# --- the command ------------------------------------------------------------------------------------------------------------


def _parse(cli: Any, *argv: str) -> Any:
    parser = cli.build_parser()
    args = parser.parse_args(["--step", "sentiment", *argv])
    cli.check_options(parser, args)
    return args


def test_the_step_needs_both_revisions_to_run_and_refuses_what_it_does_not_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    _parse(cli, "--status")  # the status needs none
    _parse(cli, "--summary-revision", "s", "--companies-revision", "c")
    for argv, message in [
        ((), "needs --summary-revision"),
        (("--summary-revision", "s"), "needs --companies-revision"),
        (
            ("--summary-revision", "s", "--companies-revision", "c", "--model-dir", "m"),
            "--model-dir does not apply",
        ),
        (
            ("--summary-revision", "s", "--companies-revision", "c", "--scope-revision", "r"),
            "--scope-revision does not apply",
        ),
        (
            ("--summary-revision", "s", "--companies-revision", "c", "--fetch"),
            "only apply to --step scope and --step cut",
        ),
    ]:
        with pytest.raises(SystemExit):
            _parse(cli, *argv)
        assert message in capsys.readouterr().err
    parser = cli.build_parser()
    with pytest.raises(SystemExit):  # --companies-revision belongs to category and sentiment
        cli.check_options(
            parser, parser.parse_args(["--step", "summary", "--companies-revision", "c"])
        )
    assert "category and --step sentiment" in capsys.readouterr().err


def test_the_recipes_of_a_recut_and_of_a_rerun_name_the_sentiment_rows() -> None:
    cli = _load_cli()
    recut = "\n".join(cli.RECUT_SQL)
    assert "DELETE FROM news_sentiment WHERE item_kind = 'sub_news'" in recut
    assert "step = 'sentiment' AND item_kind = 'sub_news'" in recut
    assert "news_sentiment WHERE model_revision" in "\n".join(cli.RESENTIMENT_SQL)
    assert "--step sentiment --status" in cli.RERUN_HELP


def test_the_status_prints_the_model_the_stack_and_the_revision_without_loading_anything(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    args = SimpleNamespace(status=True, summary_revision=SUM_REV, companies_revision=COMP_REV)
    cli.main_sentiment(args, pair)
    report = json.loads(capsys.readouterr().out)
    assert report["code_revision"] == stage.code_revision(SUM_REV, COMP_REV)
    assert report["models"]["published_as"].startswith("gamug/FinBERT-company-sentiment@")
    assert report["step"] == "sentiment"


def _run_args(**kw: Any) -> Any:
    return SimpleNamespace(
        status=False, summary_revision=SUM_REV, companies_revision=COMP_REV, limit=None, **kw
    )


def test_the_run_refuses_a_folder_that_is_not_the_pinned_one_and_a_missing_folder(
    pair: db_module.NewsNlpDatabase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    _seed(pair)
    folder = tmp_path / "model"
    monkeypatch.setattr(stage, "model_dir", lambda pin_path=None: folder)
    with pytest.raises(SystemExit, match="never downloaded"):
        cli.main_sentiment(_run_args(), pair)  # no folder
    folder.mkdir()
    monkeypatch.setattr(stage, "SentimentScorer", lambda *a, **k: StubScorer("some-other-identity"))
    with pytest.raises(SystemExit, match="not the pinned"):
        cli.main_sentiment(_run_args(), pair)
    assert pair.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0] == 0


def test_the_run_on_the_pinned_folder_scores_and_prints_the_counts(
    pair: db_module.NewsNlpDatabase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    _seed(pair)
    folder = tmp_path / "model"
    folder.mkdir()
    monkeypatch.setattr(stage, "model_dir", lambda pin_path=None: folder)
    monkeypatch.setattr(
        stage, "SentimentScorer", lambda *a, **k: StubScorer(stage.pin()["identity"])
    )
    monkeypatch.setattr(cli, "tqdm", lambda *a, **k: _NoBar())
    cli.main_sentiment(_run_args(), pair)
    out = capsys.readouterr().out
    assert '"sentiment_rows": 3' in out and '"done": 2' in out
    assert pair.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0] == 3


class _NoBar:
    def __enter__(self) -> _NoBar:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def update(self, n: int) -> None:
        return None


# --- the pin ------------------------------------------------------------------------------------------------------------------


def test_the_pin_names_the_published_copy_the_stack_and_a_local_folder() -> None:
    pin = stage.pin()
    published = json.loads((ROOT / "config" / "published_models.json").read_text())["sentiment"]
    assert pin["published"] == {"repo": published["repo"], "revision": published["revision"]}
    assert len(pin["identity"]) == 16 and not Path(pin["model_dir"]).is_absolute()
    assert (ROOT / "config" / "sentiment_stack.json").exists()


@pytest.mark.skipif(
    not (ROOT / "models" / "sentiment_pair_all" / "final").is_dir(),
    reason="the local pair model is git-ignored and absent from a clean checkout",
)
def test_the_local_folder_is_the_pinned_one() -> None:
    assert am.directory_identity(stage.model_dir()) == stage.pin()["identity"]
