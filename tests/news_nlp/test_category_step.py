"""Step 5, the category (T-168; provisional, the zero-shot model): the label and group of each main company of a text, what is pending,
what is stored and how the step is run. The model is a stub: no network, no GPU."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import sqlite3
import sys
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import category_stage
import cut_stage
import news_category_stage as stage
import news_company_stage
import news_nlp as db_module
import pipeline
from news_nlp import retag, steps
from news_nlp.steps import CategoryScore, CompanyScore
from news_nlp.triage import Company
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source

SUM_REV = "summary-test"
COMP_REV = "companies-test"
REV = "category-test"


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "news_nlp_steps_cli_category",
        Path(__file__).resolve().parents[2] / "cli" / "news_nlp_steps.py",
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


def _scores(top: str, value: float = 0.9) -> list[float]:
    """An entailment vector whose best label is ``top``."""
    return [value if label == top else 0.05 for label in stage.LABELS]


class StubScorer:
    """A scorer that loads no model: the label of a text is the marker it holds, ``broken`` markers raise like a lost GPU."""

    model_name = "stub/zeroshot"
    model_revision = "r1"
    device = "cpu"

    def __init__(self, broken: Sequence[str] = ()) -> None:
        self.broken = tuple(broken)
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
        self.calls.append(list(pairs))
        if any(m in t for t, _ in pairs for m in self.broken):
            raise RuntimeError("CUDA out of memory")
        return [_scores(next((lab for lab in stage.LABELS if lab in t), "other")) for t, _ in pairs]


class StubFinder:
    revision = REV
    model_name = "stub/zeroshot"
    model_revision = "r1"

    def __init__(self, scorer: StubScorer | None = None) -> None:
        self.scorer = scorer or StubScorer()
        self.warm_ups = 0

    def warm_up(self) -> None:
        self.warm_ups += 1

    def classify(self, pairs: Sequence[tuple[str, str]]) -> list[CategoryScore]:
        return [stage.score_of(p) for p in self.scorer.score(pairs)]


# --- the taxonomy: the labels, the groups, the score ------------------------------------------------------------------------


def test_the_labels_are_the_taxonomy_of_d15_and_every_label_is_in_one_group() -> None:
    assert set(stage.LABELS) == set(retag.D15_LABELS) - {"uncertain", "unrelated"}
    assert len(stage.LABELS) == 14
    grouped = [label for labels in stage.GROUPS.values() for label in labels]
    assert sorted(grouped) == sorted(stage.LABELS)  # each label once
    assert set(stage.DEFINITIONS) == set(stage.LABELS)


def test_the_label_is_the_best_entailment_and_the_group_is_the_group_of_the_label() -> None:
    result = stage.score_of(_scores("legal_regulatory", 0.8))
    assert result.label == "legal_regulatory" and result.score == 0.8
    assert result.group_label == "governance_legal_reputation"
    assert 0 < result.group_score <= 1
    assert set(result.slug_scores) == set(stage.LABELS)  # the audit trail has every label


def test_the_group_score_is_the_share_of_the_mass_the_group_holds() -> None:
    result = stage.score_of(_scores("other", 0.9))
    assert result.group_label == "other"
    assert result.group_score == pytest.approx(0.9 / (0.9 + 0.05 * 13))


def test_the_finder_classifies_each_pair_and_names_the_model_it_ran() -> None:
    finder = stage.CategoryFinder(SUM_REV, COMP_REV, StubScorer())
    out = finder.classify([("legal_regulatory news", "Apple Inc."), ("nothing", "Apple Inc.")])
    assert [r.label for r in out] == ["legal_regulatory", "other"]
    assert finder.revision.startswith("category-")


def test_the_company_name_loses_its_share_class() -> None:
    assert stage.company_name("Alphabet Inc. (Class A)") == "Alphabet Inc."
    assert stage.company_name("Apple Inc.") == "Apple Inc."


def test_the_revision_follows_the_model_the_definitions_and_the_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = stage.revision_of("m@1", SUM_REV, COMP_REV)
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) == base  # stable
    assert stage.revision_of("m@2", SUM_REV, COMP_REV) != base  # another model, another revision
    monkeypatch.setitem(stage.DEFINITIONS, "other", "something else")
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) != base
    monkeypatch.undo()
    monkeypatch.setattr(stage, "PREMISE_CHARS", 1000)
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) != base


def test_the_status_names_the_model_without_loading_it() -> None:
    models = stage.models_of_the_step()
    assert models["category"].startswith("MoritzLaurer/deberta-v3-base-zeroshot-v2.0@")


# --- storage and what is pending -----------------------------------------------------------------------------------------


def _seed(conn: db_module.NewsNlpDatabase) -> None:
    """Two bodies and a sub-news with a text at SUM_REV; b1 has two main companies, b2 only a mention (review), 7 one main."""
    for kind, item_id, text in [
        (steps.ITEM_BODY, "b1", "legal_regulatory: the lawsuit"),
        (steps.ITEM_BODY, "b2", "earnings_performance: results"),
        (steps.ITEM_SUB_NEWS, "7", "product_innovation: a launch"),
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


def _items(conn: db_module.NewsNlpDatabase, **kw: Any) -> list[steps.CategoryItem]:
    return list(steps.pending_category_items(conn, SUM_REV, COMP_REV, REV, **kw))


def test_the_items_read_are_those_with_a_main_company_and_a_text_at_the_named_revisions(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    items = _items(pair)
    assert [(i.kind, i.item_id, i.tickers) for i in items] == [
        ("body", "b1", ("AAPL", "MSFT")),
        ("sub_news", "7", ("MSFT",)),
    ]  # b2 is on the review list: no main company, no category
    assert _items(pair, limit=1)[0].item_id == "b1"
    assert list(steps.pending_category_items(pair, "summary-none", COMP_REV, REV)) == []
    assert list(steps.pending_category_items(pair, SUM_REV, "companies-none", REV)) == []


def test_a_category_row_is_stored_per_main_company_and_a_repeat_adds_nothing(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    found = {
        "AAPL": stage.score_of(_scores("legal_regulatory")),
        "MSFT": stage.score_of(_scores("other")),
    }
    for _ in range(2):
        assert steps.record_category(pair, "body", "b1", REV, found, "stub/zeroshot") == "done"
    rows = [
        tuple(r)
        for r in pair.execute(
            "SELECT ticker, label, group_label, model_name FROM news_category ORDER BY ticker"
        )
    ]
    assert rows == [
        ("AAPL", "legal_regulatory", "governance_legal_reputation", "stub/zeroshot"),
        ("MSFT", "other", "other", "stub/zeroshot"),
    ]
    audit = json.loads(pair.execute("SELECT slug_scores FROM news_category LIMIT 1").fetchone()[0])
    assert set(audit) == set(stage.LABELS)
    assert [i.item_id for i in _items(pair)] == ["7"]  # b1 is finished at this revision
    assert (
        len(list(steps.pending_category_items(pair, SUM_REV, COMP_REV, "category-new"))) == 2
    )  # a new revision starts over


def test_a_failed_item_is_retried_and_the_status_counts_the_rows(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    steps.record_item_failure(pair, "body", "b1", steps.STEP_CATEGORY, REV, "boom")
    assert [i.item_id for i in _items(pair)] == ["b1", "7"]
    steps.record_category(
        pair, "sub_news", "7", REV, {"MSFT": stage.score_of(_scores("product_innovation"))}, "m"
    )
    report = steps.status(pair, steps.STEP_CATEGORY, REV)
    assert report["by_state"] == {"done": 1, "failed": 1}
    assert report["news_category"] == {"rows": 1, "items": 1, "truncated": 0}
    assert report["by_label"] == {"product_innovation": 1}
    assert report["by_group"] == {"product_people": 1}


# --- running the step ----------------------------------------------------------------------------------------------------


NAMES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}


def test_a_run_labels_each_main_company_and_returns_the_counts(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    cli = _load_cli()
    counts = cli.run_category(pair, StubFinder(), NAMES, SUM_REV, COMP_REV, None)
    assert counts == {"done": 2, "failed": 0, "skipped": 0, "category_rows": 3, "name_missing": 0}
    labels = {
        (r[0], r[1]): r[2] for r in pair.execute("SELECT item_id, ticker, label FROM news_category")
    }
    assert labels == {
        ("b1", "AAPL"): "legal_regulatory",
        ("b1", "MSFT"): "legal_regulatory",
        ("7", "MSFT"): "product_innovation",
    }
    again = cli.run_category(pair, StubFinder(), NAMES, SUM_REV, COMP_REV, None)
    assert again == {
        "done": 0,
        "failed": 0,
        "skipped": 0,
        "category_rows": 0,
        "name_missing": 0,
    }  # a rerun adds nothing


def test_the_company_name_is_what_the_model_reads_and_limit_bounds_the_items(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    scorer = StubScorer()
    cli = _load_cli()
    counts = cli.run_category(pair, StubFinder(scorer), NAMES, SUM_REV, COMP_REV, 1)
    assert counts["done"] == 1
    assert scorer.calls[0] == [
        ("legal_regulatory: the lawsuit", "Apple Inc."),
        ("legal_regulatory: the lawsuit", "Microsoft Corporation"),
    ]


def test_a_batch_that_fails_is_redone_item_by_item_so_one_bad_text_fails_alone(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    cli = _load_cli()
    counts = cli.run_category(
        pair, StubFinder(StubScorer(broken=["a launch"])), NAMES, SUM_REV, COMP_REV, None
    )
    assert counts["done"] == 1 and counts["failed"] == 1
    state = dict(
        pair.execute(
            "SELECT item_id, status FROM pipeline_step_state WHERE step = 'category'"
        ).fetchall()
    )
    assert state == {"b1": "done", "7": "failed"}


def test_batches_that_all_fail_in_a_row_stop_the_run(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    cli = _load_cli()
    cli.CATEGORY_BATCH = 1
    cli.MAX_FAILED_BATCHES = 2
    for n in range(3):  # more items than the limit of failed batches
        steps.record_summary(pair, "body", f"x{n}", SUM_REV, "legal_regulatory text", "original", 0)
        steps.record_companies(pair, "body", f"x{n}", COMP_REV, [CompanyScore("AAPL", 1, 0.9)], 0.5)
    with pytest.raises(cli.RunAborted):
        cli.run_category(
            pair,
            StubFinder(StubScorer(broken=["text", "lawsuit", "launch"])),
            NAMES,
            SUM_REV,
            COMP_REV,
            None,
        )


# --- the command line ----------------------------------------------------------------------------------------------------


def test_the_category_step_needs_the_summary_and_the_companies_revision_to_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    parser = cli.build_parser()
    base = ["--step", "category"]
    full = ["--summary-revision", "summary-x", "--companies-revision", "companies-x"]
    cli.check_options(parser, parser.parse_args(base + full))  # both: accepted
    for missing, flag in (("--summary-revision", 0), ("--companies-revision", 2)):
        args = base + full[:flag] + full[flag + 2 :]
        with pytest.raises(SystemExit):
            cli.check_options(parser, parser.parse_args(args))
        assert missing in capsys.readouterr().err
    cli.check_options(parser, parser.parse_args([*base, "--status"]))  # --status needs neither


def test_the_options_of_the_category_step_belong_to_it() -> None:
    cli = _load_cli()
    parser = cli.build_parser()
    for argv in (
        ["--step", "scope", "--companies-revision", "c"],
        [
            "--step",
            "category",
            "--summary-revision",
            "s",
            "--companies-revision",
            "c",
            "--threshold",
            "0.5",
        ],
        [
            "--step",
            "category",
            "--summary-revision",
            "s",
            "--companies-revision",
            "c",
            "--model-dir",
            "x",
        ],
    ):
        with pytest.raises(SystemExit):
            cli.check_options(parser, parser.parse_args(argv))


def test_the_status_of_the_category_step_loads_no_model(
    pair: db_module.NewsNlpDatabase,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed(pair)
    steps.record_category(pair, "body", "b1", REV, {"AAPL": stage.score_of(_scores("other"))}, "m")
    cli = _load_cli()

    class Args:
        status = True

    cli.main_category(Args(), pair)
    out = json.loads(capsys.readouterr().out)
    assert out["step"] == "category" and out["by_state"] == {"done": 1}
    assert out["revisions"] == [REV]
    assert out["models"]["category"].startswith("MoritzLaurer/")


def test_items_whose_text_is_at_another_summary_revision_are_counted(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    assert (
        steps.category_items_without_text(pair, SUM_REV, COMP_REV) == 0
    )  # b2 has no main: not counted; b1 and 7 have a text
    steps.record_companies(pair, "body", "b3", COMP_REV, [CompanyScore("AAPL", 1, 0.9)], 0.5)
    assert (
        steps.category_items_without_text(pair, SUM_REV, COMP_REV) == 1
    )  # b3 has a main and no text
    assert steps.category_items_without_text(pair, "summary-none", COMP_REV) == 3


def test_a_ticker_outside_the_universe_is_read_as_its_own_name_and_counted(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    scorer = StubScorer()
    counts = _load_cli().run_category(
        pair, StubFinder(scorer), {"AAPL": "Apple Inc."}, SUM_REV, COMP_REV, None
    )
    assert counts["name_missing"] == 2  # MSFT of b1 and MSFT of the sub-news
    assert ("legal_regulatory: the lawsuit", "MSFT") in scorer.calls[0]


# --- the model side, with a stub tokenizer and model ---------------------------------------------------------------------


class _Encoded(dict[str, Any]):
    def to(self, _device: Any) -> _Encoded:
        return self


def test_the_scorer_reads_one_hypothesis_per_label_and_returns_the_entailment_of_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def tokenizer(premises: list[str], hypotheses: list[str], **kw: Any) -> _Encoded:
        seen.update(premises=premises, hypotheses=hypotheses, kw=kw)
        return _Encoded(
            input_ids=torch.zeros(len(premises), 20, dtype=torch.long),
            attention_mask=torch.ones(len(premises), 20, dtype=torch.long),
        )

    def model(**_: Any) -> SimpleNamespace:
        logits = torch.tensor([[0.0, 5.0]] * len(stage.LABELS))  # column 0 is entailment
        logits[3] = torch.tensor([5.0, 0.0])  # the fourth label is the one entailed
        return SimpleNamespace(logits=logits)

    scorer = stage.CategoryScorer(device=torch.device("cpu"))
    monkeypatch.setattr(scorer, "_load", lambda: None)
    scorer._tokenizer, scorer._model, scorer._entailment = tokenizer, model, 0
    out = scorer.score([("x" * 3000, "Alphabet Inc. (Class A)")])
    assert len(out) == 1 and len(out[0]) == len(stage.LABELS)
    assert max(range(len(out[0])), key=lambda i: out[0][i]) == 3 and out[0][3] > 0.99
    assert (
        seen["premises"][0] == "Alphabet Inc.: " + "x" * stage.PREMISE_CHARS
    )  # the share class goes, the text is cut
    assert len(seen["hypotheses"]) == len(stage.LABELS)
    assert seen["hypotheses"][0] == (
        f"The news about Alphabet Inc. is about {stage.DEFINITIONS[stage.LABELS[0]]}."
    )
    assert seen["kw"]["truncation"] == "only_first" and seen["kw"]["max_length"] == stage.MAX_LENGTH
    assert scorer.last_truncated == [False]  # 20 tokens: nothing was cut


def test_a_pair_the_model_read_cut_at_its_window_is_recorded_and_read_apart_from_the_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constitution, Pipeline models 3: a stage does not truncate without recording that it did."""

    def tokenizer(premises: list[str], hypotheses: list[str], **_: Any) -> _Encoded:
        n = stage.MAX_LENGTH if "cut" in premises[0] else 20
        return _Encoded(
            input_ids=torch.zeros(len(premises), n, dtype=torch.long),
            attention_mask=torch.ones(len(premises), n, dtype=torch.long),
        )

    def model(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=torch.tensor([[1.0, 0.0]] * len(stage.LABELS)))

    scorer = stage.CategoryScorer(device=torch.device("cpu"))
    monkeypatch.setattr(scorer, "_load", lambda: None)
    scorer._tokenizer, scorer._model, scorer._entailment = tokenizer, model, 0
    out = stage.CategoryFinder(SUM_REV, COMP_REV, scorer).classify(
        [("short", "A"), ("cut text", "B")]
    )
    assert steps.TRUNCATED_KEY not in out[0].slug_scores
    assert out[1].slug_scores[steps.TRUNCATED_KEY] == 1
    assert set(steps.class_scores(json.dumps(out[1].slug_scores))) == set(
        stage.LABELS
    )  # not a label


def test_a_blank_text_is_skipped_with_a_reason_and_not_retried(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    steps.record_summary(pair, "body", "blank", SUM_REV, "   \n", "original", 0)
    steps.record_companies(pair, "body", "blank", COMP_REV, [CompanyScore("AAPL", 1, 0.9)], 0.5)
    cli = _load_cli()
    counts = cli.run_category(pair, StubFinder(), NAMES, SUM_REV, COMP_REV, None)
    assert counts["skipped"] == 1 and counts["done"] == 2
    state = pair.execute(
        "SELECT status, error FROM pipeline_step_state WHERE step = 'category' AND item_id = 'blank'"
    ).fetchone()
    assert tuple(state) == ("skipped", "blank text")
    assert (
        pair.execute("SELECT COUNT(*) FROM news_category WHERE item_id = 'blank'").fetchone()[0]
        == 0
    )
    assert "blank" not in [
        i.item_id for i in _items(pair)
    ]  # a skipped item is finished at its revision


def test_a_recut_also_deletes_the_categories_of_its_sub_news_and_step_7_has_its_own_recipe(
    pair: db_module.NewsNlpDatabase,
) -> None:
    cli = _load_cli()
    cut_rev = "units5-test"
    body = next(steps.pending_bodies(pair, steps.STEP_SCOPE, "scope-test"))
    steps.record_scope(pair, body, "scope-test", "multi", {}, "title+800")
    steps.record_cut(
        pair, next(steps.pending_multi_bodies(pair, "scope-test", cut_rev)), cut_rev, ["a sub-news"]
    )
    sub_id = str(
        pair.execute("SELECT id FROM sub_news WHERE extractor_revision = ?", (cut_rev,)).fetchone()[
            0
        ]
    )
    found = {"MSFT": stage.score_of(_scores("other"))}
    steps.record_category(pair, "sub_news", sub_id, REV, found, "m")
    steps.record_category(pair, "body", "b1", REV, found, "m")  # a body: not part of a cut
    for statement in (*cli.RECUT_SQL, *cli.RECATEGORY_SQL):
        assert statement in cli.RERUN_HELP
    assert "<category revision>" in cli.RERUN_HELP
    for statement in cli.RECUT_SQL:
        pair.execute(statement.replace("<cut revision>", cut_rev))
    pair.commit()
    left = pair.execute("SELECT item_kind FROM news_category").fetchall()
    assert [r[0] for r in left] == ["body"]  # the sub-news category went with its cut
    assert (
        pair.execute(
            "SELECT COUNT(*) FROM pipeline_step_state WHERE step = 'category' AND item_kind = 'sub_news'"
        ).fetchone()[0]
        == 0
    )
    for statement in cli.RECATEGORY_SQL:  # to run step 7 again at one revision
        pair.execute(statement.replace("<category revision>", REV))
    pair.commit()
    assert pair.execute("SELECT COUNT(*) FROM news_category").fetchone()[0] == 0
    assert (
        pair.execute("SELECT COUNT(*) FROM pipeline_step_state WHERE step = 'category'").fetchone()[
            0
        ]
        == 0
    )


def test_the_company_name_is_the_one_of_the_companies_step_and_the_revision_follows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        stage.company_name is news_company_stage.company_name
    )  # one definition, not two that drift
    base = stage.revision_of("m@1", SUM_REV, COMP_REV)
    monkeypatch.setattr(stage, "company_name", lambda company: company.upper())
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) != base


def test_a_gpu_fault_stops_the_run_at_once_and_marks_no_item_failed(
    pair: db_module.NewsNlpDatabase,
) -> None:
    """After ``CUDA error: unknown error`` the context of the process is unusable: redoing the batch item by item cannot work and would
    only mark healthy items failed. An out-of-memory error is not this one (a smaller input can survive it)."""
    _seed(pair)
    cli = _load_cli()
    finder = StubFinder(StubScorer(broken=["lawsuit"]))
    finder.scorer.fault = "CUDA error: unknown error"  # type: ignore[attr-defined]
    original = finder.scorer.score

    def faulty(pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
        if any("lawsuit" in t for t, _ in pairs):
            raise RuntimeError(
                "CUDA error: unknown error\nCUDA kernel errors might be asynchronously reported"
            )
        return original(pairs)

    finder.scorer.score = faulty  # type: ignore[method-assign]
    with pytest.raises(cli.RunAborted, match="Rerun the same command"):
        cli.run_category(pair, finder, NAMES, SUM_REV, COMP_REV, None)
    assert (
        pair.execute("SELECT COUNT(*) FROM pipeline_step_state WHERE status = 'failed'").fetchone()[
            0
        ]
        == 0
    )
    assert cli._gpu_lost(RuntimeError("CUDA error: device-side assert triggered"))
    assert not cli._gpu_lost(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not cli._gpu_lost(RuntimeError("CUDA error: out of memory"))  # the driver's wording

    class OutOfMemoryError(RuntimeError):  # the name of torch.cuda.OutOfMemoryError
        pass

    assert not cli._gpu_lost(OutOfMemoryError("CUDA error"))
    assert not cli._gpu_lost(ValueError("CUDA error"))


def test_the_run_branch_of_the_command_checks_both_revisions_warns_and_prints_the_counts(
    pair: db_module.NewsNlpDatabase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The branch the other team runs: the two refusals, the warning for items without a text, the finder and the printed counts (a stub
    finder and the universe of the test, no model)."""
    _seed(pair)
    steps.record_companies(pair, "body", "no-text", COMP_REV, [CompanyScore("AAPL", 1, 0.9)], 0.5)
    cli = _load_cli()
    monkeypatch.setattr(
        cut_stage,
        "companies_of",
        lambda _conn: [
            Company("AAPL", "Apple Inc.", "", ""),
            Company("MSFT", "Microsoft Corporation", "", ""),
        ],
    )
    built: list[tuple[str, str]] = []

    def make_finder(summary_revision: str, companies_revision: str) -> StubFinder:
        built.append(
            (summary_revision, companies_revision)
        )  # the finder is told the inputs: its revision follows them
        return StubFinder()

    monkeypatch.setattr(stage, "CategoryFinder", make_finder)

    class Args:
        status = False
        limit = None
        summary_revision = SUM_REV
        companies_revision = COMP_REV

    cli.main_category(Args(), pair)
    assert built == [(SUM_REV, COMP_REV)]
    out, err = capsys.readouterr()
    assert (
        "1 items with a main company" in err and "no text" in err
    )  # the warning names the revision problem
    printed = json.loads(out[: out.index("}") + 1])
    assert printed["done"] == 2 and printed["items_without_text"] == 1 and printed["failed"] == 0
    Args.summary_revision = "summary-none"
    with pytest.raises(SystemExit, match="step 2 left no text"):
        cli.main_category(Args(), pair)
    Args.summary_revision, Args.companies_revision = SUM_REV, "companies-none"
    with pytest.raises(SystemExit, match="step 3 found no main company"):
        cli.main_category(Args(), pair)


def test_the_category_checkpoint_is_the_same_in_the_config_the_v1_stage_and_the_pipeline() -> None:
    """Three places name it: ``config/pipeline_models.json`` (read by the step), ``pipeline.MODEL_REVISIONS`` (read by ``python -m setup``)
    and the v1 stage's own defaults; a repin that misses one is a different model per command."""
    pinned = stage.pinned_category()
    assert pinned["model"] == pipeline.CATEGORY_MODEL == category_stage.CategoryInference.MODEL_NAME
    assert pinned["revision"] == pipeline.MODEL_REVISIONS[pipeline.CATEGORY_MODEL]
    assert pinned["revision"] == category_stage.CategoryInference.MODEL_REVISIONS[pinned["model"]]


def test_the_run_says_how_much_it_will_do_and_loads_the_model_before_the_bar(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(pair)
    finder = StubFinder()
    _load_cli().run_category(pair, finder, NAMES, SUM_REV, COMP_REV, None)
    assert "2 items (3 pairs) to do" in capsys.readouterr().err
    assert finder.warm_ups == 1
    _load_cli().run_category(
        pair, finder, NAMES, SUM_REV, COMP_REV, None
    )  # nothing left: nothing to load
    assert finder.warm_ups == 1
    assert "nothing pending" in capsys.readouterr().err


def test_the_status_counts_the_pairs_read_cut_at_the_window(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    cut = stage.score_of(_scores("other"))
    cut.slug_scores[steps.TRUNCATED_KEY] = 1
    steps.record_category(
        pair, "body", "b1", REV, {"AAPL": cut, "MSFT": stage.score_of(_scores("other"))}, "m"
    )
    report = steps.status(pair, steps.STEP_CATEGORY, REV)
    assert report["news_category"] == {"rows": 2, "items": 1, "truncated": 1}
    assert report["by_label"] == {"other": 2}  # the reserved key is not a label


# --- the documents say what the code does (as for the companies step) -----------------------------------------------------

DOCS = Path(__file__).resolve().parents[2]


def test_the_taxonomy_doc_lists_the_groups_of_the_code() -> None:
    text = (DOCS / "docs" / "category-taxonomy.md").read_text(encoding="utf-8")
    listed = {
        group: tuple(label.strip("` ") for label in labels.split(","))
        for group, labels in re.findall(r"`(\w+)`: ((?:`\w+`(?:, )?)+)\.", text)
    }
    assert listed == dict(stage.GROUPS)


def test_the_step_doc_table_is_the_definitions_and_groups_of_the_code() -> None:
    text = (DOCS / "docs" / "category_step.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `(\w+)` \| `(\w+)` \| (.+) \|$", text, flags=re.MULTILINE)
    assert [r[0] for r in rows] == list(stage.LABELS)
    assert {r[0]: r[2] for r in rows} == stage.DEFINITIONS
    assert {r[0]: r[1] for r in rows} == stage.GROUP_OF


def test_the_training_scripts_read_the_definitions_of_the_step() -> None:
    spec = importlib.util.spec_from_file_location(
        "category_baseline_for_test", DOCS / "scripts" / "category_baseline_2026_11_13.py"
    )
    assert spec and spec.loader
    baseline = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(DOCS / "scripts"))
    try:
        spec.loader.exec_module(baseline)
    finally:
        sys.path.remove(str(DOCS / "scripts"))
    assert baseline.DEFS == stage.DEFINITIONS and list(stage.LABELS) == baseline.LABELS
    assert {g: tuple(v) for g, v in baseline.GROUPS.items()} == stage.GROUPS


def test_the_sql_of_the_revision_rule_runs_and_the_records_name_the_step(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    steps.record_category(pair, "body", "b1", REV, {"AAPL": stage.score_of(_scores("other"))}, "m")
    text = (DOCS / "docs" / "category_step.md").read_text(encoding="utf-8")
    blocks = [
        b
        for b in re.findall(r"```sql\n(.*?)```", text, flags=re.DOTALL)
        if "pipeline_step_state" in b
    ]
    assert len(blocks) == 1
    block = re.match(r"(?s)(.*)", blocks[0])
    assert block
    statements = [
        part.replace("<companies revision>", COMP_REV)
        for part in re.sub(r"--[^\n]*", "", block.group(1)).split(";")
        if part.strip()
    ]
    assert len(statements) == 2
    finished = [tuple(r) for r in pair.execute(statements[0]).fetchall()]
    assert finished[0][:2] == (REV, 1)  # one finished item at the revision
    assert tuple(pair.execute(statements[1]).fetchone()) == (2,)  # two items have a main company
    spec = (DOCS / ".specify" / "memory" / "SPEC.md").read_text(encoding="utf-8")
    plan = (DOCS / ".specify" / "memory" / "PLAN.md").read_text(encoding="utf-8")
    assert "FR-022" in spec and "never `uncertain`" in spec and "_truncated_chunks" in spec
    assert "D41" in plan and "FR-022" in plan


def test_status_prints_the_revision_the_code_would_write_for_the_inputs_given(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()

    class Bare:
        status = True

    class Given(Bare):
        summary_revision = SUM_REV
        companies_revision = COMP_REV

    cli.main_category(Given(), pair)
    printed = json.loads(capsys.readouterr().out)
    assert printed["code_revision"] == stage.code_revision(SUM_REV, COMP_REV)
    scorer = StubScorer()  # the identity of the pin, so the finder names the revision of this code
    pinned = stage.pinned_category()
    scorer.model_name, scorer.model_revision = pinned["model"], pinned["revision"]
    assert printed["code_revision"] == stage.CategoryFinder(SUM_REV, COMP_REV, scorer).revision
    assert printed["revisions"] == []  # nothing ran: the revision of the code is not one that ran
    cli.main_category(
        Bare(), pair
    )  # without the inputs there is no revision to name, and it says so
    assert "--summary-revision" in json.loads(capsys.readouterr().out)["code_revision"]


def test_the_join_of_the_doc_dates_a_row_by_its_article_and_keeps_the_company_of_the_row(
    pair: db_module.NewsNlpDatabase,
) -> None:
    body = next(steps.pending_bodies(pair, steps.STEP_SCOPE, "scope-test"))
    steps.record_scope(pair, body, "scope-test", "multi", {}, "title+800")
    steps.record_cut(
        pair,
        next(steps.pending_multi_bodies(pair, "scope-test", "units5-test")),
        "units5-test",
        ["a sub-news"],
    )
    sub_id = str(pair.execute("SELECT id FROM sub_news").fetchone()[0])
    steps.record_category(
        pair, "sub_news", sub_id, REV, {"DHR": stage.score_of(_scores("other"))}, "m"
    )
    text = (DOCS / "docs" / "category_step.md").read_text(encoding="utf-8")
    sql = [
        b for b in re.findall(r"```sql\n(.*?)```", text, flags=re.DOTALL) if "JOIN sub_news" in b
    ]
    assert len(sql) == 1
    row = pair.execute(sql[0].replace("<category revision>", REV).rstrip().rstrip(";")).fetchone()
    assert row["ticker"] == "DHR"  # the company of the row, not the article's (MMM, the fixture's)
    assert row["news_date"] == "2023-01-15" and len(row["news_date"]) == 10
    assert pair.execute("SELECT ticker FROM articles WHERE id = 1").fetchone()[0] == "MMM"


@pytest.mark.parametrize(
    "script",
    [
        "category_heldout_2026_11_13",
        "category_baseline_2026_11_13",
        "category_baseline_groups_2026_11_13",
    ],
)
def test_the_scripts_that_score_the_held_out_set_refuse_to_do_it_again(
    script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof that the held-out set was scored is in the tracked record, so the refusal holds on a fresh clone (no git-ignored file needed)."""
    spec = importlib.util.spec_from_file_location(script, DOCS / "scripts" / f"{script}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(DOCS / "scripts"))
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [script])
    with pytest.raises(SystemExit, match="scored once"):
        module.main()


# --- the revision follows its inputs, and finished work can be carried to a new revision ------------------------------------


def test_the_revision_follows_the_summary_and_the_companies_revision() -> None:
    base = stage.revision_of("m@1", "summary-a", "companies-a")
    assert stage.revision_of("m@1", "summary-a", "companies-a") == base
    assert stage.revision_of("m@1", "summary-b", "companies-a") != base
    assert stage.revision_of("m@1", "summary-a", "companies-b") != base


def test_a_step_run_again_upstream_is_new_work_not_done_work(
    pair: db_module.NewsNlpDatabase,
) -> None:
    """The case that failed before: step 2 and 3 run again at new revisions (another text, one more main company) and the category step, at the
    same category revision, found nothing to do and kept the old rows. Now the new inputs are another revision, and the old one stays."""
    _seed(pair)
    cli = _load_cli()
    old = stage.CategoryFinder(SUM_REV, COMP_REV, StubScorer())
    assert cli.run_category(pair, old, NAMES, SUM_REV, COMP_REV, None)["done"] == 2
    steps.record_summary(
        pair, "body", "b1", "summary-new", "earnings_performance: other text", "original", 0
    )
    steps.record_companies(
        pair,
        "body",
        "b1",
        "companies-new",
        [CompanyScore("AAPL", 1, 0.9), CompanyScore("MSFT", 1, 0.9), CompanyScore("META", 1, 0.9)],
        0.5,
    )
    new = stage.CategoryFinder("summary-new", "companies-new", StubScorer())
    assert new.revision != old.revision
    counts = cli.run_category(pair, new, NAMES, "summary-new", "companies-new", None)
    assert counts["done"] == 1 and counts["category_rows"] == 3  # b1 again, with META
    kept = pair.execute(
        "SELECT COUNT(*) FROM news_category WHERE model_revision = ?", (old.revision,)
    ).fetchone()[0]
    assert kept == 3  # the old revision is untouched


def _adopt_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "category_adopt_for_test", DOCS / "scripts" / "category_adopt_revision_2026_11_13.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finished_work_is_carried_to_the_new_revision_without_deleting_the_old(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    cli, adopt = _load_cli(), _adopt_module()
    old = "category-old"

    class OldFinder(StubFinder):
        revision = old

    cli.run_category(pair, OldFinder(), NAMES, SUM_REV, COMP_REV, None)
    before = [
        tuple(r)
        for r in pair.execute("SELECT * FROM news_category WHERE model_revision = ?", (old,))
    ]
    report = adopt.adopt(pair, old, REV, SUM_REV, COMP_REV, execute=False)
    assert report["adoptable"] == 2 and report["finished_at_old"] == 2
    assert (
        pair.execute(
            "SELECT COUNT(*) FROM news_category WHERE model_revision = ?", (REV,)
        ).fetchone()[0]
        == 0
    )  # a report writes nothing
    assert adopt.adopt(pair, old, REV, SUM_REV, COMP_REV, execute=True)["adoptable"] == 2
    assert (
        adopt.adopt(pair, old, REV, SUM_REV, COMP_REV, execute=True)["already_at_new"] == 2
    )  # a repeat adopts nothing more
    kept = [
        tuple(r)
        for r in pair.execute("SELECT * FROM news_category WHERE model_revision = ?", (old,))
    ]
    assert kept == before  # the old revision is as it was
    now = pair.execute(
        "SELECT COUNT(*) FROM news_category WHERE model_revision = ?", (REV,)
    ).fetchone()[0]
    assert now == len(before) == 3
    assert (
        list(steps.pending_category_items(pair, SUM_REV, COMP_REV, REV)) == []
    )  # the next run has nothing left to do
    with pytest.raises(ValueError, match="the same"):
        adopt.adopt(pair, REV, REV, SUM_REV, COMP_REV, execute=True)


def test_an_item_whose_companies_changed_or_whose_text_is_gone_is_not_carried(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _seed(pair)
    cli, adopt = _load_cli(), _adopt_module()

    class OldFinder(StubFinder):
        revision = "category-old"

    cli.run_category(pair, OldFinder(), NAMES, SUM_REV, COMP_REV, None)
    steps.record_companies(  # at the companies revision b1 now has a third main company: it must be done again
        pair, "body", "b1", COMP_REV, [CompanyScore("META", 1, 0.9)], 0.5
    )
    report = adopt.adopt(pair, "category-old", REV, SUM_REV, COMP_REV, execute=True)
    assert report["companies_differ"] == 1 and report["adoptable"] == 1
    assert [i.item_id for i in steps.pending_category_items(pair, SUM_REV, COMP_REV, REV)] == ["b1"]
    other = adopt.adopt(pair, "category-old", "category-x", "summary-none", COMP_REV, execute=False)
    assert other["no_text"] == 2 and other["adoptable"] == 0  # no text at that summary revision


# --- the revision follows the scoring only: a stopped run resumes after an edit that does not change a score -------------------


def test_an_edit_that_does_not_change_a_score_is_not_a_new_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The idempotency of the step must not depend on the code of the module as a whole: a stopped run is resumed with the same command after the
    command, the storage, the logging or a document was edited. (This was the failure: the revision hashed the whole module, so each edit was a new
    revision and a resumed run started from zero.)"""
    base = stage.revision_of("m@1", SUM_REV, COMP_REV)
    monkeypatch.setattr(
        stage.CategoryScorer, "warm_up", lambda self: None
    )  # not part of what a score is
    monkeypatch.setattr(stage.CategoryFinder, "classify", lambda self, pairs: [])
    monkeypatch.setattr(stage, "pinned_category", lambda: {"model": "x", "revision": "y"})
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) == base


def test_an_edit_that_changes_a_score_is_a_new_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    base = stage.revision_of("m@1", SUM_REV, COMP_REV)
    for name in ("PREMISE", "HYPOTHESIS"):
        monkeypatch.setattr(stage, name, getattr(stage, name) + " ")
        assert stage.revision_of("m@1", SUM_REV, COMP_REV) != base
        monkeypatch.undo()
    monkeypatch.setattr(
        stage, "score_of", lambda entailment: stage.CategoryScore("other", 0.0, "other", 0.0, {})
    )
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) != base  # the code that scores is part of it


def test_the_first_run_of_the_real_store_is_continued_not_restarted() -> None:
    """The revision already in the owner's store (the first run, 48,708 items done) is the one this code writes for its inputs: the same command
    resumes it, no row is moved. Its labels were checked against a re-scoring of the stored pairs (0 of 40 differ) before it was named here."""
    pinned = stage.pinned_category()
    identity = f"{pinned['model']}@{pinned['revision']}"
    real = stage.revision_of(identity, "summary-7a9bd2062bc4", "companies-f349536cf701")
    assert real == "category-65f33b55b26b"
    assert (
        stage.scoring_revision(identity, "summary-7a9bd2062bc4", "companies-f349536cf701")
        in stage.LEGACY_REVISIONS
    )
    other = stage.revision_of(identity, "summary-other", "companies-f349536cf701")
    assert (
        other != real and other not in stage.LEGACY_REVISIONS.values()
    )  # only those inputs are the legacy run


# --- what is hashed is what decides a score: formatting and comments are not, a statement is; bookkeeping is not part of it ------


def _with_source(monkeypatch: pytest.MonkeyPatch, function: Any, source: str) -> None:
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda obj: source if obj is function else real(obj))


@pytest.mark.parametrize("function_name", ["score_of", "_read_pair", "company_name"])
def test_a_comment_a_docstring_or_the_formatting_of_the_scoring_code_is_not_a_new_revision(
    function_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = getattr(stage, function_name)
    base = stage.revision_of("m@1", SUM_REV, COMP_REV)
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    node.body = [
        ast.Expr(ast.Constant("another docstring")),
        *node.body[1 if ast.get_docstring(node) else 0 :],
    ]
    reformatted = (
        "# a comment before\n\n\n" + ast.unparse(tree) + "\n\n# and one after\n"
    )  # other layout, other docstring
    _with_source(monkeypatch, function, reformatted)
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) == base


def test_a_statement_of_the_scoring_code_is_a_new_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    base = stage.revision_of("m@1", SUM_REV, COMP_REV)
    source = inspect.getsource(stage.score_of)
    assert "max(" in source
    _with_source(monkeypatch, stage.score_of, source.replace("max(", "min(", 1))
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) != base


def test_the_bookkeeping_of_the_scorer_is_not_part_of_the_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``score`` records which pairs were cut at the window; editing that, or logging inside ``score``, must not restart a run (the reading itself is
    ``_read_pair``, which is hashed)."""
    base = stage.revision_of("m@1", SUM_REV, COMP_REV)
    monkeypatch.setattr(stage.CategoryScorer, "score", lambda self, pairs: [])
    assert stage.revision_of("m@1", SUM_REV, COMP_REV) == base


def test_the_status_counts_the_revision_this_code_writes_not_the_latest_by_time(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the inputs given, ``--status`` is about the revision a run would continue; the latest by time can be a partial revision of older code."""
    _seed(pair)
    mine = stage.code_revision(SUM_REV, COMP_REV)
    found = {"AAPL": stage.score_of(_scores("other"))}
    steps.record_category(pair, "body", "b1", mine, found, "m")
    steps.record_category(
        pair, "body", "b2", "category-older-and-later", found, "m"
    )  # finished after mine
    cli = _load_cli()

    class Given:
        status = True
        summary_revision = SUM_REV
        companies_revision = COMP_REV

    cli.main_category(Given(), pair)
    printed = json.loads(capsys.readouterr().out)
    assert printed["revision"] == mine and printed["news_category"]["items"] == 1
    assert printed["revisions"] == [mine, "category-older-and-later"]


def test_the_verification_script_compares_labels_and_scores() -> None:
    spec = importlib.util.spec_from_file_location(
        "category_verify_for_test", DOCS / "scripts" / "category_verify_revision_2026_11_13.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    same = stage.score_of(_scores("legal_regulatory"))
    differs, worst = module.compare(
        {**same.slug_scores, "_truncated_chunks": 1}, "legal_regulatory", same
    )
    assert (differs, worst) == (False, 0.0)  # the reserved key is not a label
    other = stage.score_of(_scores("other"))
    differs, worst = module.compare(same.slug_scores, "legal_regulatory", other)
    assert differs and worst > 0.5
