"""Step 5, the category (T-168; provisional, the zero-shot model): the label and group of each main company of a text, what is pending,
what is stored and how the step is run. The model is a stub: no network, no GPU."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
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
    finder = stage.CategoryFinder(StubScorer())
    out = finder.classify([("legal_regulatory news", "Apple Inc."), ("nothing", "Apple Inc.")])
    assert [r.label for r in out] == ["legal_regulatory", "other"]
    assert finder.revision.startswith("category-")


def test_the_company_name_loses_its_share_class() -> None:
    assert stage.company_name("Alphabet Inc. (Class A)") == "Alphabet Inc."
    assert stage.company_name("Apple Inc.") == "Apple Inc."


def test_the_revision_follows_the_model_the_definitions_and_the_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = stage.revision_of("m@1")
    assert stage.revision_of("m@1") == base  # stable
    assert stage.revision_of("m@2") != base  # another model, another revision
    monkeypatch.setitem(stage.DEFINITIONS, "other", "something else")
    assert stage.revision_of("m@1") != base
    monkeypatch.undo()
    monkeypatch.setattr(stage, "PREMISE_CHARS", 1000)
    assert stage.revision_of("m@1") != base


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
    out = stage.CategoryFinder(scorer).classify([("short", "A"), ("cut text", "B")])
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
    base = stage.revision_of("m@1")
    monkeypatch.setattr(stage, "company_name", lambda company: company.upper())
    assert stage.revision_of("m@1") != base


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
    monkeypatch.setattr(stage, "CategoryFinder", StubFinder)

    class Args:
        status = False
        limit = None
        summary_revision = SUM_REV
        companies_revision = COMP_REV

    cli.main_category(Args(), pair)
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
