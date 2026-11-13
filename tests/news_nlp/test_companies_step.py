"""Step 3, the companies (PLAN D39 (2), (3.1), (3.2), D40; T-164, T-165): NER on the text step 2 left, the matcher turns each span into a
universe ticker, the relevance model scores (text, ticker), every ticker over the threshold is main, none over it goes to review.
The models are stubs: no network, no GPU."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import news_company_stage as stage
import news_nlp as db_module
from news_nlp import steps
from news_nlp.steps import CompanyScore
from news_nlp.triage import Company, CompanyResolver
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source
from tests.news_nlp.test_steps import _cli

SCOPE_REV = "scope-nli-test"
CUT_REV = "units5-test"
SUM_REV = "summary-test"
REV = "companies-test"

UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", "Hardware"),
    Company("MSFT", "Microsoft Corporation", "Information Technology", "Software"),
    Company("META", "Meta Platforms, Inc.", "Communication Services", "Interactive Media"),
    Company("JPM", "JPMorgan Chase & Co.", "Financials", "Banks"),
    Company("MMM", "3M Company", "Industrials", "Conglomerates"),
]


def _load_cli() -> Any:
    return _cli()


class StubFinder:
    """A finder that loads no model: the text is looked up by a marker, ``broken`` markers raise like a lost GPU."""

    revision = REV
    threshold = 0.5

    def __init__(self, table: dict[str, list[CompanyScore]], broken: Sequence[str] = ()) -> None:
        self.table = table
        self.broken = tuple(broken)
        self.calls: list[list[str]] = []

    def find(self, texts: Sequence[str]) -> list[list[CompanyScore]]:
        self.calls.append(list(texts))
        if any(m in t for t in texts for m in self.broken):
            raise RuntimeError("CUDA out of memory")
        return [next((v for k, v in self.table.items() if k in t), []) for t in texts]


# --- the matcher: a NER span is a universe ticker or nothing ---------------------------------------------------------------


def test_a_span_that_names_a_universe_company_gives_its_ticker() -> None:
    resolver = CompanyResolver(UNIVERSE)
    assert resolver.ticker_of_name("Apple Inc.") == "AAPL"
    assert resolver.ticker_of_name("Apple") == "AAPL"  # the first word of the name, capitalised
    assert resolver.ticker_of_name("Microsoft Corporation") == "MSFT"
    assert (
        resolver.ticker_of_name("JPMorgan Chase & Co.") == "JPM"
    )  # "JPMorgan" alone is the matcher gap of T-181, not this step's


def test_a_span_written_as_a_ticker_names_its_company_when_it_cannot_be_a_word() -> None:
    resolver = CompanyResolver(UNIVERSE)
    assert resolver.ticker_of_name("MMM") == "MMM"  # the NER model tags tickers as organisations
    assert resolver.ticker_of_name("AAPL") == "AAPL"
    assert resolver.ticker_of_name("aapl") is None  # only as written in capitals
    assert resolver.ticker_of_name("JP") is None  # under three letters


def test_a_span_that_matches_no_ticker_is_ignored() -> None:
    resolver = CompanyResolver(UNIVERSE)
    assert resolver.ticker_of_name("Federal Reserve") is None
    assert resolver.ticker_of_name("Acme Holdings") is None
    assert resolver.ticker_of_name("") is None


def test_the_spans_of_a_text_are_counted_by_ticker_and_only_organisations_count() -> None:
    resolver = CompanyResolver(UNIVERSE)
    spans = [
        {"entity_type": "ORG", "text": "Apple Inc."},
        {"entity_type": "ORG", "text": "Apple"},
        {
            "entity_type": "ORG",
            "text": "Apple",
        },  # a ticker named three times is one pair with a count of 3
        {"entity_type": "ORG", "text": "Microsoft"},
        {
            "entity_type": "ORG",
            "text": "Federal Reserve",
        },  # no ticker: ignored, not stored as a mention
        {"entity_type": "PER", "text": "Apple"},  # not an organisation
    ]
    assert stage.mentions_of(resolver, spans) == {"AAPL": 3, "MSFT": 1}


def test_the_matcher_over_the_text_adds_the_companies_ner_did_not_tag() -> None:
    resolver = CompanyResolver(UNIVERSE)
    text = "Microsoft said so. Microsoft rose. Apple Inc. fell."
    ner_only = stage.mentions_of(resolver, [{"entity_type": "ORG", "text": "Apple Inc."}])
    assert ner_only == {"AAPL": 1}
    both = stage.mentions_of(resolver, [{"entity_type": "ORG", "text": "Apple Inc."}], text)
    assert both == {
        "AAPL": 1,
        "MSFT": 2,
    }  # a ticker both name keeps the larger count: one mention is not counted twice


# --- the rule of the main company ------------------------------------------------------------------------------------------


def test_a_company_is_main_at_the_threshold_and_over_it_not_under_it() -> None:
    assert steps.is_main(0.5, 0.5) is True
    assert steps.is_main(0.9, 0.5) is True
    assert steps.is_main(0.4999, 0.5) is False


def test_the_revision_follows_the_models_the_threshold_the_universe_and_the_code() -> None:
    base = stage.revision_of("ner@1", "rel@1", 0.5, UNIVERSE)
    assert base != stage.revision_of(
        "ner@1", "rel@1", 0.5, UNIVERSE, "ner+text"
    )  # the candidates are part of it
    assert base.startswith("companies-") and len(base) == len("companies-") + 12
    assert base == stage.revision_of("ner@1", "rel@1", 0.5, UNIVERSE)
    assert base != stage.revision_of("ner@2", "rel@1", 0.5, UNIVERSE)
    assert base != stage.revision_of("ner@1", "rel@2", 0.5, UNIVERSE)
    assert base != stage.revision_of(
        "ner@1", "rel@1", 0.6, UNIVERSE
    )  # D39 (vi): a new threshold is a new revision
    assert base != stage.revision_of("ner@1", "rel@1", 0.5, UNIVERSE[:-1])


def test_the_hypothesis_is_the_one_the_relevance_model_was_trained_on() -> None:
    # scripts/final_models_2026_11_10.py: HYPOTHESIS and company_name, max_length 320
    assert stage.HYPOTHESIS.format("Apple Inc.") == "This text is mainly about Apple Inc.."
    assert stage.company_name("Alphabet Inc. (Class A)") == "Alphabet Inc."
    assert stage.MAX_LENGTH == 320


# --- what is pending -------------------------------------------------------------------------------------------------------


@pytest.fixture
def pair(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    source = tmp_path / "src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    for i, title, text in [
        (1, "Single one", "Apple reported results. " * 30),
        (2, "Single two", "Microsoft reported results. " * 30),
        (3, "Multi", "Round-up. " * 40),
        (4, "Out of scope", "Nothing here. " * 40),
    ]:
        seed_article(raw, id=i, title=title, body_text=text, fetch_status="ok")
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    for article_id, scope in [(1, "single"), (2, "single"), (3, "multi"), (4, "out_of_scope")]:
        body = next(
            b
            for b in steps.pending_bodies(conn, steps.STEP_SCOPE, SCOPE_REV)
            if b.article_id == article_id
        )
        steps.record_scope(conn, body, SCOPE_REV, scope, {}, "title+800")
    multi = next(steps.pending_multi_bodies(conn, SCOPE_REV, CUT_REV))
    steps.record_cut(conn, multi, CUT_REV, ["Apple sub-news.", "Microsoft sub-news."])
    steps.record_cut(conn, multi, "units-old", ["Old sub-news."])
    # step 2 left a text for every item, at one summary revision; another revision has one more
    for body in steps.pending_single_bodies(conn, SCOPE_REV, SUM_REV):
        steps.record_summary(
            conn, steps.ITEM_BODY, body.body_hash, SUM_REV, f"summary of {body.title}", "summary", 2
        )
    for sub in steps.pending_sub_news(conn, CUT_REV, SUM_REV):
        steps.record_summary(
            conn, steps.ITEM_SUB_NEWS, str(sub.sub_news_id), SUM_REV, sub.text, "original", 0
        )
    for sub in steps.pending_sub_news(conn, "units-old", SUM_REV):
        steps.record_summary(
            conn, steps.ITEM_SUB_NEWS, str(sub.sub_news_id), SUM_REV, sub.text, "original", 0
        )
    steps.record_summary(
        conn, steps.ITEM_BODY, "orphan", "summary-other", "elsewhere", "summary", 1
    )
    yield conn
    db_module.detach_source(conn)
    conn.close()


def _texts(items: Sequence[steps.TextItem]) -> list[str]:
    return [i.text for i in items]


def test_the_texts_of_step_2_are_read_by_the_summary_and_scope_and_cut_revisions_named(
    pair: db_module.NewsNlpDatabase,
) -> None:
    got = list(
        steps.pending_texts(
            pair,
            SUM_REV,
            REV,
            scope_revision=SCOPE_REV,
            cut_revision=CUT_REV,
            items=("single", "sub_news"),
        )
    )
    assert _texts(got) == [
        "summary of Single one",
        "summary of Single two",
        "Apple sub-news.",
        "Microsoft sub-news.",
    ]
    assert [i.kind for i in got] == ["body", "body", "sub_news", "sub_news"]


def test_each_kind_alone_and_the_other_revisions_are_not_read(
    pair: db_module.NewsNlpDatabase,
) -> None:
    single = list(
        steps.pending_texts(
            pair, SUM_REV, REV, scope_revision=SCOPE_REV, cut_revision=None, items=("single",)
        )
    )
    assert [i.kind for i in single] == ["body", "body"]
    sub = list(
        steps.pending_texts(
            pair, SUM_REV, REV, scope_revision=None, cut_revision=CUT_REV, items=("sub_news",)
        )
    )
    assert _texts(sub) == ["Apple sub-news.", "Microsoft sub-news."]
    old = list(
        steps.pending_texts(
            pair, SUM_REV, REV, scope_revision=None, cut_revision="units-old", items=("sub_news",)
        )
    )
    assert _texts(old) == ["Old sub-news."]
    none = list(
        steps.pending_texts(
            pair,
            "summary-other",
            REV,
            scope_revision=SCOPE_REV,
            cut_revision=None,
            items=("single",),
        )
    )
    assert (
        none == []
    )  # the summary revision names which texts are read: the orphan is not a single body of that scope
    assert (
        len(
            list(
                steps.pending_texts(
                    pair,
                    SUM_REV,
                    REV,
                    scope_revision=SCOPE_REV,
                    cut_revision=CUT_REV,
                    items=("single", "sub_news"),
                    limit=3,
                )
            )
        )
        == 3
    )


# --- what is stored --------------------------------------------------------------------------------------------------------


def _rows(conn: db_module.NewsNlpDatabase) -> list[tuple[Any, ...]]:
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT item_kind, item_id, ticker, relevance_score, is_main, mention_count FROM news_company ORDER BY item_id, ticker"
        )
    ]


def test_every_ticker_over_the_threshold_is_main_and_the_others_are_mentions(
    pair: db_module.NewsNlpDatabase,
) -> None:
    found = [
        CompanyScore("AAPL", 3, 0.91),
        CompanyScore("MSFT", 1, 0.62),
        CompanyScore("META", 2, 0.12),
    ]
    state = steps.record_companies(pair, steps.ITEM_SUB_NEWS, "7", REV, found, 0.5)
    assert state == "done"
    assert _rows(pair) == [
        ("sub_news", "7", "AAPL", 0.91, 1, 3),
        ("sub_news", "7", "META", 0.12, 0, 2),
        ("sub_news", "7", "MSFT", 0.62, 1, 1),
    ]  # a news with two main companies has two main rows, the third is a mention


def test_no_company_over_the_threshold_gives_no_main_row_and_the_review_state(
    pair: db_module.NewsNlpDatabase,
) -> None:
    state = steps.record_companies(
        pair, steps.ITEM_BODY, "b1", REV, [CompanyScore("AAPL", 1, 0.2)], 0.5
    )
    assert state == "review"
    assert _rows(pair) == [("body", "b1", "AAPL", 0.2, 0, 1)]
    row = pair.execute(
        "SELECT status, error FROM pipeline_step_state WHERE step = 'companies' AND item_id = 'b1'"
    ).fetchone()
    assert row[0] == "review" and "over the threshold" in row[1] and "AAPL" in row[1]


def test_no_company_found_is_review_with_its_reason(pair: db_module.NewsNlpDatabase) -> None:
    assert steps.record_companies(pair, steps.ITEM_BODY, "b2", REV, [], 0.5) == "review"
    assert _rows(pair) == []
    error = pair.execute(
        "SELECT error FROM pipeline_step_state WHERE step = 'companies' AND item_id = 'b2'"
    ).fetchone()[0]
    assert "no company" in error


def test_a_repeat_adds_nothing_and_a_new_revision_starts_over(
    pair: db_module.NewsNlpDatabase,
) -> None:
    found = [CompanyScore("AAPL", 3, 0.91)]
    steps.record_companies(pair, steps.ITEM_SUB_NEWS, "7", REV, found, 0.5)
    steps.record_companies(pair, steps.ITEM_SUB_NEWS, "7", REV, found, 0.5)
    assert (
        len(_rows(pair)) == 1
    )  # one row per (text, ticker): a ticker named three times is one row
    steps.record_companies(pair, steps.ITEM_SUB_NEWS, "7", "companies-new", found, 0.4)
    assert pair.execute("SELECT COUNT(*) FROM news_company").fetchone()[0] == 2  # the old row stays


def test_review_and_done_items_are_not_pending_again_but_failed_ones_are(
    pair: db_module.NewsNlpDatabase,
) -> None:
    items = list(
        steps.pending_texts(
            pair,
            SUM_REV,
            REV,
            scope_revision=SCOPE_REV,
            cut_revision=CUT_REV,
            items=("single", "sub_news"),
        )
    )
    steps.record_companies(
        pair, items[0].kind, items[0].item_id, REV, [CompanyScore("AAPL", 1, 0.9)], 0.5
    )
    steps.record_companies(
        pair, items[1].kind, items[1].item_id, REV, [], 0.5
    )  # review: waits for the owner
    steps.record_item_failure(
        pair, items[2].kind, items[2].item_id, steps.STEP_COMPANIES, REV, "RuntimeError: x"
    )
    left = list(
        steps.pending_texts(
            pair,
            SUM_REV,
            REV,
            scope_revision=SCOPE_REV,
            cut_revision=CUT_REV,
            items=("single", "sub_news"),
        )
    )
    assert [i.item_id for i in left] == [items[2].item_id, items[3].item_id]


def test_the_status_counts_review_apart_and_the_rows(pair: db_module.NewsNlpDatabase) -> None:
    steps.record_companies(
        pair,
        steps.ITEM_BODY,
        "b1",
        REV,
        [CompanyScore("AAPL", 1, 0.9), CompanyScore("MSFT", 1, 0.1)],
        0.5,
    )
    steps.record_companies(pair, steps.ITEM_BODY, "b2", REV, [], 0.5)
    report = steps.status(pair, steps.STEP_COMPANIES, REV)
    assert report["by_state"] == {"done": 1, "review": 1}
    assert report["news_company"] == {"rows": 2, "main": 1, "items": 1, "items_with_main": 1}


# --- the run ---------------------------------------------------------------------------------------------------------------


def test_a_run_stores_each_item_and_returns_the_counts(pair: db_module.NewsNlpDatabase) -> None:
    cli = _load_cli()
    finder = StubFinder(
        {
            "Single one": [CompanyScore("AAPL", 2, 0.9)],
            "Single two": [CompanyScore("MSFT", 1, 0.2)],
            "Apple sub": [CompanyScore("AAPL", 1, 0.8), CompanyScore("MSFT", 1, 0.1)],
        }
    )
    counts = cli.run_companies(
        pair, finder, SUM_REV, SCOPE_REV, CUT_REV, ("single", "sub_news"), None
    )
    assert counts == {"done": 2, "review": 2, "failed": 0, "main_rows": 2}
    again = cli.run_companies(
        pair, finder, SUM_REV, SCOPE_REV, CUT_REV, ("single", "sub_news"), None
    )
    assert again == {
        "done": 0,
        "review": 0,
        "failed": 0,
        "main_rows": 0,
    }  # nothing pending: nothing changes
    assert len(_rows(pair)) == 4


def test_limit_bounds_the_items(pair: db_module.NewsNlpDatabase) -> None:
    cli = _load_cli()
    finder = StubFinder({})
    counts = cli.run_companies(pair, finder, SUM_REV, SCOPE_REV, CUT_REV, ("single", "sub_news"), 3)
    assert counts["done"] + counts["review"] == 3


def test_a_batch_that_fails_is_redone_item_by_item_so_one_bad_text_fails_alone(
    pair: db_module.NewsNlpDatabase,
) -> None:
    cli = _load_cli()
    finder = StubFinder({"Apple sub": [CompanyScore("AAPL", 1, 0.8)]}, broken=["Microsoft sub"])
    counts = cli.run_companies(pair, finder, SUM_REV, SCOPE_REV, CUT_REV, ("sub_news",), None)
    assert counts == {"done": 1, "review": 0, "failed": 1, "main_rows": 1}
    error = pair.execute(
        "SELECT error FROM pipeline_step_state WHERE status = 'failed' AND step = 'companies'"
    ).fetchone()[0]
    assert error == "RuntimeError: CUDA out of memory"


def test_batches_that_all_fail_in_a_row_stop_the_run(
    pair: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "COMPANY_BATCH", 1)
    finder = StubFinder({}, broken=["sub-news", "summary of"])
    with pytest.raises(cli.RunAborted):
        cli.run_companies(pair, finder, SUM_REV, SCOPE_REV, CUT_REV, ("single", "sub_news"), None)


# --- the command ------------------------------------------------------------------------------------------------------------


def test_the_companies_step_needs_the_summary_revision_a_threshold_and_the_candidates_to_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    parser = cli.build_parser()
    base = ["--step", "companies"]
    full = ["--summary-revision", "summary-x", "--threshold", "0.5", "--candidates", "ner"]
    cli.check_options(parser, parser.parse_args(base + full))  # all three: accepted
    for missing, flag in (("--summary-revision", 0), ("--threshold", 2), ("--candidates", 4)):
        args = base + full[:flag] + full[flag + 2 :]
        with pytest.raises(SystemExit):
            cli.check_options(parser, parser.parse_args(args))
        assert missing in capsys.readouterr().err
    # --status needs none of them: it loads no model
    cli.check_options(parser, parser.parse_args([*base, "--status"]))


def test_the_threshold_must_be_a_probability_and_the_options_belong_to_their_step() -> None:
    cli = _load_cli()
    parser = cli.build_parser()
    full = ["--summary-revision", "s", "--threshold", "0.5", "--candidates", "ner"]
    for bad in ("0", "1.2", "-0.1"):
        with pytest.raises(SystemExit):
            cli.check_options(
                parser,
                parser.parse_args(
                    [
                        "--step",
                        "companies",
                        "--summary-revision",
                        "s",
                        "--threshold",
                        bad,
                        "--candidates",
                        "ner",
                    ]
                ),
            )
    for step in ("scope", "cut", "summary"):
        for flag, value in (
            ("--threshold", "0.5"),
            ("--summary-revision", "s"),
            ("--candidates", "ner"),
        ):
            with pytest.raises(SystemExit):
                cli.check_options(parser, parser.parse_args(["--step", step, flag, value]))
    with pytest.raises(SystemExit):  # the models are the pins, not a folder
        cli.check_options(
            parser, parser.parse_args(["--step", "companies", *full, "--model-dir", "x"])
        )


def test_the_status_of_the_companies_step_loads_no_model(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    steps.record_companies(pair, steps.ITEM_BODY, "b1", REV, [CompanyScore("AAPL", 1, 0.9)], 0.5)

    class Args:
        status = True

    cli.main_companies(Args(), pair)
    out = json.loads(capsys.readouterr().out)
    assert out["step"] == "companies" and out["by_state"] == {"done": 1}
    assert out["revisions"] == [REV]


# --- the finder: NER, the matcher and the relevance model together --------------------------------------------------------


class StubSpans:
    model_name = "stub-ner"
    model_revision = "0" * 40

    def spans(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        return [
            [
                {"entity_type": "ORG", "text": w}
                for w in ("Apple", "Apple", "Microsoft", "Federal Reserve")
                if w in t
            ]
            for t in texts
        ]


class StubScorer:
    repo = "stub/relevance"
    revision = "1" * 40
    device = "cpu"

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.pairs.extend(pairs)
        return [0.9 if name.startswith("Apple") else 0.1 for _, name in pairs]


def test_the_finder_scores_each_ticker_a_text_names_once_with_its_company_name() -> None:
    scorer = StubScorer()
    finder = stage.CompanyFinder(UNIVERSE, 0.5, "ner", spans=StubSpans(), scorer=scorer)
    got = finder.find(
        ["Apple and Microsoft said so.", "Nothing is named.", "The Federal Reserve met."]
    )
    assert got[0] == [CompanyScore("AAPL", 2, 0.9), CompanyScore("MSFT", 1, 0.1)]
    assert (
        got[1] == [] and got[2] == []
    )  # no ticker named, or a span that matches none: no pair is scored
    assert scorer.pairs == [
        ("Apple and Microsoft said so.", "Apple Inc."),
        ("Apple and Microsoft said so.", "Microsoft Corporation"),
    ]
    assert finder.revision.startswith("companies-")
    assert (
        finder.revision
        != stage.CompanyFinder(
            UNIVERSE, 0.6, "ner", spans=StubSpans(), scorer=StubScorer()
        ).revision
    )


def test_with_the_text_matches_the_finder_scores_a_company_ner_did_not_tag() -> None:
    scorer = StubScorer()
    finder = stage.CompanyFinder(UNIVERSE, 0.5, "ner+text", spans=StubSpans(), scorer=scorer)
    got = finder.find(
        ["3M raised its guidance. 3M said so."]
    )  # NER tags nothing in this text (it does not tag "3M")
    assert got == [[CompanyScore("MMM", 2, 0.1)]]
    ner_only = stage.CompanyFinder(UNIVERSE, 0.5, "ner", spans=StubSpans(), scorer=StubScorer())
    assert ner_only.find(["3M raised its guidance. 3M said so."]) == [[]]
    assert finder.revision != ner_only.revision
