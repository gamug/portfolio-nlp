"""The sector step (T-167; PLAN D39 (4)): the gate, the company votes, the revision, and the stored step. The embedder is a stub: no model,
no network, no GPU."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import news_nlp as db_module
import sector_stage as stage
from news_nlp import steps
from news_nlp.triage import Company
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source
from tests.news_nlp.test_steps import _cli

SCOPE_REV = "scope-nli-test"
COMPANIES = [
    Company("XOM", "Exxon Mobil", "Energy", ""),
    Company("NVDA", "Nvidia", "Information Technology", ""),
    Company("JPM", "JPMorgan Chase", "Financials", ""),
]
CONFIG = {
    "embedder": {"model": "stub", "revision": "0" * 40, "max_length": 512, "batch_size": 8},
    "chunk_words": 5,
    "top_chunks": 1,
    "noise_alpha": 0.0,
    "margin_threshold": 0.2,
    "max_votes_per_company": 3,
}
#: The stub reads three words: oil, chip, bank. A description or a text is the unit vector of their counts (plus a floor).
WORDS = ("oil", "chip", "bank")


def stub_embed(texts: list[str]) -> np.ndarray:
    rows = np.array([[t.lower().count(w) for w in WORDS] + [0.05] for t in texts], dtype=float)
    return np.asarray(rows / np.linalg.norm(rows, axis=1, keepdims=True))


def _load_cli() -> Any:
    return _cli()


def classifier(config: dict[str, Any] | None = None) -> stage.SectorClassifier:
    return stage.SectorClassifier(COMPANIES, config or CONFIG, embed=stub_embed)


# --- the gate ---------------------------------------------------------------------------------------------------------


def scores(**kw: float) -> dict[str, float]:
    base = dict.fromkeys(stage.CLASSES, 0.1)
    base.update({k.replace("c", "", 1) if k.startswith("c") else k: v for k, v in kw.items()})
    return base


def test_a_confident_sector_is_assigned() -> None:
    s = scores(c10=0.8, c45=0.3)
    assert stage.decide(s, {}, 0.2)[:2] == ("assigned", "10")


def test_a_close_call_is_assigned_only_if_the_companies_vote_for_the_same_sector() -> None:
    s = scores(c10=0.55, c45=0.5)
    assert stage.decide(s, {"10": 2, "45": 1}, 0.2)[:2] == ("assigned", "10")
    assert stage.decide(s, {"45": 2}, 0.2)[:2] == ("out_of_scope", "")
    assert stage.decide(s, {}, 0.2)[:2] == ("out_of_scope", "")


def test_a_not_a_sector_class_on_top_is_out_of_scope_even_when_confident_or_voted() -> None:
    s = scores(market_economy=0.9, c10=0.3)
    status, sector, candidate, _ = stage.decide(s, {"10": 5}, 0.2)
    assert (status, sector, candidate) == ("out_of_scope", "", "market_economy")


def test_the_top_vote_is_the_lowest_code_on_a_tie_and_empty_without_votes() -> None:
    assert stage.top_vote({"45": 2, "10": 2, "40": 1}) == "10"
    assert stage.top_vote({}) == ""


def test_a_company_votes_at_most_three_times_for_its_sector() -> None:
    from news_nlp.triage import CompanyResolver  # noqa: PLC0415

    resolver = CompanyResolver(COMPANIES)
    sector_of = {c.ticker: stage.SECTOR_CODES[c.gics_sector] for c in COMPANIES}
    text = "Exxon Mobil said so. Exxon Mobil again. Exxon Mobil once more. Exxon Mobil! Nvidia too."
    votes = stage.sector_votes(resolver, sector_of, text, 3)
    assert votes == {"10": 3, "45": 1}


# --- the classifier ---------------------------------------------------------------------------------------------------


def test_a_text_about_oil_gets_the_energy_sector_and_the_companies_it_names_vote() -> None:
    [r] = classifier().classify([("Oil", "oil oil oil rigs. Exxon Mobil pumps more oil and oil.")])
    assert (r.status, r.sector, r.candidate) == ("assigned", "10", "10")
    assert r.vote_sector == "10" and r.votes == {"10": 1}
    assert set(r.scores) == set(stage.CLASSES)


def test_every_text_gets_one_result_in_order() -> None:
    items = [("a", "oil oil oil"), ("b", "chip chip chip"), ("c", "bank bank bank")]
    out = classifier().classify(items)
    assert [r.candidate for r in out] == ["10", "45", "40"]


def test_the_revision_follows_the_model_the_numbers_the_descriptions_and_the_universe() -> None:
    base = classifier().revision
    assert base == classifier().revision and base.startswith("sector-")
    assert classifier({**CONFIG, "margin_threshold": 0.3}).revision != base
    assert classifier({**CONFIG, "chunk_words": 7}).revision != base
    other = stage.SectorClassifier(COMPANIES[:2], CONFIG, embed=stub_embed)
    assert other.revision != base


def test_the_pinned_config_names_the_model_by_revision() -> None:
    cfg = stage.pinned_config()
    assert cfg["embedder"]["model"] == "BAAI/bge-small-en-v1.5"
    assert len(cfg["embedder"]["revision"]) == 40 and 0 < cfg["margin_threshold"] < 0.1


def test_the_embedder_loads_no_model_until_asked() -> None:
    emb = stage.BgeEmbedder("nobody/nothing", "0" * 40, 512, 8)
    assert emb._model is None


# --- the stored step --------------------------------------------------------------------------------------------------


BODIES = {
    1: ("Oil rally", "oil oil oil. " * 12 + "Exxon Mobil gains."),
    2: ("Chips", "chip chip chip. " * 12 + "Nvidia gains."),
    3: ("Markets", "nothing about anything here. " * 10),
    4: ("Single", "Exxon Mobil alone. " * 10),
}


class Stub:
    """A classifier that never loads a model: ``assigned`` energy for oil, out_of_scope otherwise; ``broken`` makes a title fail."""

    revision = "sector-stub"

    def __init__(self, broken: Sequence[str] = ()) -> None:
        self.calls: list[list[str]] = []
        self.broken = tuple(broken)

    def classify(self, items: Sequence[tuple[str, str]]) -> list[stage.SectorResult]:
        self.calls.append([t for t, _ in items])
        if any(b in t for t, _ in items for b in self.broken):
            raise RuntimeError("CUDA out of memory")
        return [
            stage.SectorResult("assigned", "10", "10", 0.5, "10", {"10": 0.9}, {"10": 2})
            if "oil" in x.lower()
            else stage.SectorResult("out_of_scope", "", "unrelated", 0.01, "", {"10": 0.1}, {})
            for _, x in items
        ]


@pytest.fixture
def pair(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    source = tmp_path / "src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    for i, (title, text) in BODIES.items():
        seed_article(raw, id=i, title=title, body_text=text, fetch_status="ok")
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    for article_id, scope in [(1, "sector"), (2, "sector"), (3, "sector"), (4, "single")]:
        body = next(
            b
            for b in steps.pending_bodies(conn, steps.STEP_SCOPE, SCOPE_REV)
            if b.article_id == article_id
        )
        steps.record_scope(conn, body, SCOPE_REV, scope, {}, "title+800")
    yield conn
    db_module.detach_source(conn)
    conn.close()


def _run(conn: db_module.NewsNlpDatabase, stub: Stub, limit: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = _load_cli().run_sector(conn, stub, SCOPE_REV, limit)
    return counts


def test_only_sector_bodies_are_pending(pair: db_module.NewsNlpDatabase) -> None:
    got = [b.article_id for b in steps.pending_sector_bodies(pair, SCOPE_REV, "sector-stub")]
    assert got == [1, 2, 3]


def test_assigned_and_out_of_scope_bodies_are_stored_with_their_reason(
    pair: db_module.NewsNlpDatabase,
) -> None:
    counts = _run(pair, Stub())
    assert counts == {"done": 3, "failed": 0, "assigned": 1, "out_of_scope": 2}
    rows = pair.execute(
        "SELECT status, sector, candidate, vote_sector FROM body_sector ORDER BY rowid"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("assigned", "10", "10", "10"),
        ("out_of_scope", "", "unrelated", ""),
        ("out_of_scope", "", "unrelated", ""),
    ]


def test_a_rerun_adds_nothing_and_a_new_revision_starts_over(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _run(pair, Stub())
    before = pair.execute("SELECT COUNT(*) FROM body_sector").fetchone()[0]
    assert _run(pair, Stub()) == {"done": 0, "failed": 0}
    assert pair.execute("SELECT COUNT(*) FROM body_sector").fetchone()[0] == before
    newer = Stub()
    newer.revision = "sector-newer"
    assert _run(pair, newer)["done"] == 3
    assert pair.execute("SELECT COUNT(*) FROM body_sector").fetchone()[0] == 2 * before


def test_limit_touches_at_most_n_bodies(pair: db_module.NewsNlpDatabase) -> None:
    assert _run(pair, Stub(), limit=2)["done"] == 2


def test_a_failed_batch_is_redone_body_by_body_and_the_bad_body_is_retried(
    pair: db_module.NewsNlpDatabase,
) -> None:
    stub = Stub(broken=["Chips"])
    counts = _run(pair, stub)
    assert counts["done"] == 2 and counts["failed"] == 1
    again = _run(pair, Stub())
    assert again["done"] == 1 and again["failed"] == 0


def test_the_status_counts_the_bodies_by_status_and_sector(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _run(pair, Stub())
    report = steps.status(pair, steps.STEP_SECTOR, "sector-stub")
    assert report["by_state"] == {"done": 3}
    by_status = report["body_sector"]
    assert isinstance(by_status, dict) and sum(by_status.values()) == 3
    by_sector = report["by_sector"]
    assert isinstance(by_sector, dict) and "out_of_scope" in by_sector


def test_the_command_status_needs_no_model_and_the_options_of_other_steps_are_refused(
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "status_src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    raw.commit()
    raw.close()
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(source))
    argv = ["news_nlp_steps.py", "--step", "sector", "--results-db", str(results_db_path)]
    monkeypatch.setattr(sys, "argv", [*argv, "--status"])
    _load_cli().main()
    report = json.loads(capsys.readouterr().out)
    assert report["step"] == "sector" and report["by_state"] == {}
    monkeypatch.setattr(sys, "argv", [*argv, "--model-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        _load_cli().main()
    monkeypatch.setattr(sys, "argv", [*argv, "--reading", "units5"])
    with pytest.raises(SystemExit):
        _load_cli().main()


# --- the summary of the sector bodies (--step summary --items sector) ---------------------------------------------------


class StubSummarizer:
    """A summariser that never loads a model: texts over ``min_chars`` become ``summary of <n> chars``."""

    revision = "summary-test"
    min_chars = 250
    model_name = "stub-model"
    model_revision = "0" * 40
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def summarize(self, texts: Sequence[str]) -> list[tuple[str, int]]:
        self.calls.append(list(texts))
        return [(f"summary of {len(t)} chars", 2) for t in texts]


def _summarise(
    conn: db_module.NewsNlpDatabase,
    summarizer: StubSummarizer,
    sector_revision: str | None,
    limit: int | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = _load_cli().run_summary(
        conn, summarizer, None, None, ("sector",), limit, sector_revision
    )
    return counts


def test_only_the_bodies_the_sector_step_assigned_at_the_named_revision_are_pending(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _run(pair, Stub())  # article 1 assigned, 2 and 3 out_of_scope
    got = [
        b.article_id
        for b in steps.pending_assigned_sector_bodies(pair, "sector-stub", "summary-test")
    ]
    assert got == [1]
    assert list(steps.pending_assigned_sector_bodies(pair, "sector-other", "summary-test")) == []


def test_a_sector_body_the_step_has_not_reached_waits_and_a_single_body_is_never_read(
    pair: db_module.NewsNlpDatabase,
) -> None:
    _run(pair, Stub(), limit=1)  # only article 1 is classified
    assert [
        b.article_id for b in steps.pending_assigned_sector_bodies(pair, "sector-stub", "x")
    ] == [1]
    # the one body is under the stub's size: stored as it is
    assert _summarise(pair, StubSummarizer(), "sector-stub") == {
        "summary": 0,
        "original": 1,
        "failed": 0,
    }
    rows = pair.execute("SELECT item_kind, text_source FROM news_summary").fetchall()
    assert [tuple(r) for r in rows] == [("body", "original")]


def test_a_long_sector_body_is_summarised_and_a_short_one_is_kept_and_a_rerun_adds_nothing(
    pair: db_module.NewsNlpDatabase,
) -> None:
    class AllAssigned(Stub):
        def classify(self, items: Sequence[tuple[str, str]]) -> list[stage.SectorResult]:
            return [stage.SectorResult("assigned", "45", "45", 0.5, "45", {}, {}) for _ in items]

    _run(pair, AllAssigned())  # the three sector bodies are assigned
    summariser = StubSummarizer()
    first = _summarise(pair, summariser, "sector-stub")
    assert first == {
        "summary": 1,
        "original": 2,
        "failed": 0,
    }  # one body is over the stub's 250 characters
    sources = {r[0] for r in pair.execute("SELECT text_source FROM news_summary")}
    assert sources == {"summary", "original"}
    again = _summarise(pair, StubSummarizer(), "sector-stub")
    assert again == {"summary": 0, "original": 0, "failed": 0}
    assert pair.execute("SELECT COUNT(*) FROM news_summary").fetchone()[0] == 3


def test_a_new_sector_revision_does_not_summarise_a_body_again(
    pair: db_module.NewsNlpDatabase,
) -> None:
    class AllAssigned(Stub):
        def classify(self, items: Sequence[tuple[str, str]]) -> list[stage.SectorResult]:
            return [stage.SectorResult("assigned", "45", "45", 0.5, "45", {}, {}) for _ in items]

    _run(pair, AllAssigned())
    _summarise(pair, StubSummarizer(), "sector-stub")
    newer = AllAssigned()
    newer.revision = "sector-newer"
    _run(pair, newer)
    assert _summarise(pair, StubSummarizer(), "sector-newer") == {
        "summary": 0,
        "original": 0,
        "failed": 0,
    }  # the text is the same, so is its summary


def test_the_command_names_the_sector_revision_and_refuses_what_does_not_apply(
    tmp_path: Path,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "s.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    raw.commit()
    raw.close()
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(source))
    base = ["news_nlp_steps.py", "--step", "summary", "--results-db", str(results_db_path)]

    def refused(*extra: str) -> str:
        monkeypatch.setattr(sys, "argv", [*base, *extra])
        with pytest.raises(SystemExit) as exc:
            _load_cli().main()
        return capsys.readouterr().err + str(exc.value)

    assert "--sector-revision" in refused("--items", "sector")  # it must be named
    assert "assigned no body" in refused("--items", "sector", "--sector-revision", "sector-nope")
    assert "only applies to --items sector" in refused(
        "--items", "single", "--sector-revision", "x"
    )
    assert "only applies to --items sector" in refused("--sector-revision", "x")
    assert "does not apply to --items sector" in refused(
        "--items", "sector", "--sector-revision", "x", "--cut-revision", "units-x"
    )
    assert "does not apply to --items sector" in refused(
        "--items", "sector", "--sector-revision", "x", "--scope-revision", "scope-x"
    )
    monkeypatch.setattr(
        sys, "argv", ["news_nlp_steps.py", "--step", "sector", "--sector-revision", "x"]
    )
    with pytest.raises(SystemExit):
        _load_cli().main()


# --- the manager's review of #179: the revision is of the code as parsed, and the run says what it will do ----------------


def test_a_comment_or_a_docstring_does_not_change_the_revision_but_a_constant_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = stage._module_source()
    base = classifier().revision
    monkeypatch.setattr(stage, "_module_source", lambda: source + "\n# a comment, nothing runs\n")
    assert classifier().revision == base
    monkeypatch.setattr(
        stage,
        "_module_source",
        lambda: source.replace('"""The sector step', '"""A new first line', 1),
    )
    assert classifier().revision == base  # the module docstring
    monkeypatch.setattr(
        stage, "_module_source", lambda: source.replace("STATUS_OUT = ", "STATUS_OUT = 'x' or ", 1)
    )
    assert classifier().revision != base  # code that runs


def test_the_run_says_how_many_bodies_it_will_do_and_loads_the_model_before_the_first_batch(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    class Warm(Stub):
        def __init__(self) -> None:
            super().__init__()
            self.order: list[str] = []

        def prepare(self) -> None:
            self.order.append("prepare")

        def classify(self, items: Sequence[tuple[str, str]]) -> list[stage.SectorResult]:
            self.order.append("classify")
            return super().classify(items)

    warm = Warm()
    _run(pair, warm)
    assert warm.order[0] == "prepare" and warm.order.count("prepare") == 1
    err = capsys.readouterr().err
    assert "pending 3 sector bodies at revision sector-stub" in err
    _run(pair, warm)  # nothing left: no model is loaded for nothing
    assert warm.order.count("prepare") == 1
