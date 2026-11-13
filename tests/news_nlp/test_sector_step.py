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


def test_the_top_vote_is_the_sector_with_most_votes_and_empty_on_a_tie_or_without_votes() -> None:
    assert stage.top_vote({"45": 3, "10": 2, "40": 1}) == "45"
    assert stage.top_vote({"45": 2, "10": 2, "40": 1}) == ""  # a tie is not agreement
    assert stage.top_vote({}) == ""


def test_a_split_vote_does_not_back_a_close_call_even_when_its_sector_has_the_lower_code() -> None:
    s = scores(c10=0.55, c45=0.5)  # a close call for energy
    assert stage.decide(s, {"10": 1, "45": 1}, 0.2)[:2] == (
        "out_of_scope",
        "",
    )  # was assigned while the lowest code won ties
    assert stage.decide(s, {"10": 2, "45": 1}, 0.2)[:2] == ("assigned", "10")
    assert stage.decide(s, {"10": 1}, 0.2)[:2] == (
        "assigned",
        "10",
    )  # one vote with no rival is still agreement


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


# --- the manager's second review of #179: the real embedder is tested without a model --------------------------------------


class _FakeBatch(dict):  # type: ignore[type-arg]
    def to(self, device: object) -> _FakeBatch:
        return self


def _fake_transformers(
    monkeypatch: pytest.MonkeyPatch, lengths: dict[str, int], calls: list[dict[str, object]]
) -> None:
    """A tokenizer that gives each text ``lengths[text]`` tokens (cut at ``max_length``) and a model whose first row is not its mean."""
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    class Tok:
        def __call__(self, texts: list[str], **kw: object) -> _FakeBatch:
            width = int(kw["max_length"])  # type: ignore[call-overload]
            n = [min(lengths.get(t, 3), width) for t in texts]
            mask = torch.zeros(len(texts), max(n))
            for i, k in enumerate(n):
                mask[i, :k] = 1
            return _FakeBatch(attention_mask=mask, input_ids=mask.long())

    class Model:
        def to(self, device: object) -> Model:
            return self

        def eval(self) -> Model:
            return self

        def __call__(self, **batch: torch.Tensor) -> object:
            rows = batch["attention_mask"].shape[0]
            hidden = torch.zeros(rows, batch["attention_mask"].shape[1], 3)
            hidden[:, 0] = torch.tensor([3.0, 4.0, 0.0])  # the CLS row: norm 5
            hidden[:, 1:] = torch.tensor(
                [0.0, 0.0, 9.0]
            )  # a mean over the tokens would not give the CLS row
            return type("Out", (), {"last_hidden_state": hidden})()

    def tok_from(name: str, revision: str) -> Tok:
        calls.append({"what": "tokenizer", "name": name, "revision": revision})
        return Tok()

    def model_from(name: str, revision: str) -> Model:
        calls.append({"what": "model", "name": name, "revision": revision})
        return Model()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(tok_from))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(model_from))


def test_the_embedder_loads_the_pinned_revision_pools_the_cls_row_and_returns_unit_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _fake_transformers(monkeypatch, {}, calls)
    emb = stage.BgeEmbedder("org/embedder", "a" * 40, 512, 2)
    out = emb(["one", "two", "three"])  # three texts, batches of two
    assert out.shape == (3, 3)
    assert np.allclose(out, [[0.6, 0.8, 0.0]] * 3)  # the CLS row (3, 4, 0) normalised, not the mean
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert {c["what"] for c in calls} == {"tokenizer", "model"}
    assert all(c["name"] == "org/embedder" and c["revision"] == "a" * 40 for c in calls)
    emb(["again"])
    assert len(calls) == 2  # loaded once, not per call


def test_a_chunk_that_fills_the_window_is_marked_per_text_and_per_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_transformers(monkeypatch, {"T. long words here": 9999, "second": 9999}, [])
    emb = stage.BgeEmbedder("org/embedder", "a" * 40, 8, 4)
    emb(["short", "second", "third"])
    assert emb.last_truncated.tolist() == [False, True, False]  # in the order of the texts
    clf = stage.SectorClassifier(COMPANIES, CONFIG, embed=emb)
    cut, fine = clf.classify([("T", "long words here"), ("U", "short words")])
    assert (cut.truncated_chunks, fine.truncated_chunks) == (1, 0)
    assert (
        classifier().classify([("T", "oil")])[0].truncated_chunks == 0
    )  # an embedder that does not say


def test_the_cut_chunks_are_stored_with_the_body_and_counted_by_the_run(
    pair: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    class Cut(Stub):
        def classify(self, items: Sequence[tuple[str, str]]) -> list[stage.SectorResult]:
            return [
                stage.SectorResult(
                    "assigned", "10", "10", 0.5, "10", {"10": 0.9}, {}, 3 if "Oil" in t else 0
                )
                for t, _ in items
            ]

    counts = _run(pair, Cut())
    assert counts["truncated_chunks"] == 3
    assert "3 chunk(s) were cut" in capsys.readouterr().err
    rows = [json.loads(r[0]) for r in pair.execute("SELECT scores FROM body_sector ORDER BY rowid")]
    assert [steps.TRUNCATED_KEY in r for r in rows] == [
        True,
        False,
        False,
    ]  # only the body that had one
    assert rows[0][steps.TRUNCATED_KEY] == 3
    clean = _run(pair, Stub())  # a new revision, nothing cut
    assert "truncated_chunks" not in clean


# --- the manager's fourth review of #179: the read side of the reserved key, and the tie rule end to end ------------------


def test_the_scores_of_a_stored_row_are_read_through_class_scores_which_drops_the_reserved_key() -> (
    None
):
    raw = {"10": 0.61, "45": 0.55, "market_economy": 0.4, steps.TRUNCATED_KEY: 2}
    assert max(raw, key=raw.get) == steps.TRUNCATED_KEY  # type: ignore[arg-type]  # the trap: a count beats any score
    clean = steps.class_scores(raw)
    assert set(clean) == {"10", "45", "market_economy"} and max(clean, key=clean.get) == "10"  # type: ignore[arg-type]
    assert steps.class_scores(json.dumps(raw)) == clean  # the stored text, as the table holds it
    assert steps.truncated_chunks(json.dumps(raw)) == 2 and steps.truncated_chunks({"10": 0.6}) == 0


def test_a_stored_row_with_cut_chunks_reads_back_as_classes_and_a_count(
    pair: db_module.NewsNlpDatabase,
) -> None:
    class Cut(Stub):
        def classify(self, items: Sequence[tuple[str, str]]) -> list[stage.SectorResult]:
            return [
                stage.SectorResult(
                    "assigned",
                    "10",
                    "10",
                    0.5,
                    "10",
                    {"10": 0.9, "45": 0.2},
                    {},
                    3 if "Oil" in t else 0,
                )
                for t, _ in items
            ]

    _run(pair, Cut())
    rows = [str(r[0]) for r in pair.execute("SELECT scores FROM body_sector ORDER BY rowid")]
    assert [steps.truncated_chunks(r) for r in rows] == [3, 0, 0]
    assert all(
        set(steps.class_scores(r)) == {"10", "45"} for r in rows
    )  # no reserved key reaches a reader


def test_a_text_that_names_two_companies_of_different_sectors_once_each_is_out_of_scope_with_no_vote_sector() -> (
    None
):
    clf = classifier()
    split = "oil chip. Exxon Mobil and Nvidia are both mentioned once."
    [tied] = clf.classify([("Mixed", split)])
    assert (
        tied.margin < 0.2
    )  # below CONFIG margin_threshold: a close call, only the vote could let it through
    assert tied.votes == {"10": 1, "45": 1}
    assert (tied.status, tied.vote_sector) == ("out_of_scope", "")  # a tie is not agreement
    [backed] = clf.classify([("Mixed", split + " Exxon Mobil again.")])
    assert backed.votes == {"10": 2, "45": 1} and backed.vote_sector == "10"
    assert (backed.status, backed.sector) == (
        "assigned",
        "10",
    )  # one company named twice makes a top


def test_the_helpers_take_the_scores_column_and_refuse_a_database_row_with_a_message_that_says_so() -> (
    None
):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (scores TEXT)")
    conn.execute("INSERT INTO t VALUES (?)", (json.dumps({"10": 0.6, steps.TRUNCATED_KEY: 2}),))
    row = conn.execute("SELECT scores FROM t").fetchone()
    assert (
        steps.class_scores(row["scores"]) == {"10": 0.6}
        and steps.truncated_chunks(row["scores"]) == 2
    )
    for helper in (steps.class_scores, steps.truncated_chunks):
        with pytest.raises(TypeError, match=r"scores column.*row\['scores'\]"):
            helper(row)
        with pytest.raises(TypeError):
            helper(None)


# --- the manager's sixth review of #179: the run stops when batch after batch fails completely -----------------------------


class Broken:
    """A classifier that fails every batch, as a lost GPU or a broken checkpoint would."""

    revision = "sector-broken"

    def classify(self, items: Sequence[tuple[str, str]]) -> list[stage.SectorResult]:
        raise RuntimeError("CUDA error")


@pytest.fixture
def many_sector_bodies(
    tmp_path: Path, results_db_path: Path
) -> Iterator[db_module.NewsNlpDatabase]:
    """More sector bodies than three batches hold, all scoped ``sector`` at one scope revision."""
    source = tmp_path / "many.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    for i in range(1, 80):
        seed_article(
            raw, id=i, title=f"Story {i}", body_text=f"Body number {i}. " * 5, fetch_status="ok"
        )
    raw.commit()
    raw.close()
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    for body in list(steps.pending_bodies(conn, steps.STEP_SCOPE, SCOPE_REV)):
        steps.record_scope(conn, body, SCOPE_REV, "sector", {}, "title+800")
    yield conn
    db_module.detach_source(conn)
    conn.close()


def test_the_run_stops_when_batch_after_batch_fails_completely_and_the_failed_bodies_wait_for_the_next_run(
    many_sector_bodies: db_module.NewsNlpDatabase,
) -> None:
    cli = _load_cli()
    with pytest.raises(cli.RunAborted, match="in a row"):
        cli.run_sector(many_sector_bodies, Broken(), SCOPE_REV, None)
    report = steps.status(many_sector_bodies, steps.STEP_SECTOR, "sector-broken")
    assert report["by_state"] == {"failed": 3 * cli.BATCH}  # three batches, and not one more
    examples = report["failed_examples"]
    assert isinstance(examples, list) and "CUDA error" in examples[0]["error"]
    assert (
        many_sector_bodies.execute("SELECT COUNT(*) FROM body_sector").fetchone()[0] == 0
    )  # nothing stored as a result
    retried = steps.pending_sector_bodies(many_sector_bodies, SCOPE_REV, "sector-broken")
    assert (
        len(list(retried)) == 79
    )  # failed bodies are retried by the next run: none is marked done


def test_the_command_exits_with_stopped_when_the_run_aborts(
    many_sector_bodies: db_module.NewsNlpDatabase,
    results_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import cut_stage  # noqa: PLC0415 -- the command reads the universe through it

    monkeypatch.setattr(cut_stage, "companies_of", lambda conn: COMPANIES)
    monkeypatch.setattr(stage, "SectorClassifier", lambda companies: Broken())
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv(
        "SOURCE_DATABASE_URL",
        str(many_sector_bodies.execute("PRAGMA database_list").fetchall()[-1][2]),
    )
    monkeypatch.setattr(
        sys, "argv", ["news_nlp_steps.py", "--step", "sector", "--results-db", str(results_db_path)]
    )
    with pytest.raises(SystemExit) as exc:
        _load_cli().main()
    assert str(exc.value).startswith("stopped: ") and "in a row" in str(exc.value)
    assert json.loads(capsys.readouterr().out)["by_state"] == {
        "failed": 3 * _load_cli().BATCH
    }  # the status is printed first
