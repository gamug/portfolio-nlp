"""calibration_judge.py / calibration_sample.py: the T-141 truth-label judge and the seeded,
stratified, half-split calibration sample. Hermetic: the agent is a fake."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

import news_nlp as db_module
from experiment import Stage
from news_nlp.eval import calibration_judge as cj
from news_nlp.eval import calibration_sample as cs
from news_nlp.eval.calibration import ConfirmedArticle, cell_of
from news_nlp.eval.config import JUDGE_TEMPERATURE
from news_nlp.eval.judges import JUDGES, JudgeAgent
from news_nlp.eval.sampling import STAGES, EvalItem
from news_nlp.eval.store import create_eval_run, record_inference, record_verdict
from news_nlp.eval.verdicts import VERDICT_MODELS
from news_nlp.taxonomy import CATEGORY_GROUP_FLOOR, CATEGORY_SLUGS

#: Pinned so that a change to the frozen wording is a deliberate, reviewed act that also
#: re-budgets the judge calls (TASKS T-141): update this hash in the same commit, on purpose.
FROZEN_PROMPT_SHA256 = "de1cedad523f8868a445b06650b18f31f0bbd573781edf7af977045be533464d"


class FakeAgent:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else "(exhausted)"


def _item(article_id: int = 1, prediction: dict | None = None) -> EvalItem:
    return EvalItem(
        article_id=article_id,
        bucket="mergers_acquisitions|2",
        stratum_population=100,
        title="Acme buys Beta",
        body_text="Acme agreed to acquire Beta for $1bn.",
        prediction=prediction or {"label": "other", "score": 0.55, "distribution": {}},
    )


def _article(i: int, slug_index: int, score: float, group: float = 0.9) -> ConfirmedArticle:
    leaves = [0.05] * len(CATEGORY_SLUGS)
    leaves[slug_index] = score
    return ConfirmedArticle(i, "other", group, tuple(leaves))


# --- the judge ---------------------------------------------------------------


def test_prompt_is_frozen() -> None:
    assert (
        cj.prompt_hash()
        == hashlib.sha256(cj.load_prompt(cj.PROMPT_NAME).encode("utf-8")).hexdigest()
    )
    assert cj.prompt_hash() == FROZEN_PROMPT_SHA256


def test_prompt_lists_every_label_and_hides_the_classifier_output() -> None:
    text = cj.load_prompt(cj.PROMPT_NAME)
    for slug in (*CATEGORY_SLUGS, "other"):
        assert f"`{slug}`" in text
    agent = FakeAgent(['{"truth_slug": "other"}'])
    cj.judge_truth(agent, _item(prediction={"label": "legal_regulatory", "score": 0.9}))
    assert "legal_regulatory" not in agent.prompts[0]
    assert "MODEL PREDICTION" not in agent.prompts[0]


def test_judge_returns_the_truth_slug() -> None:
    agent = FakeAgent(['{"truth_slug": "mergers_acquisitions", "rationale": "a deal"}'])
    verdict = cj.judge_truth(agent, _item())
    assert verdict.truth_slug == "mergers_acquisitions"
    assert not verdict.parse_failed


def test_unknown_slug_is_repaired_then_falls_back_rather_than_becoming_other() -> None:
    agent = FakeAgent(['{"truth_slug": "sports"}', '{"truth_slug": "sports"}'])
    verdict = cj.judge_truth(agent, _item())
    assert verdict.parse_failed
    assert len(agent.prompts) == 2
    repaired = cj.judge_truth(
        FakeAgent(['{"truth_slug": "sports"}', '{"truth_slug": "legal_regulatory"}']), _item()
    )
    assert repaired.truth_slug == "legal_regulatory"
    assert not repaired.parse_failed


def test_calibration_judge_is_opt_in_and_the_default_stage_lists_are_unchanged() -> None:
    assert STAGES == ("sentiment", "category", "ner", "c_summary", "sector_summary")
    assert set(JUDGES) == set(STAGES)
    assert set(VERDICT_MODELS) == set(STAGES)
    assert get_args(Stage) == ("sentiment", "ner", "category", "c_summary")
    assert cj.TASK not in STAGES


def test_the_judge_temperature_is_the_one_the_model_uses() -> None:
    assert JUDGE_TEMPERATURE == 0.0


# --- verdict reuse -----------------------------------------------------------


def _new_run(conn: db_module.NewsNlpDatabase, task: str, experiment: str = "base") -> int:
    run = create_eval_run(
        conn,
        stage=task,
        sample_size=1,
        low_conf_n=0,
        random_n=1,
        seed=1,
        judge_model="m",
        judge_url="u",
        code_version="v",
        experiment=experiment,
    )
    conn.commit()
    return run


def _stored(
    conn: db_module.NewsNlpDatabase,
    item: EvalItem,
    *,
    task: str,
    verdict: dict,
    experiment: str = "base",
) -> None:
    run = _new_run(conn, task, experiment)
    inf = record_inference(
        conn,
        run,
        article_id=item.article_id,
        task=task,
        experiment=experiment,
        bucket=item.bucket,
        prediction=item.prediction,
    )
    record_verdict(
        conn,
        inf,
        run,
        article_id=item.article_id,
        task=task,
        experiment=experiment,
        verdict=verdict,
        correct=None,
        severity=None,
        rationale="",
    )
    conn.commit()


def test_a_stored_label_verdict_never_satisfies_a_truth_lookup(
    conn: db_module.NewsNlpDatabase,
) -> None:
    item = _item()
    _stored(
        conn,
        item,
        task="category",
        verdict={"agrees": True, "ideal_slug": "other", "severity": 0, "rationale": ""},
    )
    assert cj.find_truth_verdict(conn, item, experiment="base") is None
    _stored(conn, item, task=cj.TASK, verdict={"truth_slug": "other", "rationale": ""})
    found = cj.find_truth_verdict(conn, item, experiment="base")
    assert found is not None
    assert found.truth_slug == "other"


def test_a_changed_prediction_is_not_reused(conn: db_module.NewsNlpDatabase) -> None:
    item = _item()
    _stored(conn, item, task=cj.TASK, verdict={"truth_slug": "other", "rationale": ""})
    changed = _item(prediction={"label": "other", "score": 0.99, "distribution": {}})
    assert cj.find_truth_verdict(conn, changed, experiment="base") is None


def test_the_reuse_key_carries_the_prompt_hash_and_the_judge_model(
    conn: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _item()
    key = cj.truth_experiment("model-a")
    _stored(conn, item, task=cj.TASK, verdict={"truth_slug": "other"}, experiment=key)
    assert cj.find_truth_verdict(conn, item, experiment=cj.truth_experiment("model-a"))
    assert cj.find_truth_verdict(conn, item, experiment=cj.truth_experiment("model-b")) is None
    monkeypatch.setattr(cj, "prompt_hash", lambda: "0" * 64)
    assert cj.find_truth_verdict(conn, item, experiment=cj.truth_experiment("model-a")) is None


# --- the run -----------------------------------------------------------------


def _sample_of(ids: list[int], cell: str = "legal_regulatory|2") -> cs.CalibrationSample:
    half = len(ids) // 2
    drawn = [
        cs.SampledArticle(i, cell, "fit" if n < half else "report", 100, half or 1)
        for n, i in enumerate(ids)
    ]
    return cs.CalibrationSample(seed=1, drawn=tuple(drawn), populations={cell: 100})


def _start(conn: db_module.NewsNlpDatabase, ids: list[int]) -> int:
    return cj.start_calibration_run(
        conn,
        _sample_of(ids),
        judge_model="m",
        judge_url="u",
        code_version="v",
        date="2026-10-23",
    )


def _factory(replies: dict[int, list[str]], items: list[EvalItem]) -> Callable[[str], JudgeAgent]:
    """An agent factory that serves each article its own replies (matched on the title)."""
    by_title = {
        f"T{it.article_id}": replies[it.article_id] for it in items if it.article_id in replies
    }

    def factory(_prompt: str) -> JudgeAgent:
        class _Agent:
            def __init__(self) -> None:
                self.replies: list[str] | None = None

            def __call__(self, prompt: str) -> str:
                if self.replies is None:
                    title = next(t for t in by_title if t in prompt)
                    self.replies = list(by_title[title])
                return self.replies.pop(0)

        return _Agent()

    return factory


def _numbered(i: int) -> EvalItem:
    base = _item(i)
    return EvalItem(
        article_id=i,
        bucket=base.bucket,
        stratum_population=100,
        title=f"T{i}",
        body_text=base.body_text,
        prediction=base.prediction,
    )


def test_start_calibration_run_stores_the_config(conn: db_module.NewsNlpDatabase) -> None:
    run = _start(conn, [1, 2, 3, 4])
    row = conn.execute(
        "SELECT stage, experiment, strata_json FROM eval_run WHERE id = ?", (run,)
    ).fetchone()
    cfg = json.loads(row[2])
    assert row[0] == cj.TASK
    assert row[1] == cj.truth_experiment("m")
    assert cfg["prompt_hash"] == cj.prompt_hash()
    assert cfg["temperature"] == JUDGE_TEMPERATURE
    assert cfg["date"] == "2026-10-23"
    assert cfg["halves"]["legal_regulatory|2"] == {"fit": [1, 2], "report": [3, 4]}


def test_a_run_started_under_other_wording_is_refused(
    conn: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _start(conn, [1, 2])
    monkeypatch.setattr(cj, "prompt_hash", lambda: "f" * 64)
    with pytest.raises(ValueError, match="frozen"):
        cj.judge_and_record(
            conn, run, [_numbered(1)], agent_factory=lambda _p: FakeAgent([]), max_calls=10
        )
    other = _new_run(conn, "category")
    with pytest.raises(ValueError, match="not a category_truth run"):
        cj.judge_and_record(
            conn, other, [_numbered(1)], agent_factory=lambda _p: FakeAgent([]), max_calls=10
        )


def test_judge_and_record_reuses_judges_and_records(conn: db_module.NewsNlpDatabase) -> None:
    run = _start(conn, [1, 2])
    items = [_numbered(1), _numbered(2)]
    _stored(
        conn,
        items[0],
        task=cj.TASK,
        verdict={"truth_slug": "legal_regulatory"},
        experiment=cj.truth_experiment("m"),
    )
    factory = _factory({2: ['{"truth_slug": "product_innovation"}']}, items)
    pairs, usage = cj.judge_and_record(conn, run, items, agent_factory=factory, max_calls=10)
    assert [v.truth_slug for _, v in pairs] == ["legal_regulatory", "product_innovation"]
    assert (usage.reused, usage.judged, usage.parse_failed, usage.calls) == (1, 1, 0, 1)
    rows = conn.execute(
        "SELECT article_id, task FROM eval_verdict WHERE run_id = ? ORDER BY article_id", (run,)
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [(1, cj.TASK), (2, cj.TASK)]
    assert cj.calls_spent(conn) == 1


def test_a_crash_keeps_every_verdict_already_paid_for(
    conn: db_module.NewsNlpDatabase, test_db_path: Path
) -> None:
    run = _start(conn, [1, 2, 3])
    items = [_numbered(i) for i in (1, 2, 3)]
    good = FakeAgent(['{"truth_slug": "other"}'])

    def factory(_prompt: str) -> JudgeAgent:
        if not factory.served:  # type: ignore[attr-defined]
            factory.served = True  # type: ignore[attr-defined]
            return good
        raise KeyboardInterrupt

    factory.served = False  # type: ignore[attr-defined]
    with pytest.raises(KeyboardInterrupt):
        cj.judge_and_record(conn, run, items, agent_factory=factory, max_calls=10, max_workers=1)
    other = db_module.connect(test_db_path)  # a second connection sees only what was committed
    try:
        rows = other.execute(
            "SELECT article_id FROM eval_verdict WHERE run_id = ?", (run,)
        ).fetchall()
        calls = json.loads(
            other.execute("SELECT metrics_json FROM eval_run WHERE id = ?", (run,)).fetchone()[0]
        )["calls"]
    finally:
        other.close()
    assert [r[0] for r in rows] == [1]
    assert calls == 1


def test_the_cap_is_on_calls_and_the_run_stops_cleanly_at_it(
    conn: db_module.NewsNlpDatabase,
) -> None:
    run = _start(conn, [1, 2, 3])
    items = [_numbered(i) for i in (1, 2, 3)]
    factory = _factory(
        {
            1: ["not json", '{"truth_slug": "other"}'],  # a repair: 2 calls
            2: ['{"truth_slug": "other"}'],
            3: ['{"truth_slug": "other"}'],
        },
        items,
    )
    pairs, usage = cj.judge_and_record(
        conn, run, items, agent_factory=factory, max_calls=3, max_workers=1
    )
    # item 1 used 2 calls; 1 is left, below the 2 an article may need: stop, do not start item 2
    assert [it.article_id for it, _ in pairs] == [1]
    assert (usage.calls, usage.judged, usage.not_judged) == (2, 1, 2)
    status = conn.execute("SELECT status FROM eval_run WHERE id = ?", (run,)).fetchone()[0]
    assert status == "stopped_at_cap"
    assert cj.remaining_calls(conn, cap=10) == 8
    assert cj.remaining_calls(conn) == cj.D10_CAP - 2


def test_duplicate_articles_are_refused(conn: db_module.NewsNlpDatabase) -> None:
    run = _start(conn, [1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        cj.judge_and_record(
            conn,
            run,
            [_numbered(1), _numbered(1)],
            agent_factory=lambda _p: FakeAgent([]),
            max_calls=10,
        )


def test_load_truths_excludes_parse_failures_and_reweights(
    conn: db_module.NewsNlpDatabase,
) -> None:
    run = _start(conn, [1, 2, 3, 4, 5, 6])  # fit 1,2,3; report 4,5,6; cell population 100
    items = [_numbered(i) for i in (1, 2, 3, 4, 5)]  # 6 is never judged
    factory = _factory(
        {
            1: ['{"truth_slug": "other"}'],
            2: ["bad", "bad"],  # parse_failed
            3: ['{"truth_slug": "legal_regulatory"}'],
            4: ['{"truth_slug": "other"}'],
            5: ['{"truth_slug": "other"}'],
        },
        items,
    )
    cj.judge_and_record(conn, run, items, agent_factory=factory, max_calls=20)
    truths = cj.load_truths(conn, run)
    assert (truths.parse_failed, truths.unjudged) == (1, 1)
    assert truths.parse_failed_rate == pytest.approx(1 / 6)
    assert sorted(t.article_id for t in truths.rows) == [1, 3, 4, 5]
    weights = {t.article_id: t.weight for t in truths.rows}
    assert weights[1] == weights[3] == 100 / 2  # fit half: 2 usable
    assert weights[4] == weights[5] == 100 / 2  # report half: 2 usable


# --- the sampler -------------------------------------------------------------


def _population() -> list[ConfirmedArticle]:
    articles: list[ConfirmedArticle] = []
    i = 0
    for slug_index in (0, 1):
        for score, n in ((0.30, 200), (0.45, 200), (0.55, 200), (0.70, 150), (0.90, 80)):
            for _ in range(n):
                i += 1
                articles.append(_article(i, slug_index, score))
    for _ in range(100):
        i += 1
        articles.append(_article(i, 0, 0.5, group=CATEGORY_GROUP_FLOOR - 0.1))
    return articles


def test_same_seed_same_draw_different_seed_different_draw() -> None:
    pop = _population()
    a, b, c = (cs.draw_sample(pop, seed=s) for s in (7, 7, 8))
    assert a.drawn == b.drawn
    assert a.drawn != c.drawn


def test_a_cell_draw_does_not_depend_on_other_cells() -> None:
    pop = _population()
    full = cs.draw_sample(pop, seed=3)
    other_cell = cell_of(0.9, _article(0, 1, 0.9).leaf_scores)
    kept = [a for a in pop if cell_of(a.group_score, a.leaf_scores) != other_cell]
    without = cs.draw_sample(kept, seed=3)
    keep = cell_of(0.9, _article(0, 0, 0.55).leaf_scores)
    assert [a for a in full.drawn if a.cell == keep] == [a for a in without.drawn if a.cell == keep]


def test_cell_sizes_halves_and_weights() -> None:
    sample = cs.draw_sample(_population(), seed=1)
    by_cell: dict[str, list[cs.SampledArticle]] = {}
    for a in sample.drawn:
        by_cell.setdefault(a.cell, []).append(a)
    band2 = cell_of(0.9, _article(0, 0, 0.55).leaf_scores)
    assert len(by_cell[band2]) == 90
    assert {a.half for a in by_cell[band2]} == {"fit", "report"}
    assert sum(a.half == "fit" for a in by_cell[band2]) == 45
    top = cell_of(0.9, _article(0, 0, 0.9).leaf_scores)
    assert len(by_cell[top]) == 60  # 30 per half outside the 0.4-0.6 bands
    assert all(a.weight == 80 / 30 for a in by_cell[top])
    assert len(by_cell["short_circuit"]) == 60
    ids = [a.article_id for a in sample.drawn]
    assert len(ids) == len(set(ids))
    # the halves are disjoint within a cell and a weight sums back to the population
    for cell, members in by_cell.items():
        for half in cs.HALVES:
            part = [a for a in members if a.half == half]
            assert sum(a.weight for a in part) == pytest.approx(sample.populations[cell])


def test_a_small_cell_is_taken_whole_and_split_evenly() -> None:
    sample = cs.draw_sample([_article(i, 0, 0.9) for i in range(1, 41)], seed=1)
    assert len(sample.drawn) == 40
    assert {a.n_half for a in sample.drawn} == {20}
    assert {a.weight for a in sample.drawn} == {2.0}


def test_design_effect_is_one_for_equal_weights_and_larger_otherwise() -> None:
    equal = cs.draw_sample([_article(i, 0, 0.55) for i in range(1, 400)], seed=1)
    assert cs.design_effect(equal, "report") == pytest.approx(1.0)
    unequal = cs.draw_sample(_population(), seed=1)
    deff = cs.design_effect(unequal, "report")
    assert deff is not None
    assert deff > 1.0
    empty = cs.draw_sample([], seed=1)
    assert cs.design_effect(empty, "fit") is None


def test_run_config_records_what_a_rerun_and_a_reweighting_need() -> None:
    sample = cs.draw_sample(_population(), seed=5)
    cfg = cs.run_config(
        sample,
        judge_model="m",
        temperature=0.0,
        judge_seed=None,
        prompt_hash=cj.prompt_hash(),
        date="2026-10-23",
    )
    assert json.loads(json.dumps(cfg)) == cfg
    assert cfg["seed"] == 5
    assert cfg["band_edges"] == [0.4, 0.5, 0.6, 0.8]
    assert cfg["prompt_hash"] == cj.prompt_hash()
    assert set(cfg["cell_populations"]) == set(cfg["halves"])
    some = next(iter(cfg["halves"].values()))
    assert set(some) == {"fit", "report"}


def test_sample_items_reads_text_and_the_stored_prediction(
    eval_conn: db_module.NewsNlpDatabase,
) -> None:
    rows = eval_conn.execute(
        "SELECT article_id, score FROM article_category ORDER BY article_id"
    ).fetchall()
    assert rows
    sample = cs.CalibrationSample(
        seed=1,
        drawn=(cs.SampledArticle(int(rows[0][0]), "cell", "fit", 10, 5),),
        populations={"cell": 10},
    )
    ((drawn, item),), effective = cs.sample_items(eval_conn, sample)
    assert item.article_id == drawn.article_id
    assert item.bucket == "cell"
    assert item.stratum_population == 10
    assert item.body_text.strip()
    assert item.prediction["score"] == rows[0][1]
    assert effective.drawn[0].n_half == 1  # recomputed: 1 remains, not the 5 drawn


def test_a_dropped_article_is_recorded_and_the_weights_are_recomputed(
    eval_conn: db_module.NewsNlpDatabase,
) -> None:
    first = int(eval_conn.execute("SELECT MIN(article_id) FROM article_category").fetchone()[0])
    sample = cs.CalibrationSample(
        seed=1,
        drawn=(
            cs.SampledArticle(first, "cell", "fit", 10, 2),
            cs.SampledArticle(999_999, "cell", "fit", 10, 2),  # no text, no stored row
        ),
        populations={"cell": 10},
    )
    pairs, effective = cs.sample_items(eval_conn, sample)
    assert [a.article_id for a, _ in pairs] == [first]
    assert effective.dropped == (999_999,)
    assert effective.drawn[0].weight == 10.0  # was 10 / 2: the half still sums to the cell
    cfg = cs.run_config(
        effective,
        judge_model="m",
        temperature=0.0,
        judge_seed=None,
        prompt_hash="h",
        date="2026-10-23",
    )
    assert cfg["dropped"] == [999_999]


def test_restrict_keeps_each_half_summing_to_its_cell_population() -> None:
    sample = cs.draw_sample([_article(i, 0, 0.55) for i in range(1, 300)], seed=2)
    victims = {a.article_id for a in sample.drawn[::7]}
    kept = cs.restrict(sample, {a.article_id for a in sample.drawn} - victims)
    assert set(kept.dropped) == victims
    cell = next(iter(kept.populations))
    for half in cs.HALVES:
        assert sum(a.weight for a in kept.half(half)) == pytest.approx(kept.populations[cell])


def test_load_confirmed_reads_only_accepted_articles() -> None:
    db = sqlite3.connect(":memory:")
    db.execute("ATTACH DATABASE ':memory:' AS r")
    db.execute("CREATE TABLE main.discovered_urls (id INTEGER, match_class TEXT)")
    leaves = ", ".join(f"{s} REAL" for s in CATEGORY_SLUGS)
    db.execute(
        f"CREATE TABLE r.article_category (article_id INTEGER, label TEXT, group_score REAL, {leaves})"
    )
    for i, klass in ((2, "accept"), (1, "accept"), (3, "reject")):
        db.execute("INSERT INTO main.discovered_urls VALUES (?, ?)", (i, klass))
        marks = ", ".join("?" for _ in range(3 + len(CATEGORY_SLUGS)))
        db.execute(
            f"INSERT INTO r.article_category VALUES ({marks})",  # noqa: S608 -- test placeholders
            (i, "other", 0.8, *([0.1 * i] * len(CATEGORY_SLUGS))),
        )
    articles = cs.load_confirmed(db)
    assert [a.article_id for a in articles] == [1, 2]
    assert articles[0].group_score == 0.8
    assert len(articles[0].leaf_scores) == len(CATEGORY_SLUGS)
    with pytest.raises(ValueError, match="schema names"):
        cs.load_confirmed(db, results_schema="r; DROP TABLE x")
