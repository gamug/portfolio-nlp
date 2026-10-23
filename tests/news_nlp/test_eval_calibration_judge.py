"""calibration_judge.py / calibration_sample.py: the T-141 truth-label judge and the seeded,
stratified, half-split calibration sample. Hermetic: the agent is a fake."""

from __future__ import annotations

import hashlib
import json

import pytest

import news_nlp as db_module
from news_nlp.eval import calibration_judge as cj
from news_nlp.eval import calibration_sample as cs
from news_nlp.eval.calibration import ConfirmedArticle, cell_of
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


def test_calibration_judge_is_opt_in() -> None:
    assert cj.TASK not in STAGES
    assert cj.TASK not in VERDICT_MODELS


# --- verdict reuse -----------------------------------------------------------


def _stored(conn: db_module.NewsNlpDatabase, item: EvalItem, *, task: str, verdict: dict) -> None:
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
        experiment="base",
    )
    inf = record_inference(
        conn,
        run,
        article_id=item.article_id,
        task=task,
        experiment="base",
        bucket=item.bucket,
        prediction=item.prediction,
    )
    record_verdict(
        conn,
        inf,
        run,
        article_id=item.article_id,
        task=task,
        experiment="base",
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


def test_judge_and_record_reuses_judges_and_records(conn: db_module.NewsNlpDatabase) -> None:
    done, fresh = _item(1), _item(2)
    _stored(conn, done, task=cj.TASK, verdict={"truth_slug": "legal_regulatory"})
    run = create_eval_run(
        conn,
        stage=cj.TASK,
        sample_size=2,
        low_conf_n=0,
        random_n=2,
        seed=1,
        judge_model="m",
        judge_url="u",
        code_version="v",
        experiment="base",
    )
    calls: list[str] = []

    def factory(prompt: str) -> FakeAgent:
        calls.append(prompt)
        return FakeAgent(['{"truth_slug": "product_innovation"}'])

    verdicts, usage = cj.judge_and_record(
        conn, run, [done, fresh], agent_factory=factory, experiment="base", max_workers=1
    )
    assert [v.truth_slug for v in verdicts] == ["legal_regulatory", "product_innovation"]
    assert (usage.reused, usage.judged, usage.parse_failed) == (1, 1, 0)
    assert len(calls) == 1
    rows = conn.execute(
        "SELECT article_id, task FROM eval_verdict WHERE run_id = ? ORDER BY article_id", (run,)
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [(1, cj.TASK), (2, cj.TASK)]


def test_judge_and_record_refuses_to_exceed_the_call_cap(
    conn: db_module.NewsNlpDatabase,
) -> None:
    calls: list[str] = []

    def factory(prompt: str) -> FakeAgent:
        calls.append(prompt)
        return FakeAgent(['{"truth_slug": "other"}'])

    with pytest.raises(ValueError, match="cap of 1"):
        cj.judge_and_record(
            conn, 1, [_item(1), _item(2)], agent_factory=factory, experiment="base", max_calls=1
        )
    assert calls == []
    assert conn.execute("SELECT COUNT(*) FROM eval_verdict").fetchone()[0] == 0


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
    ((drawn, item),) = cs.sample_items(eval_conn, sample)
    assert item.article_id == drawn.article_id
    assert item.bucket == "cell"
    assert item.stratum_population == 10
    assert item.body_text.strip()
    assert item.prediction["score"] == rows[0][1]
