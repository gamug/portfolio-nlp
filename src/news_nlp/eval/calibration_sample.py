"""The calibration sample T-121(b) judges: confirmed articles stratified by (argmax slug x
score band) plus the level-1 short-circuit stratum (PLAN.md Work item 15 step 4, TASKS T-141).

Pure and seeded: the same articles and seed give the same draw. Each cell is shuffled with its
own generator (seeded from the run seed and the cell name), so a change to one cell's
population never moves another cell's draw. A cell contributes ``2 * report_half_size`` articles
(90 in the 0.4-0.6 bands, 60 elsewhere, 60 short-circuit); a cell with fewer articles
contributes all of them. The draw is split at random into a **fit** half and a **report** half
(thresholds are fitted on one and reported on the other, so a number is never scored on the data
it was tuned on). Each half is a simple random sample of its cell, so an article's population
weight in a half is ``cell population / articles of that cell in that half``.

No database access here: :func:`sample_items` turns the draw into judgeable ``EvalItem`` s.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from news_nlp.db import NewsNlpDatabase
from news_nlp.eval.calibration import (
    BAND_EDGES,
    ConfirmedArticle,
    cell_of,
    report_half_size,
)
from news_nlp.eval.sampling import EvalItem, _prediction, _require_source, _text

Half = Literal["fit", "report"]
HALVES: tuple[Half, ...] = ("fit", "report")


class SampledArticle(NamedTuple):
    article_id: int
    cell: str
    half: Half
    population: int  # N of the cell (all confirmed articles in it)
    n_half: int  # articles of this cell drawn into this half

    @property
    def weight(self) -> float:
        """How many confirmed articles this one stands for within its half."""
        return self.population / self.n_half


@dataclass(frozen=True)
class CalibrationSample:
    seed: int
    drawn: tuple[SampledArticle, ...]
    populations: Mapping[str, int]

    def half(self, half: Half) -> list[SampledArticle]:
        return [a for a in self.drawn if a.half == half]


def draw_sample(articles: Sequence[ConfirmedArticle], *, seed: int) -> CalibrationSample:
    """Stratify *articles* by :func:`cell_of` and draw each cell's fit and report halves."""
    by_cell: dict[str, list[int]] = defaultdict(list)
    for article in articles:
        by_cell[cell_of(article.group_score, article.leaf_scores)].append(article.article_id)
    drawn: list[SampledArticle] = []
    populations: dict[str, int] = {}
    for cell in sorted(by_cell):
        ids = sorted(by_cell[cell])
        populations[cell] = len(ids)
        random.Random(f"{seed}|{cell}").shuffle(ids)  # noqa: S311 -- sample selection
        chosen = ids[: 2 * report_half_size(cell)]
        n_fit = len(chosen) // 2
        halves: dict[Half, list[int]] = {"fit": chosen[:n_fit], "report": chosen[n_fit:]}
        for half, members in halves.items():
            drawn.extend(
                SampledArticle(i, cell, half, len(ids), len(members)) for i in sorted(members)
            )
    return CalibrationSample(seed, tuple(drawn), populations)


def design_effect(sample: CalibrationSample, half: Half) -> float | None:
    """Kish design effect ``n * sum(w^2) / (sum w)^2`` of one half's weights: the factor the
    sizing in ``calibration.items_to_decide`` multiplies the sample by. ``None`` if the half
    is empty."""
    weights = [a.weight for a in sample.half(half)]
    if not weights:
        return None
    return len(weights) * sum(w * w for w in weights) / sum(weights) ** 2


def run_config(
    sample: CalibrationSample,
    *,
    judge_model: str,
    temperature: float,
    judge_seed: int | None,
    prompt_hash: str,
    date: str,
) -> dict[str, Any]:
    """What the run must store to be reproducible and re-weightable: the seed, band edges,
    cell populations and half assignment, plus the judge's identity (model id, temperature,
    seed, prompt hash) and the date."""
    assignment: dict[str, dict[str, list[int]]] = {
        cell: {half: [] for half in HALVES} for cell in sample.populations
    }
    for a in sample.drawn:
        assignment[a.cell][a.half].append(a.article_id)
    return {
        "seed": sample.seed,
        "band_edges": list(BAND_EDGES),
        "cell_populations": dict(sample.populations),
        "halves": assignment,
        "judge_model": judge_model,
        "temperature": temperature,
        "judge_seed": judge_seed,
        "prompt_hash": prompt_hash,
        "date": date,
    }


def sample_items(
    conn: NewsNlpDatabase, sample: CalibrationSample
) -> list[tuple[SampledArticle, EvalItem]]:
    """The judgeable item of every drawn article, in draw order. ``bucket`` is the cell and
    ``stratum_population`` its population. Articles whose text or stored row is missing are
    skipped (the caller compares lengths)."""
    schema = _require_source(conn)
    out: list[tuple[SampledArticle, EvalItem]] = []
    for drawn in sample.drawn:
        text = _text(conn, schema, drawn.article_id, "category")
        prediction = _prediction(conn, "category", drawn.article_id)
        if text is None or prediction is None:
            continue
        title, body = text
        out.append(
            (
                drawn,
                EvalItem(
                    article_id=drawn.article_id,
                    bucket=drawn.cell,
                    stratum_population=drawn.population,
                    title=title,
                    body_text=body,
                    prediction=prediction,
                ),
            )
        )
    return out
