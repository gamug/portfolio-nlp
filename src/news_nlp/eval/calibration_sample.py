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

:func:`load_confirmed` reads the population, :func:`sample_items` turns a draw into judgeable
``EvalItem`` s. An article whose text or stored row is missing is dropped **and the weights are
recomputed** from the articles that remain (:func:`restrict`), the drops are listed in the
sample and in :func:`run_config`; the estimates then assume the missing ones are missing at
random within their cell and half, which the recorded list lets a reader check.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, Protocol

from news_nlp.db import NewsNlpDatabase
from news_nlp.eval.calibration import (
    BAND_EDGES,
    ConfirmedArticle,
    cell_of,
    report_half_size,
)
from news_nlp.eval.sampling import EvalItem, _prediction, _require_source, _text
from news_nlp.taxonomy import CATEGORY_SLUGS


def text_fingerprint(title: str, body: str) -> str:
    """SHA-256 of the text the judge reads."""
    return hashlib.sha256(f"{title}\n{body}".encode()).hexdigest()


Half = Literal["fit", "report"]
HALVES: tuple[Half, ...] = ("fit", "report")


#: Schema names :func:`load_confirmed` may interpolate (never caller input beyond this set).
_SCHEMAS = frozenset({"main", "r", "source"})
_COLUMNS = ("group_score", *CATEGORY_SLUGS)


class _Executes(Protocol):
    def execute(self, sql: str, params: tuple[Any, ...] = (), /) -> Any: ...


def load_confirmed(
    conn: _Executes, *, results_schema: str = "r", urls_schema: str = "main"
) -> list[ConfirmedArticle]:
    """Every article whose ``discovered_urls.match_class`` is ``accept`` (the confirmed
    population), with its stored group score and nine leaf scores. *results_schema* holds
    ``article_category`` and *urls_schema* ``discovered_urls`` (two attached databases in the
    real stores)."""
    if results_schema not in _SCHEMAS or urls_schema not in _SCHEMAS:
        raise ValueError(f"schema names must be in {sorted(_SCHEMAS)}")
    cols = ", ".join(f"rc.{c}" for c in _COLUMNS)
    rows = conn.execute(
        f"SELECT rc.article_id, rc.label, {cols} "  # noqa: S608 -- schemas allowlisted, columns constants
        f"FROM {results_schema}.article_category rc "
        f"JOIN {urls_schema}.discovered_urls d ON d.id = rc.article_id "
        "WHERE d.match_class = 'accept' ORDER BY rc.article_id"
    )
    return [ConfirmedArticle(i, label, g, tuple(leaves)) for i, label, g, *leaves in rows]


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
    dropped: tuple[int, ...] = ()  # drawn articles removed by :func:`restrict`

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


def restrict(sample: CalibrationSample, keep: set[int]) -> CalibrationSample:
    """The sample without the articles outside *keep*, weights recomputed: ``n_half`` is the
    number of articles that remain in the cell and half, so a half still sums to its cell's
    population. A cell-half left empty vanishes (``design_effect`` and the weights only see
    what was judged)."""
    kept = [a for a in sample.drawn if a.article_id in keep]
    counts = Counter((a.cell, a.half) for a in kept)
    drawn = tuple(a._replace(n_half=counts[(a.cell, a.half)]) for a in kept)
    dropped = tuple(sorted({a.article_id for a in sample.drawn} - keep))
    return replace(sample, drawn=drawn, dropped=sample.dropped + dropped)


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
    seed, prompt hash) and the date, and the size and Kish design effect of each half (the factor
    ``calibration.items_to_decide`` multiplies its sample size by)."""
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
        "dropped": list(sample.dropped),
        "sampling": {
            half: {"n": len(sample.half(half)), "design_effect": design_effect(sample, half)}
            for half in HALVES
        },
    }


def sample_items(
    conn: NewsNlpDatabase, sample: CalibrationSample
) -> tuple[list[tuple[SampledArticle, EvalItem]], CalibrationSample]:
    """The judgeable item of every drawn article, in draw order, and the sample the items
    belong to: an article whose text or stored row is missing is dropped and the returned
    sample's weights are recomputed without it (:func:`restrict`; the drops are in its
    ``dropped``). ``bucket`` is the cell and ``stratum_population`` its population.

    An item's ``prediction`` is only ``{"text_sha256": ...}``: the truth judge never sees the
    classifier's output, so its verdict is keyed (``calibration_judge.find_truth_verdict``) on the
    text it read, and a classifier change, a new rule or T-124's re-label never invalidates a
    verdict already paid for. The label and scores T-121(b) analyses against come from
    ``article_category`` by article id; do not put them back into the item."""
    schema = _require_source(conn)
    built: dict[int, EvalItem] = {}
    for drawn in sample.drawn:
        text = _text(conn, schema, drawn.article_id, "category")
        if text is None or _prediction(conn, "category", drawn.article_id) is None:
            continue
        title, body = text
        built[drawn.article_id] = EvalItem(
            article_id=drawn.article_id,
            bucket=drawn.cell,
            stratum_population=drawn.population,
            title=title,
            body_text=body,
            prediction={"text_sha256": text_fingerprint(title, body)},
        )
    effective = restrict(sample, set(built))
    return [(a, built[a.article_id]) for a in effective.drawn], effective
