"""Re-read a stored category eval run on a *subset* of the corpus, with no judge call
(PLAN.md Work item 15, T-133).

A stored run (``eval_run``/``eval_inference``/``eval_verdict``) sampled the whole RESULTS
store. To ask "how good was category on the articles that survive the upstream re-tag?"
the stored judgements are split by a per-article partition (``accept``/``reject``/
``review``) and each part is run back through :func:`news_nlp.eval.metrics.aggregate_category`,
the same code path as a live run, so the numbers are the constitution #12 metric set.

Reweighting: ``aggregate_category`` estimates population totals stratum by stratum with
weight ``N_h / n_h`` (``N_h`` the stratum's population, ``n_h`` the items present). Passed
only a partition's items, ``n_h`` shrinks to the partition's count, which would inflate
every weight. :func:`domain_pairs` therefore rescales each stratum's population to
``N_h * n_h(part) / n_h``, which leaves the weight at the original ``N_h / n_h``: the
standard domain (subpopulation) ratio estimator. It is an estimate with the partition's
own, smaller, sample behind it; the judged counts are reported next to every figure.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from news_nlp.eval.metrics import aggregate_category
from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.verdicts import CategoryVerdict

Pair = tuple[EvalItem, CategoryVerdict]


class _Executes(Protocol):
    def execute(self, sql: str, params: tuple[Any, ...] = (), /) -> Any: ...


def load_category_pairs(conn: _Executes, run_id: int) -> list[Pair]:
    """The stored ``(item, verdict)`` pairs of one category run, read-only.

    The item carries the stratum population recorded in the run's ``strata_json``; title
    and body are not needed by the metrics and are left empty."""
    strata = json.loads(
        conn.execute("SELECT strata_json FROM eval_run WHERE id = ?", (run_id,)).fetchone()[0]
    )
    pairs: list[Pair] = []
    rows = conn.execute(
        "SELECT i.article_id, i.bucket, i.prediction_json, v.verdict_json "
        "FROM eval_inference i JOIN eval_verdict v ON v.inference_id = i.id "
        "WHERE i.run_id = ? AND i.task = 'category' ORDER BY i.id",
        (run_id,),
    )
    for article_id, bucket, prediction_json, verdict_json in rows:
        item = EvalItem(
            article_id=article_id,
            bucket=bucket,
            stratum_population=int(strata[bucket]["population"]),
            title="",
            body_text="",
            prediction=json.loads(prediction_json),
        )
        pairs.append((item, CategoryVerdict.model_validate_json(verdict_json)))
    return pairs


def domain_pairs(pairs: Sequence[Pair], member: Callable[[int], bool]) -> list[Pair]:
    """The pairs whose article is in the partition, with each stratum's population
    rescaled so the Horvitz-Thompson weight is unchanged (see the module docstring).

    Parse failures are dropped first, as ``aggregate_category`` does, so ``n_h`` counts
    the same items the estimator will see."""
    ok = [(it, v) for it, v in pairs if not v.parse_failed]
    n_all = Counter(it.bucket for it, _ in ok)
    inside = [(it, v) for it, v in ok if member(it.article_id)]
    n_in = Counter(it.bucket for it, _ in inside)
    return [
        (
            EvalItem(
                article_id=it.article_id,
                bucket=it.bucket,
                stratum_population=round(
                    it.stratum_population * n_in[it.bucket] / n_all[it.bucket]
                ),
                title=it.title,
                body_text=it.body_text,
                prediction=it.prediction,
            ),
            v,
        )
        for it, v in inside
    ]


def _in_partition(partition_of: Mapping[int, str], name: str) -> Callable[[int], bool]:
    return lambda article_id: partition_of.get(article_id, "(none)") == name


def metrics_by_partition(
    pairs: Sequence[Pair], partition_of: Mapping[int, str]
) -> dict[str, dict[str, float]]:
    """``aggregate_category`` for the whole run (key ``"all"``) and for each partition
    present among the judged articles. An article missing from *partition_of* is the
    partition ``"(none)"``."""
    out: dict[str, dict[str, float]] = {
        "all": aggregate_category([it for it, _ in pairs], [v for _, v in pairs])
    }
    names = sorted({partition_of.get(it.article_id, "(none)") for it, _ in pairs})
    for name in names:
        part = domain_pairs(pairs, _in_partition(partition_of, name))
        out[name] = aggregate_category([it for it, _ in part], [v for _, v in part])
    return out


def judged_counts(
    pairs: Sequence[Pair], partition_of: Mapping[int, str]
) -> dict[str, dict[str, int]]:
    """Judged (parse-ok) items per partition and bucket, so each figure's sample is visible."""
    counts: dict[str, Counter[str]] = {}
    for it, v in pairs:
        if v.parse_failed:
            continue
        counts.setdefault(partition_of.get(it.article_id, "(none)"), Counter())[it.bucket] += 1
    return {name: dict(c) for name, c in sorted(counts.items())}
