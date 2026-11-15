"""The multi-label reading of the category scores (T-187, owner 2026-11-14).

A news text can carry more than one category (the sheet of 1,000 rows was tagged that way: 213 rows carry two and 10 carry three; the decision of 2026-11-15, D43, is one tag per row). The zero-shot model already scores each label by itself (entailment, the scores do not sum to 1); the step
keeps the largest. This module holds what a multi-label reading needs and no model: the labels of a row, the set a
row is given from its 14 scores, the pairs a fine-tune learns from (every gold label is a positive, and a negative is
never a gold label), the figures, and the choice of the rule on other folds only. No torch here, so it is tested.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np

#: The tags only: the text is in ``gold/texts/`` (docs/gold_datasets.md).
GOLD_COLUMNS = ("review_order", "article_id", "ticker", "company", "labels")
#: The texts of the set, tracked beside the tags (docs/gold_datasets.md, owner 2026-11-10); no Spanish aid.
TEXT_COLUMNS = (
    "review_order",
    "article_id",
    "ticker",
    "company",
    "pub_date",
    "source_domain",
    "title",
    "text",
)
#: Not a class (T-174): a row with only this is left out, as one of several it is dropped.
NOT_A_CLASS = "uncertain"
#: The rules tried (1.01 is "the top label alone"): a label other than the top joins when its score is at least ``min_score`` and at least
#: ``ratio`` times the top score.
GRID_MIN_SCORE = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)
GRID_RATIO = (0.0, 0.25, 0.5, 0.75)


def parse_labels(cell: str, labels: Sequence[str]) -> list[str]:
    """The labels of a tagged cell (``a, b``): in order, without repeats and without ``uncertain``."""
    parts = [p.strip() for p in cell.split(",") if p.strip()]
    if not parts:
        raise ValueError("an empty label cell")
    unknown = [p for p in parts if p not in labels and p != NOT_A_CLASS]
    if unknown:
        raise ValueError(f"not a label: {unknown[0]}")
    out: list[str] = []
    for p in parts:
        if p != NOT_A_CLASS and p not in out:
            out.append(p)
    return out


def predict_set(
    scores: np.ndarray,
    labels: Sequence[str],
    min_score: float,
    ratio: float,
    max_labels: int = 3,
) -> list[str]:
    """The top label always, then the others over the rule, best first, up to ``max_labels``."""
    order = [int(j) for j in np.argsort(-scores, kind="stable")]
    top = float(scores[order[0]])
    out = [labels[order[0]]]
    for j in order[1:]:
        if len(out) >= max_labels:
            break
        if scores[j] >= min_score and scores[j] >= ratio * top:
            out.append(labels[j])
    return out


def label_pairs(
    gold: Sequence[Sequence[str]],
    labels: Sequence[str],
    neg: int,
    rng: random.Random,
    first_only: bool = False,
) -> list[tuple[int, str, int]]:
    """(row, label, entails) for one epoch. Every gold label is a positive pair; a negative is drawn from the
    labels that are not gold, so the model is never taught that a true second label is false. ``first_only`` keeps
    only the first label as positive (the old single-label recipe) and still keeps the other gold labels out of the
    negatives, so the two recipes differ in the positives only."""
    out: list[tuple[int, str, int]] = []
    for i, g in enumerate(gold):
        positives = list(g[:1]) if first_only else list(g)
        out += [(i, label, 1) for label in positives]
        others = [label for label in labels if label not in g]
        out += [(i, label, 0) for label in rng.sample(others, min(neg, len(others)))]
    rng.shuffle(out)
    return out


def figures(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], labels: Sequence[str]
) -> dict[str, float]:
    """Top-label hit, exact set, and micro and macro F1 over (row, label) pairs."""
    n = len(gold)
    hit = [p[0] in g for g, p in zip(gold, pred, strict=True)]
    exact = [set(g) == set(p) for g, p in zip(gold, pred, strict=True)]
    tp = {label: 0 for label in labels}
    fp = dict(tp)
    fn = dict(tp)
    for g, p in zip(gold, pred, strict=True):
        for label in set(p) & set(g):
            tp[label] += 1
        for label in set(p) - set(g):
            fp[label] += 1
        for label in set(g) - set(p):
            fn[label] += 1
    ttp, tfp, tfn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mp = ttp / (ttp + tfp) if ttp + tfp else 0.0
    mr = ttp / (ttp + tfn) if ttp + tfn else 0.0
    f1s = []
    for label in labels:
        if tp[label] + fn[label] == 0:
            continue  # no gold row: not averaged in
        prec = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        rec = tp[label] / (tp[label] + fn[label])
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {
        "n": float(n),
        "hit_at_1": sum(hit) / n,
        "exact": sum(exact) / n,
        "micro_precision": mp,
        "micro_recall": mr,
        "micro_f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_set_size": float(np.mean([len(p) for p in pred])),
    }


def bootstrap_ci(
    values: np.ndarray, groups: Sequence[object], boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval of the mean, resampling groups (articles) and not rows."""
    rng = np.random.default_rng(seed)
    by: dict[object, list[float]] = {}
    for v, g in zip(values, groups, strict=True):
        by.setdefault(g, []).append(float(v))
    keys = list(by)
    means = []
    for _ in range(boot):
        draw = [keys[int(i)] for i in rng.integers(0, len(keys), len(keys))]
        vals = [x for k in draw for x in by[k]]
        means.append(sum(vals) / len(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def folds_of(article_ids: Sequence[object], k: int, seed: int = 0) -> list[int]:
    """A fold per row; an article is in one fold only."""
    ids = sorted(set(article_ids), key=str)
    random.Random(seed).shuffle(ids)  # noqa: S311 -- a reproducible split, not security
    fold = {a: i % k for i, a in enumerate(ids)}
    return [fold[a] for a in article_ids]


def best_rule(
    scores: np.ndarray, gold: Sequence[Sequence[str]], labels: Sequence[str]
) -> tuple[float, float]:
    """The (min_score, ratio) of the grid with the best micro F1 on these rows (ties: the stricter rule)."""
    best, arg = -1.0, (GRID_MIN_SCORE[-1], GRID_RATIO[-1])
    for ms in reversed(GRID_MIN_SCORE):
        for ra in reversed(GRID_RATIO):
            pred = [predict_set(s, labels, ms, ra) for s in scores]
            f = figures(gold, pred, labels)["micro_f1"]
            if f > best + 1e-12:
                best, arg = f, (ms, ra)
    return arg


def cross_fitted_sets(
    scores: np.ndarray,
    gold: Sequence[Sequence[str]],
    labels: Sequence[str],
    fold: Sequence[int] | np.ndarray,
) -> tuple[dict[int, tuple[float, float]], list[list[str]]]:
    """The set of every row from a rule picked on the OTHER folds, so no row is scored by a rule that saw it."""
    fold_a = np.asarray(fold)
    rules: dict[int, tuple[float, float]] = {}
    sets: list[list[str]] = [[] for _ in range(len(gold))]
    for k in sorted(set(fold_a.tolist())):
        train = np.flatnonzero(fold_a != k)
        rules[k] = best_rule(scores[train], [gold[int(i)] for i in train], labels)
        for i in np.flatnonzero(fold_a == k):
            sets[int(i)] = predict_set(scores[i], labels, *rules[k])
    return rules, sets


def gold_rows(
    sheet: Sequence[dict[str, str]], labels: Sequence[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """The tags and the texts of the owner's sheet. A row whose cell is empty, holds a label that is not one, or
    holds only ``uncertain`` is left out and listed in ``left_out``; the other cases stop the build with the row named: the owner fixes the sheet, nothing is guessed."""
    gold: list[dict[str, str]] = []
    texts: list[dict[str, str]] = []
    left_out: list[str] = []
    seen: set[str] = set()
    for r in sheet:
        order = r["review_order"]
        if order in seen:
            raise ValueError(f"review_order {order} appears twice")
        seen.add(order)
        try:
            parsed = parse_labels(r["label"], labels)
        except ValueError as e:
            raise ValueError(f"review_order {order}: {e}") from e
        if not parsed:
            left_out.append(order)  # only "uncertain": not a class, kept out of the set
            continue
        gold.append(
            {**{c: r[c] for c in GOLD_COLUMNS if c != "labels"}, "labels": ",".join(parsed)}
        )
        texts.append({c: r[c] for c in TEXT_COLUMNS})
    return gold, texts, left_out


def per_label(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    labels: Sequence[str],
    weights: Sequence[float] | None = None,
) -> dict[str, dict[str, float | None]]:
    """One versus rest, label by label: the rows that carry it (support), the rows predicted with it, precision,
    recall and F1. A figure with no row behind it is None, not 0. ``weights`` count a row by its weight (see
    ``article_weights``); without them every row counts once."""
    w = [1.0] * len(gold) if weights is None else list(weights)
    out: dict[str, dict[str, float | None]] = {}
    for label in labels:
        sup = sum(x for g, x in zip(gold, w, strict=True) if label in g)
        npred = sum(x for p, x in zip(pred, w, strict=True) if label in p)
        tp = sum(x for g, p, x in zip(gold, pred, w, strict=True) if label in g and label in p)
        prec = tp / npred if npred else None
        rec = tp / sup if sup else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if sup and npred else None)
        out[label] = {
            "support": float(sup),
            "predicted": float(npred),
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return out


def article_weights(article_ids: Sequence[object]) -> list[float]:
    """1 over the rows of the article, so each article counts once however many rows it gave."""
    n: dict[object, int] = {}
    for a in article_ids:
        n[a] = n.get(a, 0) + 1
    return [1.0 / n[a] for a in article_ids]


def per_label_ci(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    labels: Sequence[str],
    article_ids: Sequence[object],
    boot: int = 1000,
    seed: int = 0,
    weights: Sequence[float] | None = None,
) -> dict[str, dict[str, tuple[float, float] | None]]:
    """95% intervals of each label's precision and recall, resampling articles. None where no resample has a figure."""
    rng = np.random.default_rng(seed)
    by: dict[object, list[int]] = {}
    for i, a in enumerate(article_ids):
        by.setdefault(a, []).append(i)
    keys = list(by)
    draws: dict[str, dict[str, list[float]]] = {
        label: {"precision": [], "recall": []} for label in labels
    }
    for _ in range(boot):
        idx = [i for k in rng.integers(0, len(keys), len(keys)) for i in by[keys[int(k)]]]
        res = per_label(
            [gold[i] for i in idx],
            [pred[i] for i in idx],
            labels,
            None if weights is None else [weights[i] for i in idx],
        )
        for label in labels:
            for m in ("precision", "recall"):
                v = res[label][m]
                if v is not None:
                    draws[label][m].append(v)
    out: dict[str, dict[str, tuple[float, float] | None]] = {}
    for label in labels:
        out[label] = {}
        for m in ("precision", "recall"):
            d = draws[label][m]
            out[label][m] = (
                None if not d else (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
            )
    return out


def roc_auc(scores: Sequence[float], positive: Sequence[int]) -> float | None:
    """The area under the ROC curve by ranks (a tie counts half); None if a class has no row."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(positive, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def explode_pairs(
    gold: Sequence[Sequence[str]], output: Sequence[str]
) -> list[tuple[int, str, str]]:
    """One row per gold tag of a record, each carrying the model's output for that record: (record, gold tag,
    output). A record with two tags is two rows with the same output, so a single output is right on at most one."""
    return [(i, tag, out) for i, (g, out) in enumerate(zip(gold, output, strict=True)) for tag in g]


def pair_ceiling(gold: Sequence[Sequence[str]]) -> float:
    """The share of the pair rows a single output per record can get right at best: one row per record."""
    return len(gold) / sum(len(g) for g in gold)


#: The cuts tried per label, and the one a label gets when no row of the training folds carries it.
GRID_THRESHOLD = tuple(round(0.05 * k, 2) for k in range(1, 20))
DEFAULT_THRESHOLD = 0.5


def fit_thresholds(
    scores: np.ndarray, gold: Sequence[Sequence[str]], labels: Sequence[str]
) -> dict[str, float]:
    """A cut per label: the one with the best F1 on these rows (one versus rest); among equal cuts, the middle one."""
    out: dict[str, float] = {}
    for j, label in enumerate(labels):
        y = np.array([label in g for g in gold])
        if not y.any():
            out[label] = DEFAULT_THRESHOLD
            continue
        f1s = []
        for t in GRID_THRESHOLD:
            hit = scores[:, j] >= t
            tp = float((hit & y).sum())
            f1s.append(2 * tp / (hit.sum() + y.sum()))
        best = max(f1s)
        tied = [t for t, f in zip(GRID_THRESHOLD, f1s, strict=True) if f >= best - 1e-12]
        out[label] = tied[len(tied) // 2]
    return out


def set_by_thresholds(
    scores: np.ndarray, labels: Sequence[str], thresholds: dict[str, float], max_labels: int = 3
) -> list[str]:
    """The labels over their own cut, best first, up to ``max_labels``; if none is over its cut, the top label."""
    order = [int(j) for j in np.argsort(-scores, kind="stable")]
    out = [labels[j] for j in order if scores[j] >= thresholds[labels[j]]][:max_labels]
    return out or [labels[order[0]]]


def cross_fitted_thresholds(
    scores: np.ndarray,
    gold: Sequence[Sequence[str]],
    labels: Sequence[str],
    fold: Sequence[int] | np.ndarray,
    max_labels: int = 3,
) -> list[list[str]]:
    """The set of every row from cuts fitted on the OTHER folds only."""
    fold_a = np.asarray(fold)
    sets: list[list[str]] = [[] for _ in range(len(gold))]
    for k in sorted(set(fold_a.tolist())):
        train = np.flatnonzero(fold_a != k)
        th = fit_thresholds(scores[train], [gold[int(i)] for i in train], labels)
        for i in np.flatnonzero(fold_a == k):
            sets[int(i)] = set_by_thresholds(scores[i], labels, th, max_labels)
    return sets


#: Labels folded into ``other`` (owner, 2026-11-14): too few tagged rows to measure (``competitive_actions`` 15,
#: ``peer_comparison`` 8, ``other`` 30 of 1,222).
FOLD_INTO_OTHER = {"competitive_actions": "other", "peer_comparison": "other"}


def merge_gold(gold: Sequence[Sequence[str]], mapping: dict[str, str]) -> list[list[str]]:
    """The tags with the folded labels renamed; a row that had both gets the label once."""
    out: list[list[str]] = []
    for g in gold:
        row: list[str] = []
        for tag in g:
            new = mapping.get(tag, tag)
            if new not in row:
                row.append(new)
        out.append(row)
    return out


def merge_labels(labels: Sequence[str], mapping: dict[str, str]) -> tuple[str, ...]:
    """The labels that stay, in their order."""
    return tuple(label for label in labels if label not in mapping)


def merge_scores(scores: np.ndarray, labels: Sequence[str], mapping: dict[str, str]) -> np.ndarray:
    """The scores of the labels that stay; a label that takes others in gets the largest of them. This is a
    reading of scores a model gave for the old labels, not a model trained on the merged ones."""
    keep = merge_labels(labels, mapping)
    out = np.zeros((scores.shape[0], len(keep)))
    for j, label in enumerate(keep):
        cols = [labels.index(label)] + [labels.index(k) for k, v in mapping.items() if v == label]
        out[:, j] = scores[:, cols].max(axis=1)
    return out


def markdown_table(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A pipe table; None is written as a dash."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join("-" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def length_batches(
    lengths: Sequence[int], bs: int, rng: random.Random, pool: int = 64
) -> list[list[int]]:
    """Batches of row positions for one epoch: shuffle, cut into pools, sort each pool by length and cut it into
    batches, then shuffle the batches. Rows of similar length share a batch, so little padding is paid for."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches: list[list[int]] = []
    for i in range(0, len(order), pool):
        chunk = sorted(order[i : i + pool], key=lambda j: lengths[j])
        batches += [chunk[k : k + bs] for k in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


def pos_weights(
    gold: Sequence[Sequence[str]], labels: Sequence[str], cap: float = 4.0
) -> dict[str, float]:
    """The weight of a label's positives in the loss: the root of negatives over positives, at least 1 and at most ``cap``."""
    n = len(gold)
    out: dict[str, float] = {}
    for label in labels:
        pos = sum(1 for g in gold if label in g)
        out[label] = cap if pos == 0 else min(cap, max(1.0, ((n - pos) / pos) ** 0.5))
    return out


def global_thresholds(labels: Sequence[str], cut: float) -> dict[str, float]:
    """The same cut for every label (the parameter of a run that does not fit one per label)."""
    return {label: cut for label in labels}


#: Where the out-of-fold scores of the 2026-11-14/15 runs are kept in the repository (no text: article id, company, tag, scores).
COMMITTED_SCORES = Path("docs") / "category_results_2026_11_15" / "scores"


def scores_dir(root: Path, name: str) -> Path:
    """The folder of one run's out-of-fold scores: the local ``data/<name>`` when a run left it there, else the committed copy."""
    local = root / "data" / name
    return local if local.is_dir() else root / COMMITTED_SCORES / name


def refit_seed(pin: dict[str, object]) -> int:
    """The seed of the one refit on all rows, read from ``config/category_model.json`` (``recipe.refit_seed``); a missing or non-integer seed stops the refit,
    so the weights that are scored on the held-out and wired are always the ones of a seed that is written down (T-201, T-202)."""
    recipe = pin.get("recipe")
    seed = recipe.get("refit_seed") if isinstance(recipe, dict) else None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"recipe.refit_seed must be an integer, not {seed!r}")  # noqa: TRY004 -- a wrong value in the pin file, not a wrong argument type
    return seed


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """The sha256 of a file (the weights of a refit are identified by it)."""
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def body_overlap(
    article_ids: Sequence[int], body_hashes: dict[int, str], folds: Sequence[int]
) -> dict[str, object]:
    """Which bodies are shared by several articles and sit in more than one fold (the leak the grouping by ``article_id`` cannot see).
    ``body_hashes`` maps an article id to the hash of its body; the rows are ``article_ids`` and their ``folds``, in the same order."""
    articles_of: dict[str, set[int]] = {}
    for article, digest in body_hashes.items():
        articles_of.setdefault(digest, set()).add(article)
    shared = {d: a for d, a in articles_of.items() if len(a) > 1}
    folds_of_body: dict[str, set[int]] = {}
    rows_of_body: dict[str, list[int]] = {}
    for i, (article, fold) in enumerate(zip(article_ids, folds, strict=True)):
        digest = body_hashes[article]
        folds_of_body.setdefault(digest, set()).add(int(fold))
        rows_of_body.setdefault(digest, []).append(i)
    crossing = sorted(d for d in shared if len(folds_of_body.get(d, ())) > 1)
    rows = sorted(i for d in crossing for i in rows_of_body[d])
    return {
        "articles": len(set(article_ids)),
        "bodies": len(folds_of_body),
        "shared_bodies": len(shared),
        "crossing_bodies": len(crossing),
        "crossing_articles": sorted(
            a for d in crossing for a in shared[d] if a in set(article_ids)
        ),
        "crossing_rows": rows,
    }


def data_hash_problems(pin: dict[str, object], root: Path) -> list[str]:
    """The data files of the pin (``data.*.file`` with a ``sha256``) whose hash on disk is not the pinned one, or that are missing: a refit must fit the data
    the figures were measured on, not an edited copy (T-200 allows an edit, and an edit changes the pin first)."""
    problems: list[str] = []
    data = pin.get("data")
    entries = data.values() if isinstance(data, dict) else []
    for entry in entries:
        if not (isinstance(entry, dict) and "file" in entry and "sha256" in entry):
            continue
        path = root / str(entry["file"])
        if not path.exists():
            problems.append(f"{entry['file']}: missing")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"{entry['file']}: the sha256 is not the pinned one")
    return problems


def recipe_hyperparameters(pin: dict[str, object]) -> dict[str, float | int]:
    """The hyper-parameters of the refit, from ``recipe`` of the pin and nowhere else: epochs, lr, negatives per positive, batch size, accumulation, max length."""
    recipe = pin.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("the pin has no recipe")  # noqa: TRY004 -- a wrong pin file, not a wrong argument type
    keys = {
        "epochs": "epochs",
        "lr": "learning_rate",
        "neg": "negatives",
        "bs": "batch_size",
        "accum": "gradient_accumulation",
        "max_len": "max_length",
    }
    missing = [v for v in keys.values() if v not in recipe]
    if missing:
        raise ValueError(f"recipe lacks {missing}")
    return {k: recipe[v] for k, v in keys.items()}


def check_refit_folder(folder: Path) -> dict[str, object]:
    """The record of a refit folder, after checking that the weights in it are the ones the record names (sha256 of ``model.safetensors``). A folder with no
    record, or whose weights were replaced, is refused: a score has to say which weights it is of (T-201)."""
    import json  # noqa: PLC0415

    record_path = folder / "refit_record.json"
    weights = folder / "model.safetensors"
    if not record_path.exists() or not weights.exists():
        raise ValueError(
            f"{folder} has no refit_record.json or no model.safetensors: not a refit folder"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if sha256_file(weights) != record.get("model_safetensors_sha256"):
        raise ValueError(f"{weights} is not the file the record names")
    return dict(record)


def refit_flag_problems(
    hyper: dict[str, float | int], values: dict[str, object]
) -> dict[str, object]:
    """The flags given to ``refit`` whose value is not the pin's: the refit takes its recipe from ``recipe`` of the pin, never from the command line."""
    return {k: v for k, v in values.items() if k in hyper and v != hyper[k]}


#: The 14 labels of D15 with the short definitions and the groups they were measured with (2026-11-13 to 2026-11-15). The step (``news_category_stage``) reads 12 since T-203
#: (``competitive_actions`` and ``peer_comparison`` read as ``other``); the research scripts of those days keep reading these 14, in this order, so their figures can be reproduced.
DEFINITIONS14: dict[str, str] = {
    "earnings_performance": "results, guidance, revenue or profit figures",
    "mergers_acquisitions": "deals, takeovers, divestitures",
    "leadership_governance": "executive appointments and departures, board actions, an executive's insider trade",
    "legal_regulatory": "litigation, regulatory action, sanctions, bankruptcy",
    "product_innovation": "product launches, R&D, technology",
    "capital_shareholder_returns": "dividends, buybacks, debt and credit actions of the company",
    "labor_human_capital": "layoffs, hiring, workforce and labour relations",
    "market_analyst_sentiment": "analyst ratings and price targets, stock price moves, an investor's bullish or bearish view",
    "partnerships_business_dev": "strategic alliances, joint ventures, distribution deals",
    "reputational_harm": "an event that damages or repairs the company's public standing: apologies, service failures, controversies, boycotts",
    "competitive_actions": "the company's own move or statement against or about a competitor: share fights, pricing pressure",
    "investor_activity": "a fund or famous investor buys or sells the stock, an activist position, 13F filings, block trades",
    "peer_comparison": "a comparison between companies: which is the better stock, who is ahead",
    "other": "an event that fits none of the other topics",
}
LABELS14: tuple[str, ...] = tuple(DEFINITIONS14)
GROUPS14: dict[str, tuple[str, ...]] = {
    "results_capital": ("earnings_performance", "capital_shareholder_returns"),
    "market_view": ("market_analyst_sentiment", "investor_activity", "peer_comparison"),
    "deals_alliances": ("mergers_acquisitions", "partnerships_business_dev", "competitive_actions"),
    "product_people": ("product_innovation", "labor_human_capital"),
    "governance_legal_reputation": (
        "leadership_governance",
        "legal_regulatory",
        "reputational_harm",
    ),
    "other": ("other",),
}
GROUP_OF14: dict[str, str] = {label: g for g, labels in GROUPS14.items() for label in labels}
