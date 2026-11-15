# ruff: noqa: E402, PLR0915, C901, PLR2004  -- one-shot research script
"""Logs every category model of 2026-11-14/15 to MLflow, one run per model trained (T-187).

    uv run scripts/category_mlflow_log_2026_11_15.py [--tracking-uri file:///workspaces/portfolio-nlp/mlruns] [--no-weights] [--force]

The training scripts (``category_*_cv_*.py``) did not log; this reads what they saved (the out-of-fold scores, the held-out and real-data
files) and records, for each model, its parameters, the per-class precision, recall and ROC AUC (names as ``news_nlp.eval``:
``precision_<class>``), accuracy, macro F1 and the D37 selection score (the macro mean of the per-class precision, recall and ROC AUC; a
class with no predicted row counts 0), and the files of the run. One MLflow experiment, ``news_nlp_train/category``. The refit
(``models/category_single_all/``) is logged with its held-out and real-data figures and its weights, and registered as
``category-entailment-single-12`` (status in the tags: a candidate, the held-out bar of D24 not met). A run that already exists (same name) is left as it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import category_multilabel as ml

L14 = ml.LABELS14
L12 = list(ml.merge_labels(tuple(L14), ml.FOLD_INTO_OTHER))
EXPERIMENT = "news_nlp_train/category"
OLD = ml.scores_dir(ROOT, "category_multilabel")
RESULTS = ROOT / "docs" / "category_results_2026_11_15"
PIN = json.loads((ROOT / "config" / "category_model.json").read_text(encoding="utf-8"))
ENTAIL = {
    k: v
    for k, v in PIN["recipe"].items()
    if k not in ("reading", "pairs", "output", "script", "reading_sha256_of")
}


def figures(gold: list[str], S: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Accuracy, macro F1, the D37 selection score and the per-class precision, recall and ROC AUC of the top label."""
    pred = [L12[j] for j in S.argmax(1)]
    per = ml.per_label([[g] for g in gold], [[p] for p in pred], L12)
    out = {
        f"{prefix}n_rows": float(len(gold)),
        f"{prefix}accuracy": float(np.mean([p == g for p, g in zip(pred, gold, strict=True)])),
    }
    f1, trio = [], []
    for j, lab in enumerate(L12):
        p, r = per[lab]["precision"] or 0.0, per[lab]["recall"] or 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
        auc = ml.roc_auc(S[:, j], [g == lab for g in gold]) if per[lab]["support"] else None
        if per[lab]["support"]:
            out[f"{prefix}precision_{lab}"], out[f"{prefix}recall_{lab}"] = p, r
            trio += [p, r, auc if auc is not None else 0.0]
            if auc is not None:
                out[f"{prefix}roc_auc_{lab}"] = auc
    out[f"{prefix}macro_f1"] = float(np.mean(f1))
    out[f"{prefix}selection_d37"] = float(np.mean(trio))  # over the classes that have rows
    return out


def first_tags(rows: pd.DataFrame) -> list[str]:
    return [
        ml.FOLD_INTO_OTHER.get(g.split(",")[0].strip(), g.split(",")[0].strip()) for g in rows.gold
    ]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI") or (ROOT / "mlruns").as_uri(),
    )
    ap.add_argument(
        "--no-weights", action="store_true", help="do not copy the weights (712 MB) into the store"
    )
    ap.add_argument("--force", action="store_true", help="log again a run that exists")
    a = ap.parse_args()
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_registry_uri(a.tracking_uri)
    exp = mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    have = {
        r.data.tags.get("mlflow.runName")
        for r in client.search_runs([exp.experiment_id], max_results=500)
    }

    def log(
        name: str,
        params: dict,
        metrics: dict,
        tags: dict,
        files: list[tuple[Path, str]] | None = None,
        weights: Path | None = None,
    ) -> str | None:
        if name in have and not a.force:
            print(f"[mlflow] {name}: exists, left as it is")
            return None
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"stage": "category", "experiment": name, **tags})
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: v for k, v in metrics.items() if math.isfinite(v)})
            for f, sub in files or []:
                mlflow.log_artifact(str(f), sub)
            if weights is not None and not a.no_weights:
                mlflow.log_artifacts(str(weights), "model")
            print(f"[mlflow] {name}: run {run.info.run_id[:8]}, {len(metrics)} metrics")
            return str(run.info.run_id)

    # --- the arms of the old tags (1,222 rows; the top label against the FIRST tag; 5 folds of the old partition)
    rows_old = pd.read_csv(OLD / "rows.csv", dtype=str)
    gold_old = first_tags(rows_old)
    arms = {
        "untouched_zero_shot__old_tags": (
            "oof_base.npz",
            {"model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "training": "none"},
        ),
        "entailment_every_tag_positive__old_tags": (
            "oof_multi.npz",
            {
                "model": "zero-shot start",
                "epochs": 3,
                "lr": 2e-5,
                "neg": 4,
                "positives": "every gold tag",
            },
        ),
        "head_on_zero_shot_encoder__old_tags": (
            "oof_head.npz",
            {"model": "zero-shot encoder, 14 sigmoid outputs", "loss": "BCE"},
        ),
        "deberta_v3_base_plain__old_tags": (
            "oof_deb.npz",
            json.loads((OLD / "oof_deb.meta.json").read_text(encoding="utf-8")) | {"labels": "12"},
        ),
    }
    for name, (file, params) in arms.items():
        S = np.load(OLD / file)["scores"]
        if S.shape[1] == 14:
            S = ml.merge_scores(S, tuple(L14), ml.FOLD_INTO_OTHER)
        log(
            name,
            params
            | {
                "rows": len(rows_old),
                "cv": "5 folds grouped by article, old partition (1,048 articles)",
            },
            figures(gold_old, S),
            {
                "tags_convention": "old (the owner's, before the review of 2026-11-15)",
                "kind": "cross-validation out of fold",
                "reading": "top label against the first tag",
            },
        )

    # --- the single-tag runs (1,216 / 1,224 rows, the review of 2026-11-15)
    new = ml.scores_dir(ROOT, "category_single", "rows_single.csv", "oof_single.npz")
    v1 = ml.scores_dir(ROOT, "category_single_v1", "rows_single.csv", "oof_single.npz")
    rows2 = pd.read_csv(new / "rows_single.csv", dtype=str)
    gold2 = [g.split(",")[0] for g in rows2.gold]
    S2 = np.load(new / "oof_single.npz")["scores"]
    rows1 = pd.read_csv(v1 / "rows_single.csv", dtype=str)
    S1 = np.load(v1 / "oof_single.npz")["scores"]
    pin_data = {
        k: v["sha256"] for k, v in PIN["data"].items() if isinstance(v, dict) and "sha256" in v
    }
    common = {
        "cv": "5 folds grouped by article, single-tag partition (1,049 articles)",
        "start_model": PIN["start_model"]["repo"],
        "start_revision": PIN["start_model"]["revision"],
        **ENTAIL,
    }
    log(
        "entailment_single__run1_older_rows_old_tags",
        common | {"rows": len(rows1)},
        figures([g.split(",")[0] for g in rows1.gold], S1),
        {
            "tags_convention": "single tags; the 225 older rows still with the old tags",
            "kind": "cross-validation out of fold",
        },
    )
    base_old = np.load(OLD / "oof_base.npz")["scores"]
    base12 = ml.merge_scores(base_old, tuple(L14), ml.FOLD_INTO_OTHER)
    idx = {(r.article_id, r.company): i for i, r in enumerate(rows_old.itertuples())}
    pairs = [
        (i, idx[(r.article_id, r.company)])
        for i, r in enumerate(rows2.itertuples())
        if (r.article_id, r.company) in idx
    ]
    log(
        "untouched_zero_shot__single_tags",
        {
            "rows": len(pairs),
            "model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
            "training": "none",
            "note": "1,214 of the 1,216 rows: two sheet rows were in no earlier run",
        },
        figures([gold2[i] for i, _ in pairs], base12[[j for _, j in pairs]]),
        {
            "tags_convention": "single tags (reviewed 2026-11-15)",
            "kind": "scored out of fold (no training)",
        },
    )
    log(
        "entailment_single__run2_selected_candidate",
        common | {"rows": len(rows2), **{f"sha256_{k}": v[:16] for k, v in pin_data.items()}},
        figures(gold2, S2),
        {
            "tags_convention": "single tags (reviewed 2026-11-15)",
            "kind": "cross-validation out of fold",
            "status": "candidate; the held-out bar of D24 is NOT met (see the refit run)",
        },
        [
            (ROOT / "config" / "category_model.json", "pin"),
            (RESULTS / "ovr_single.md", "report"),
            (RESULTS / "single_comparison.md", "report"),
        ],
    )
    deb_file = ROOT / "data" / "category_single" / "oof_deb_single.npz"
    if not deb_file.exists():
        deb_file = new / "oof_deb_single.npz"
    if deb_file.exists():
        meta = (
            json.loads(deb_file.with_suffix(".meta.json").read_text(encoding="utf-8"))
            if deb_file.with_suffix(".meta.json").exists()
            else {}
        )
        log(
            "deberta_v3_base_plain__single_tags",
            meta | {"rows": len(rows2), "cv": common["cv"]},
            figures(gold2, np.load(deb_file)["scores"]),
            {
                "tags_convention": "single tags (reviewed 2026-11-15)",
                "kind": "cross-validation out of fold",
                "same_folds_as": "entailment_single__run2_selected_candidate",
            },
        )

    # --- the refit: held-out (once) and the real-data sample (300 rows), with the weights
    held = pd.read_csv(RESULTS / "heldout_selected.csv", dtype={"article_id": str})
    S_h = held[[f"score_{lab}" for lab in L12]].to_numpy()
    real = pd.read_csv(RESULTS / "real_sample.csv", dtype={"article_id": str})
    real_gold = list(real.tag)
    real_pred = list(real.selected)
    per = ml.per_label([[g] for g in real_gold], [[p] for p in real_pred], L12)
    m = figures(list(held.gold), S_h, "heldout_")
    m["real300_accuracy"] = float(
        np.mean([p == g for p, g in zip(real_pred, real_gold, strict=True)])
    )
    m["real300_n_rows"] = float(len(real))
    for lab in L12:
        if per[lab]["support"]:
            m[f"real300_precision_{lab}"], m[f"real300_recall_{lab}"] = (
                per[lab]["precision"] or 0.0,
                per[lab]["recall"] or 0.0,
            )
    weights = ROOT / "models" / "category_single_all"
    wfile = weights / "model.safetensors"
    tags = {
        "tags_convention": "refit on the 1,216 single-tag rows; held-out tags of the earlier convention; real-data tags by the assistant, blind to the models",
        "kind": "refit on all training rows (no fold held out); held-out scored ONCE",
        "status": "candidate; held-out bar of D24 (80%) NOT met: 54.1%",
        "weights_path": str(weights),
        "weights_sha256": sha256(wfile)[:16] if wfile.exists() else "missing",
    }
    run_id = log(
        "entailment_single__refit_all_1216",
        common
        | {
            "rows": len(rows2),
            "refit_seed": 0,
            "heldout_rows": len(held),
            "real_sample_rows": len(real),
        },
        m,
        tags,
        [
            (RESULTS / "heldout_selected.md", "heldout"),
            (RESULTS / "heldout_selected.csv", "heldout"),
            (RESULTS / "real_sample.md", "real_sample"),
            (RESULTS / "real_sample.csv", "real_sample"),
            (ROOT / "config" / "category_model.json", "pin"),
        ],
        weights if weights.exists() else None,
    )
    if run_id and not a.no_weights and weights.exists():
        mv = mlflow.register_model(
            f"runs:/{run_id}/model",
            "category-entailment-single-12",
            tags={
                "status": "candidate, not wired",
                "heldout_label_agreement": f"{m['heldout_accuracy']:.3f}",
            },
        )
        print(f"[mlflow] registered category-entailment-single-12 version {mv.version}")


if __name__ == "__main__":
    main()
