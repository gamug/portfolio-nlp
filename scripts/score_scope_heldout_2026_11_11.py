"""T-147b closing check (owner 2026-11-11): the held-out 200 scored from the **saved weights** of step 1.

    uv run python scripts/score_scope_heldout_2026_11_11.py [--model-dir models/scope_final] [--out data/scope_heldout_saved.csv]

The weights are the ones ``cli/news_nlp_steps.py --step scope`` runs on (``scripts/train_scope_final_2026_11_11.py``), read through
the step's own ``ScopeClassifier`` (same fragment, same hypotheses, same revision), not through the experiment scripts. The 200
were never trained on. Printed the way ``scripts/scope_end_to_end_2026_11_10.py`` prints them (weighted to the corpus, four
classes, paired gain over the rules triage, confusion matrix), plus the model revision. This is **one fit, one seed**: it is a check
of the saved weights against ``docs/scope_training.md`` (83.0%, one earlier fit), not a seed study. Needs a GPU.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scope_end_to_end_2026_11_10 as e2e
import scope_title_baseline_2026_11_10 as base

from news_nlp import gold_metrics, triage

BATCH = 16


def predict(classifier: Any, rows: Sequence[dict[str, str]], batch: int = BATCH) -> list[str]:
    """The scope of every row from the classifier of the step, in order (the four classes)."""
    out: list[str] = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        out.extend(r.scope for r in classifier.classify([(r["title"], r["text"]) for r in chunk]))
    return out


def report(rows: list[dict[str, str]], predicted: list[str]) -> list[str]:
    """Weighted figures, the paired gain over the rules and the confusion matrix, as lines of text."""
    rules = [base.merged(triage.RULE_SCOPE_TO_GOLD[r["stratum"]]) for r in rows]
    lines = [
        *base.lines("saved weights (two fine-tuned NLI models)", rows, predicted),
        "",
        *base.lines("rules triage", rows, rules),
        "",
    ]
    populations: dict[str, int] = {}
    for r in rows:
        populations.setdefault(r["stratum"], int(r["frame"]))
    figures = sorted(
        (
            {
                "article_id": r["article_id"],
                "stratum": r["stratum"],
                "gold": base.merged(r["scope"]),
                "predicted": p,
                "baseline": q,
            }
            for r, p, q in zip(rows, predicted, rules, strict=True)
        ),
        key=lambda r: int(r["article_id"]),
    )
    gain = gold_metrics.paired_stratified_gain(
        figures, populations, metric="agreement", stratum_key="stratum"
    )
    lines.append(
        f"paired gain in weighted agreement over the rules: {gain} -> {gold_metrics.gain_verdict(gain)}"
    )
    count = Counter((base.merged(r["scope"]), p) for r, p in zip(rows, predicted, strict=True))
    lines.append(
        f"confusion matrix of the held-out {len(rows)} (plain), rows = gold, columns = predicted {base.LABELS}"
    )
    lines += [f"{g:>14} {[count[(g, p)] for p in base.LABELS]}" for g in base.LABELS]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=Path("models/scope_final"))
    parser.add_argument("--out", type=Path, default=Path("data/scope_heldout_saved.csv"))
    args = parser.parse_args()
    from scope_stage import ScopeClassifier  # noqa: PLC0415 -- loads torch

    classifier = ScopeClassifier(args.model_dir)
    test = e2e.load_sets()["test"]
    predicted = predict(classifier, test)
    print(f"model revision {classifier.revision}; {len(test)} held-out bodies")
    print("\n".join(report(test, predicted)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["article_id", "gold", "predicted"])
        for r, p in zip(test, predicted, strict=True):
            writer.writerow([r["article_id"], base.merged(r["scope"]), p])
    print(f"predictions -> {args.out}")


if __name__ == "__main__":
    main()
