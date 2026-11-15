"""The 24 pairs of the blind re-scoring of the DeepSeek arm (T-206): how they were chosen, repeatable.

    uv run scripts/summary_blind_pairs_2026_11_15.py check   [--results experiments/results/summary_lead_2026_11_13]   # the choice against docs/.../blind_24.csv
    uv run scripts/summary_blind_pairs_2026_11_15.py sheet   [--results ...] [--out blind.txt]                         # the pairs to read, arm hidden, and the key

The choice (``choose``): from the 300 articles of the experiment, the **English** ones whose body is **under 9,000 characters**, 8 for each of the three tiers
(``random.Random(7)``), shuffled, and for each pair a coin decides which summary is shown first as ``X`` (the other is ``Y``). **The filters leave the long bodies
out** (the tier of 3 or more chunks, where the current step is weakest, is therefore represented by its shorter bodies) and the sample is 24: read the result as a
check of the direction of the judge's figures, not as an estimate. The reading itself (the scores of ``blind_24.csv``) was made by the assistant reading the article
and the two summaries, not by a person and not by a script.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results" / "summary_lead_2026_11_13"
BLIND = ROOT / "docs" / "summarization_results_2026_11_15" / "blind_24.csv"
SEED = 7
PER_TIER = 8
MAX_BODY_CHARS = 9000


def choose(
    sample: list[dict[str, Any]],
    seed: int = SEED,
    per_tier: int = PER_TIER,
    max_chars: int = MAX_BODY_CHARS,
) -> list[dict[str, Any]]:
    """``item`` (1-based, the order read), ``article_id``, ``tier`` and ``x_arm`` (``B`` or ``D``: the arm shown as ``X``) of each pair."""
    rng = random.Random(seed)  # noqa: S311 -- a sample, not security
    pick: list[dict[str, Any]] = []
    for tier in (1, 2, 3):
        ids = [
            r
            for r in sample
            if r["tier"] == tier and r["language"] == "en" and len(r["body"]) < max_chars
        ]
        pick += rng.sample(ids, per_tier)
    rng.shuffle(pick)
    out = []
    for n, r in enumerate(pick, 1):
        shown_d_first = rng.random() < 0.5  # noqa: PLR2004 -- a coin
        out.append(
            {
                "item": n,
                "article_id": r["article_id"],
                "tier": r["tier"],
                "x_arm": "D" if shown_d_first else "B",
            }
        )
    return out


def _load(results: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample = json.loads((results / "sample.json").read_text(encoding="utf-8"))
    arms = json.loads((results / "arms.json").read_text(encoding="utf-8"))
    return sample, arms


def check(results: Path) -> int:
    sample, _ = _load(results)
    chosen = choose(sample)
    with BLIND.open(encoding="utf-8", newline="") as f:
        committed = list(csv.DictReader(f))
    same = [
        (int(c["item"]), int(c["article_id"]), int(c["tier"]), c["X_arm"]) for c in committed
    ] == [(c["item"], c["article_id"], c["tier"], c["x_arm"]) for c in chosen]
    print(
        "the committed blind_24.csv is the choice of this script"
        if same
        else "DIFFERENT from the committed blind_24.csv"
    )
    return 0 if same else 1


def sheet(results: Path, out: Path) -> None:
    sample, arms = _load(results)
    by_id = {r["article_id"]: r for r in sample}
    summaries = {a["article_id"]: a["arms"] for a in arms}
    lines = []
    for c in choose(sample):
        r, s = by_id[c["article_id"]], summaries[c["article_id"]]
        x, y = (s["D"], s["B"]) if c["x_arm"] == "D" else (s["B"], s["D"])
        lines.append(
            f"##### {c['item']}\nTITLE: {r['title']}\nBODY: {r['body']}\nSUMMARY X: {x}\nSUMMARY Y: {y}\n"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"wrote {out} (the arm order of each pair is in blind_24.csv, column X_arm: read the sheet before it)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["check", "sheet"])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default="blind.txt")
    args = ap.parse_args()
    if args.cmd == "check":
        raise SystemExit(check(Path(args.results)))
    sheet(Path(args.results), Path(args.out))


if __name__ == "__main__":
    main()
