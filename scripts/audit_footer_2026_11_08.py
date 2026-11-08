"""Audit the footer rules of ``news_nlp.attribution_units.strip_footer`` on a random sample of the
corpus (Work item 18, T-148b4).

    uv run scripts/audit_footer_2026_11_08.py [--n 4000] [--seed 20261108] [--show 13] [--sample 10]

Draws ``--n`` bodies of ``source.articles`` (by a seeded draw over the ids), strips each footer and
prints: how many bodies change, how many lose more than a fifth of their text, the id and the removed
text of each of those (up to ``--show``), and the removed text of ``--sample`` of the others, so the
reading of what the rules take out can be repeated. It reads SOURCE only and writes nothing.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import news_nlp as db
from news_nlp import attribution_units

LOSS = 0.2  # a body that loses more than this share of its text is read one by one


def removed(body: str, kept: str) -> str:
    """The lines of *body* that are not in *kept*, in order."""
    left = list(kept.split("\n"))
    out: list[str] = []
    for line in body.split("\n"):
        if left and line == left[0]:
            left.pop(0)
        else:
            out.append(line)
    return "\n".join(out).strip()


def audit(
    bodies: dict[int, str], seed: int = 1, show: int = 13, sample: int = 10
) -> dict[str, object]:
    """Counts and the removed text of the bodies that change: *show* of those that lose more than
    :data:`LOSS` of their text and a seeded *sample* of the others."""
    changed: dict[int, str] = {}
    heavy: list[int] = []
    for article, body in sorted(bodies.items()):
        kept = attribution_units.strip_footer(body)
        if kept == body:
            continue
        changed[article] = removed(body, kept)
        if len(kept) < (1 - LOSS) * len(body):
            heavy.append(article)
    rest = [a for a in changed if a not in heavy]
    picked = random.Random(seed).sample(rest, min(sample, len(rest)))  # noqa: S311
    return {
        "bodies": len(bodies),
        "changed": len(changed),
        "lose more than a fifth": len(heavy),
        "heavy": [(a, changed[a]) for a in heavy[:show]],
        "sample": [(a, changed[a]) for a in picked],
    }


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--show", type=int, default=13)
    parser.add_argument("--sample", type=int, default=10)
    args = parser.parse_args()
    conn = db.connect_pipeline()
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM source.articles WHERE body_text IS NOT NULL AND body_text != ''"
        )
    ]
    picked = random.Random(args.seed).sample(ids, min(args.n, len(ids)))  # noqa: S311
    bodies = {
        a: conn.execute("SELECT body_text FROM source.articles WHERE id = ?", (a,)).fetchone()[0]
        for a in picked
    }
    result = audit(bodies, args.seed, args.show, args.sample)
    for key in ("bodies", "changed", "lose more than a fifth"):
        print(f"{key}: {result[key]}")
    for label in ("heavy", "sample"):
        print(f"--- removed text, {label}:")
        for article, text in result[label]:  # type: ignore[attr-defined]
            print(f"[{article}] {text[:400]!r}")


if __name__ == "__main__":
    main()
