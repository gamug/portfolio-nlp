"""How many sector bodies does the sector step assign a sector to? A random sample, scored to the end (2026-11-13; answers the review of #179).

    uv run scripts/sector_rate_sample_2026_11_13.py [--n 1500] [--seed 20261113] [--scope-revision REV]

The "117 of 176" the first version of ``docs/sector_step.md`` quoted was a progress line of the draw of the clean test, which stopped when it
had enough of each kind (a stopping rule biases the share), and the 64 bodies of the first real run are in the order of the table (34 were IT).
This takes ``--n`` of the ``sector`` bodies of step 1 at random (a seed), scores every one of them with the step as it is, writes nothing, and
prints the share assigned (with its Wilson interval), the sectors given, and what the bodies sent ``out_of_scope`` looked like.
Needs the RESULTS store (``body_scope``) and SOURCE (the texts and the companies).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_sector_fresh_2026_11_13 import wilson

import news_nlp as db
from cut_stage import companies_of
from news_nlp import steps
from sector_stage import SECTOR_CODES, SectorClassifier

BATCH = 16
NAMES = {code: name for name, code in SECTOR_CODES.items()}


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--scope-revision", default=None)
    args = parser.parse_args()
    conn = db.connect_pipeline()
    scope_revision = args.scope_revision or steps.latest_classifier_revision(conn)
    hashes = sorted(
        str(r[0])
        for r in conn.execute(
            "SELECT body_hash FROM body_scope WHERE classifier_revision = ? AND scope = 'sector'",
            (scope_revision,),
        )
    )
    sample = random.Random(args.seed).sample(hashes, min(args.n, len(hashes)))  # noqa: S311 -- a reproducible sample
    print(
        f"{len(sample):,} of the {len(hashes):,} sector bodies of {scope_revision}; reading SOURCE ...",
        flush=True,
    )
    bodies = list(steps._bodies_by_hash(conn, sample))
    classifier = SectorClassifier(companies_of(conn))
    classifier.prepare()
    print(f"revision {classifier.revision}", flush=True)
    status: Counter[str] = Counter()
    sector: Counter[str] = Counter()
    out_candidate: Counter[str] = Counter()
    for start in range(0, len(bodies), BATCH):
        batch = bodies[start : start + BATCH]
        for result in classifier.classify([(b.title, b.text) for b in batch]):
            status[result.status] += 1
            if result.status == "assigned":
                sector[result.sector] += 1
            else:
                out_candidate[NAMES.get(result.candidate, result.candidate)] += 1
    n = sum(status.values())
    lo, hi = wilson(status["assigned"], n)
    print(
        f"assigned {status['assigned']:,} of {n:,} = {status['assigned'] / n:.1%} (95%: {lo:.0%} to {hi:.0%}); out_of_scope {status['out_of_scope']:,}"
    )
    print("sectors given:", {NAMES[c]: k for c, k in sector.most_common()})
    print("out_of_scope, best class:", dict(out_candidate.most_common()))


if __name__ == "__main__":
    main()
