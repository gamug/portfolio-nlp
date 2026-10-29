"""Export the blind-judge truth labels of a calibration run for human review (silver -> gold).

    uv run scripts/export_gold_review_2026_10_26.py [--run-id 45] [--out data]

Writes three CSVs (``data/`` is git-ignored; the article text is not to be committed). It
**refuses to overwrite a review file that already holds human labels**, and writes the review and
key files together or not at all (``news_nlp.gold_review``). The run is the latest one that is not
still running or in error.

* ``category_gold_review.csv``: what the human sees, in a random order (any prefix is a random
  subset): the article and EMPTY ``human_label`` / ``human_notes`` columns. It carries neither
  the LLM's label nor the classifier's, so the human is not anchored by them.
* ``category_gold_review_key.csv``: the same rows by ``article_id`` with the classifier's stored
  label and scores, the sample cell and half, the pooled weight and the LLM's label and
  rationale, to be joined after tagging (agreement, Cohen's kappa, per-label confusion).
* ``category_labels_guide.csv``: the 10 labels and their scope, as the LLM judge was given them.

The text is the one the judge read (title and the first 6,000 characters).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import news_nlp as db
from news_nlp.eval import sampling
from news_nlp.eval.calibration_analysis import judged_articles
from news_nlp.eval.calibration_judge import PROMPT_NAME, TASK, load_truths
from news_nlp.eval.calibration_sample import load_confirmed
from news_nlp.eval.judges import load_prompt
from news_nlp.gold_review import TaggedFileError, write_review_files

SEED = 20261026


def labels_guide() -> list[tuple[str, str]]:
    """The taxonomy table of the judge prompt: (slug, scope)."""
    rows = re.findall(r"^\| `(\w+)` \| (.+?) \|$", load_prompt(PROMPT_NAME), re.MULTILINE)
    return [(slug, scope) for slug, scope in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-id", type=int, default=0, help="default: the latest calibration run")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    conn = db.connect_pipeline()
    run_id = (
        args.run_id
        or conn.execute(
            "SELECT MAX(id) FROM eval_run WHERE stage = ? AND status IN ('ok', 'stopped_at_cap', 'incomplete')",
            (TASK,),
        ).fetchone()[0]
    )
    articles = load_confirmed(conn, results_schema="main", urls_schema="source")
    truths = load_truths(conn, run_id)
    judged = judged_articles(truths, articles)
    verdicts = {
        int(a): v
        for a, v in conn.execute(
            "SELECT article_id, verdict_json FROM eval_verdict WHERE run_id = ? AND task = ?",
            (run_id, TASK),
        )
    }
    import json  # noqa: PLC0415

    order = list(range(len(judged)))
    random.Random(SEED).shuffle(order)  # noqa: S311 -- review order, not cryptography
    args.out.mkdir(parents=True, exist_ok=True)
    review, key = [], []
    for rank, index in enumerate(order, start=1):
        j = judged[index]
        meta = conn.execute(
            "SELECT ticker, company, pub_date, source_domain, title FROM source.articles WHERE id = ?",
            (j.article.article_id,),
        ).fetchone()
        text = sampling._text(conn, "source", j.article.article_id, "category")
        title, body = text if text else ("", "")
        review.append(
            {
                "review_order": rank,
                "article_id": j.article.article_id,
                "ticker": meta["ticker"],
                "company": meta["company"],
                "pub_date": meta["pub_date"],
                "source_domain": meta["source_domain"],
                "title": title,
                "text": body,
                "human_label": "",
                "human_notes": "",
            }
        )
        verdict = json.loads(verdicts[j.article.article_id])
        leaves = j.article.leaf_scores
        key.append(
            {
                "review_order": rank,
                "article_id": j.article.article_id,
                "cell": j.cell,
                "half": j.half,
                "pooled_weight": round(j.weight, 3),
                "classifier_stored_label": j.article.stored_label,
                "classifier_max_leaf_score": round(max(leaves), 4),
                "classifier_group_score": round(j.article.group_score, 4),
                "llm_label": j.truth,
                "llm_rationale": verdict.get("rationale", ""),
            }
        )
    try:
        write_review_files(args.out, review, key)
    except TaggedFileError as exc:
        sys.exit(f"{exc}. Copy it elsewhere or pass another --out directory.")
    with (args.out / "category_labels_guide.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "scope"])
        writer.writerows(labels_guide())
    print(f"eval_run {run_id}: {len(review)} articles -> {args.out}/category_gold_review*.csv")


if __name__ == "__main__":
    main()
