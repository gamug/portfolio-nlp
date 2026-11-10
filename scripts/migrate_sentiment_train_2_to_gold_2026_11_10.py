# ruff: noqa: E402  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Migrate the tagged data/sentiment_eval_to_tag.csv to gold/sentiment_train_2.csv with the checks of
build_sentiment_train_2026_11_07.py (every mark a class, each text the one drawn by hash, ISO dates, the draw's design).
Split (chosen by the assistant on 2026-11-09, NOT decided by the owner; to be confirmed): blocks 1-18 (270 texts) are `train`, blocks
19-27 (135, 45 per predicted class) are `validation`: new texts never trained on, the check of the extra data. The working files keep their "eval" names (`data/sentiment_eval_*`).
Also rewrites the slice/batch of data/sentiment_eval_ids.csv to match (backup kept)."""

import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

WT = sys.argv[1]
sys.path[:0] = [WT + "/src", WT + "/scripts"]
import build_sentiment_train_2026_11_07 as b

from news_nlp import gold_metrics, retag

TRAIN_BLOCKS = 18
b.TO_TAG_NAME, b.IDS_NAME = "sentiment_eval_to_tag.csv", "sentiment_eval_ids.csv"
CL = gold_metrics.SENTIMENT_CLASSES


def design(rows):
    per = 15 // len(CL)
    counts = Counter(r["predicted"] for r in rows)
    assert set(counts) == set(CL) and len(set(counts.values())) == 1, counts
    blocks = {}
    for r in rows:
        blocks.setdefault(int(r["block"]), Counter())[r["predicted"]] += 1
    assert sorted(blocks) == list(range(1, len(blocks) + 1)), "blocks not 1..N"
    assert all(set(c.values()) == {per} and set(c) == set(CL) for c in blocks.values()), "block mix"
    assert len({r["body_hash"] for r in rows}) == len(rows), "two rows share a body"


b.check_design = design

out = Path("data")
marks = b.read_marks(out / b.TO_TAG_NAME)
ids = gold_metrics.read_rows(out / b.IDS_NAME)
rows = b.build(marks, ids)
for r in rows:
    r["batch"] = "2"
    r["slice"] = "train" if int(r["block"]) <= TRAIN_BLOCKS else "validation"
# the same body must not be in the first set
old = {r["body_hash"] for r in gold_metrics.read_rows(Path("gold/sentiment_train.csv"))}
assert not old & {r["body_hash"] for r in rows}, "a body is also in the first training set"
target = Path("gold/sentiment_train_2.csv")
assert not target.exists(), f"{target} exists"
gold_metrics.write_pair_atomically(
    {target: gold_metrics.render_csv(b.TRAIN_COLUMNS, retag.spreadsheet_safe(rows))}
)
shutil.copy(out / b.IDS_NAME, out / "sentiment_eval_ids.before_split.csv")
by = {r["key"]: r for r in rows}
fields = list(ids[0].keys())
with (out / b.IDS_NAME).open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fields)
    w.writeheader()
    for t in ids:
        w.writerow({**t, "batch": "2", "slice": by[t["key"]]["slice"]})
print(f"{target}: {len(rows)} rows")
for s in ("train", "validation"):
    m = [r for r in rows if r["slice"] == s]
    print(s, len(m), dict(Counter(r["sentiment"] for r in m)))
both = Counter((r["predicted"], r["sentiment"]) for r in rows)
print("pinned model (rows) vs owner (cols):", *CL)
for p in CL:
    print(f"  {p:>8}:", [both[(p, m)] for m in CL])
print("warnings:", b.warnings(rows))
