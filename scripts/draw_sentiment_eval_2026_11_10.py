# ruff: noqa: E402  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Draw the NEW evaluation set for the sentiment model (the owner's 2026-11-09 decision: ~400 new tags).

Same design as draw_sentiment_train_2026_11_07.py (summaries of the cut, scored by the pinned model only to balance
the draw per predicted class, rows in blocks of 15 = 5 per class, same tagging format with the Spanish pair), with the
one change that matters: the **never** set also holds the 450 training/validation bodies (and their headlines), so
the set is disjoint from everything the model was trained or validated on, and from the gold and held-out sets.

    uv run python draw_sentiment_eval.py <PR #154 worktree> [--per-class 135] [--seed 20261109] [--yes]

Writes data/sentiment_eval_ids.csv (with the texts and the stratum: git-ignored record) and
data/sentiment_eval_to_tag.csv (never overwritten); the Spanish cache is data/sentiment_eval/.
"""

import argparse
import sys
from pathlib import Path

WT = Path(sys.argv[1])
sys.path[:0] = [str(WT / "src"), str(WT / "scripts")]
import draw_sector_extra_2026_11_05 as extra
import draw_sentiment_gold_2026_11_05 as gold_draw
import draw_sentiment_train_2026_11_07 as d
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import retag

ap = argparse.ArgumentParser()
ap.add_argument("worktree")
ap.add_argument("--per-class", type=int, default=135)
ap.add_argument("--pool", type=int, default=6000)
ap.add_argument("--seed", type=int, default=20261109)
ap.add_argument("--yes", action="store_true")
args = ap.parse_args()
args.cut = d.CUT_DIR
out = Path("data")
ids_path, target = out / "sentiment_eval_ids.csv", out / "sentiment_eval_to_tag.csv"
if extra.has_rows(target):
    sys.exit(f"{target} exists: it is your tagging file and is not overwritten")
db.env.load_env_file()
conn = db.connect_pipeline()
gold_ids = {int(r["article_id"]) for r in extra._read(d.GOLD)}
held_ids = {int(r["article_id"]) for r in extra._read(out / "heldout_ids.csv")}
train_ids = {int(r["article_id"]) for r in extra._read(out / "sentiment_train_ids.csv")}
never, never_titles = d.exam_keys(conn, gold_ids | held_ids | train_ids)
print(
    f"never drawn: {len(never)} bodies, {len(never_titles)} headlines (gold {len(gold_ids)}, held-out {len(held_ids)}, train+validation {len(train_ids)})"
)
if ids_path.exists():
    chosen = extra._read(ids_path)
    print(f"{ids_path} exists: not drawn again")
else:
    chosen = d.draw_new(conn, args, never, never_titles)
    for t in chosen:  # an evaluation set: no train/validation split, no batches
        t["slice"], t["batch"] = "eval", "1"
    fields = tuple(f for f in d.IDS_FIELDS)
    extra._write(ids_path, [{f: t.get(f, "") for f in fields} for t in chosen])
d.check_draw(chosen, never, never_titles)
cache = out / "sentiment_eval"
cache.mkdir(exist_ok=True)
keyed = [{"article_id": t["key"], "title": t["title"], "text": t["text"]} for t in chosen]
done = retag_cli.add_spanish(
    cache, keyed, retag_cli._translator(), yes=args.yes, max_new=retag_cli.DEFAULT_MAX_NEW
)
spanish = {r["article_id"]: r for r in done}
rows = gold_draw.tag_rows(chosen, spanish)
retag.require_tagging_format(rows, target.name, retag.SENTIMENT_TAGGING_COLUMNS)
extra._write(target, retag.spreadsheet_safe(rows))
print(f"{len(rows)} texts -> {target}")
