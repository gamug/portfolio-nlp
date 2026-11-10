# ruff: noqa: E402, E741, PLR0124  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Experiment `sentiment_pair_more_data`: the PR #154 recipe (2 epochs, lr 3e-5, seeds 1-4) with
  A = the first training set (341 summaries + replay)     B = A + the 270 new training summaries (gold/sentiment_train_2.csv)
Same replay for both (341 sentences, excluding the bodies of BOTH mark sets). Scored on the old validation slice (83) and on
the 135 new validation texts, each weighted by the strata of its own draw. The pinned text-only model is scored on the new 135 too."""

import importlib.util
import shutil
import statistics as st
import sys
from pathlib import Path

WT = Path(sys.argv[1])
MLRUNS = sys.argv[2]
sys.path[:0] = [str(WT / "src"), str(WT / "scripts")]
spec = importlib.util.spec_from_file_location(
    "pair_script", WT / "scripts/train_sentiment_pair_2026_11_09.py"
)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)
import json

import draw_sentiment_gold_2026_11_05 as gold_draw
import mlflow

import sentiment_pair_report as rep
import sentiment_pair_train as spt
from news_nlp import env
from sentiment_stage import _text_mentions_subject

pair.OUT = Path("data/sentiment_pair_more_data")
pair.MODELS = Path("models/sentiment_pair_more_data")
env.load_env_file()
conn = pair.connect_source()
old_marks, old_ids = pair.read_csv(pair.MARKS), pair.read_csv(pair.IDS)
new_marks, new_ids = (
    pair.read_csv(Path("gold/sentiment_train_2.csv")),
    pair.read_csv(Path("data/sentiment_eval_ids.csv")),
)
old = spt.load_marked_summaries(old_marks, old_ids)
new = spt.load_marked_summaries(new_marks, new_ids)
print(
    "old train",
    len(old.train),
    "val",
    len(old.validation),
    "| new train",
    len(new.train),
    "val",
    len(new.validation),
    "left out",
    new.left_out,
)
bodies = old.body_hashes | new.body_hashes
titles_m = {
    k
    for r in old_ids + new_ids
    if r["body_hash"] in bodies and (k := pair.headline_key(r["title"]))
}
hashes, titles = pair.exam_keys(conn, bodies, titles_m)
sentences = [json.loads(l) for l in pair.REPLAY_POOL.read_text(encoding="utf-8").splitlines()]
infos = pair.article_infos(conn, {int(s["article_id"]) for s in sentences})
replay, report = spt.replay_examples(
    sentences,
    infos.get,
    hashes,
    _text_mentions_subject,
    n=len(old.train),
    seed=pair.REPLAY_SEED,
    never_titles=titles,
)
print("\n".join(report.lines()))
sets = {"A_old": old.train + replay, "B_plus_new": old.train + new.train + replay}
s_old = pair.ValidationScorer(old.validation, old_marks)
s_new = pair.ValidationScorer(new.validation, new_marks)


def stats(preds, marks):
    sc, pops = rep.scored_from_rows(preds, marks)
    return rep.statistics_of(sc, rep.row_weights(sc, pops))


# the pinned text-only model on the new validation (the reference the pair models are set against)
items = [{"key": e.key, "text": e.text, "company": e.company} for e in new.validation]
pinned = gold_draw.predict_probs([dict(i) for i in items], None, pair=False)
pin_preds = [
    {
        "key": t["key"],
        "expected": s_new._expected[t["key"]],
        "predicted": t["predicted"],
        **{f"p_{c}": repr(float(t["probs"][c])) for c in rep.CLASSES},
    }
    for t in pinned
]
pin = stats(pin_preds, new_marks)
mlflow.set_tracking_uri(MLRUNS)
mlflow.set_experiment("sentiment_pair_more_data")
out = {}
KEYS = (
    "score",
    "macro precision",
    "macro recall",
    "macro roc_ovr",
    "neutral precision",
    "neutral recall",
    "negative precision",
    "negative recall",
)
for seed in (1, 2, 3, 4):
    for name, train in sets.items():
        s = spt.RunSettings(2, 3e-5, seed)
        keep = pair.MODELS / f"{name}_s{seed}"
        row = pair.one_run(s, train, s_old, keep=keep, tag=name)
        a = stats(s_old.predictions, old_marks)
        s_new(str(keep))
        b = stats(s_new.predictions, new_marks)
        shutil.rmtree(keep, ignore_errors=True)
        out.setdefault(name, []).append((a, b))
        with mlflow.start_run(run_name=f"{name}__s{seed}"):
            mlflow.set_tags({"stage": "sentiment", "kind": "more_data", "variant": name})
            mlflow.log_params(
                {
                    "epochs": 2,
                    "learning_rate": 3e-5,
                    "seed": seed,
                    "n_train": len(train),
                    "n_new_summaries": len(new.train) if name != "A_old" else 0,
                }
            )
            mlflow.log_metrics(
                {f"old_val_{k.replace(' ', '_')}": v for k, v in a.items() if v == v}
            )
            mlflow.log_metrics(
                {f"new_val_{k.replace(' ', '_')}": v for k, v in b.items() if v == v}
            )
        print(
            f"{name} seed {seed}: old-val score {a['score']:.3f}  new-val score {b['score']:.3f}",
            flush=True,
        )
print("\n== weighted figures, mean (sd) over 4 seeds ==")
print("pinned text-only on the new 135:   " + "  ".join(f"{k} {pin[k]:.3f}" for k in KEYS))
for name, runs in out.items():
    for label, idx in (("old val 83 ", 0), ("new val 135", 1)):
        vals = [r[idx] for r in runs]
        print(
            f"{name:11} {label}: "
            + "  ".join(
                f"{k} {st.mean(v[k] for v in vals):.3f}"
                + (f"({st.stdev(v[k] for v in vals):.3f})" if k == "score" else "")
                for k in KEYS
            )
        )
