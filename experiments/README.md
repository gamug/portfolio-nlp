# experiments/

`ExperimentSpec` JSON files (`src/experiment.py`, PLAN.md Work item 11,
SPEC.md FR-017) — each one fully describes a real, already-run model
experiment for one of the four ML stages. Run any of them end to end:

```bash
uv run cli/run_experiment.py --config experiments/sentiment_v4_class_weighted.json
```

`--source-db`/`--results-db` override `$SOURCE_DATABASE_URL`/`$DATABASE_URL`
the same way `cli/news_nlp_eval.py`'s own flags do. A run writes its result
record to `experiments/results/<name>.result.json` — local scratch output,
git-ignored (`.gitignore`), not checked in.

## The specs

| File | Stage | What it reproduces |
|---|---|---|
| `sentiment_v2_chunklevel_finetuned.json` | sentiment | The currently-*production* candidate: `ProsusAI/finbert` continue-fine-tuned, chunk-level entity-scoped aggregation. See the "v2/v3 are the same spec" gap below. |
| `sentiment_v3_downsampled.json` | sentiment | The neutral-downsampled rebalance (2026-09-14) — see the gap below; **content-identical** to v2's spec. |
| `sentiment_v4_class_weighted.json` | sentiment | The class-weighted-loss rebalance (2026-09-15) — `--weighted`, full unbalanced pool, no data discarded. **Adopted for production** (`PLAN.md` Work item 9, `src/pipeline.py`'s `MODEL_REVISIONS`). |
| `sentiment_v5_secbert_base.json` | sentiment | The base-checkpoint swap to `nlpaueb/sec-bert-base` (2026-09-15) — rejected before adoption, but run through the full downstream eval anyway at the user's request. |
| `sentiment_chunklevel_base_finbert.json` | sentiment | The un-fine-tuned base-FinBERT comparison arm (`PLAN.md` Work item 4) — chunk-level aggregation applied to plain `ProsusAI/finbert`, no fine-tuning at all. Eval-only (`pretrain.enabled=false`), scores `ProsusAI/finbert` as a candidate. |
| `ner_production.json` | ner | Eval-only, against the pinned production NER model — mirrors the post-subword-fragmentation-fix baseline (`docs/evaluation.md`'s 2026-09-12 follow-up, `sample_size=8000, seed=1`, SPEC.md §9's `ner` row). |
| `category_production.json` | category | Eval-only, against the pinned production category model — mirrors the hierarchical-classifier threshold-calibration baseline (`docs/evaluation.md`'s 2026-09-09 follow-up, `eval_run` 19, `sample_size=2800, seed=1`). |
| `c_summary_production.json` | c_summary | Eval-only, against the pinned production `c_summary` model — mirrors the post-sampling-fix baseline (`docs/evaluation.md`'s 2026-09-14 follow-up, `eval_run` 34, `sample_size=1000, seed=1`, SPEC.md §9's `c_summary` row). |

Every sentiment `pretrain.enabled=true` spec sets
`eval.candidate_prescore_size=2500` (not just `eval.sample_size=2000`),
matching the exact `--candidate-prescore-size 2500 --sample-size 2000
--seed 1` invocation v4's and v5's own real downstream evals used
(`docs/evaluation.md`'s 2026-09-15 follow-ups) — `EvalSettings` otherwise
silently defaults the pre-score pool to `sample_size` itself, which would
under-reproduce the historical run without this override.

## Three disclosed gaps — named, not silently omitted

**1. The removed "title-only" sentiment aggregation arms have no
runnable equivalent.** Two of the three candidates measured head-to-head
in `PLAN.md` Work item 4 (`docs/evaluation.md`'s 2026-09-13 follow-ups) —
title-only scoring (PR #43) and title-only-plus-fine-tuned — were rejected
and their aggregation code removed from `src/pipeline.py`/`sentiment_stage.py`
after the decision. There is no code path left for `run_experiment` to
drive; a JSON spec for either would describe something this repo can no
longer actually run. Not backfilled.

**2. Category's confidence-threshold calibration and `c_summary`'s
generation output-length-budget test are a different axis than this
schema.** Both were real, measured experiments (`docs/evaluation.md`'s
2026-09-09 and 2026-09-14 follow-ups) that tuned an *inference-time*
constant (`CATEGORY_CONFIDENCE_THRESHOLD`, `SUMMARY_MIN_OUTPUT_TOKENS`/
`SUMMARY_MAX_OUTPUT_TOKENS`) — not a training run, a dataset choice, or a
train/test split. `ExperimentSpec`'s `pretrain`/`eval` shape has no field
for "sweep this inference constant and re-evaluate at each value," and
adding one is out of scope for this schema's first version (`PLAN.md`
Work item 11's own scope note). `category_production.json`/
`c_summary_production.json` reproduce evaluating the *already-calibrated*
production models, not the calibration sweep itself.

**3. (New, found while backfilling) `sentiment_v2_chunklevel_finetuned.json`
and `sentiment_v3_downsampled.json` are byte-identical except for `name`
— this is deliberate, not a copy-paste error.** `docs/evaluation.md`'s
2026-09-14 follow-up states it plainly: v3 used "the same
procedure/hyperparameters as before (`ProsusAI/finbert` base, lr 2e-5, 4
epochs, seed 42)" as v2 — the *only* difference is whether
`data/sentiment_finetune/labeled_sentences_balanced_2026_09_14.jsonl`
happens to exist on disk at train time (`train_sentiment.py`'s
`load_training_pool` silently prefers it when present, over the original
unbalanced pool). `ExperimentSpec` has no field for "this external data
file must/must not exist" — a deliberate scope boundary (`PretrainSpec`
describes *configuration*, not filesystem state), not an oversight. One
real, honest consequence: running either of these two specs **today**
reproduces v3's behavior (the rebalanced file already exists in this
repo, checked in as part of `PLAN.md` Work item 9), not v2's original
pre-rebalance training pool — v2's specific historical run is no longer
directly reproducible via any JSON spec. `v2`'s file is kept anyway
(named for what it represents historically, matching `TASKS.md` T-101's
own naming), with this note as the record of why it can't do what its
name implies today.

**One more disclosed limitation, not a gap in the schema itself:**
`ProsusAI/finbert` (the `candidate_model` in `sentiment_chunklevel_base_finbert.json`,
and the `base_model` every sentiment `pretrain` spec continues fine-tuning
from) has never been pinned anywhere else in this repo — `src/pipeline.py`'s
`MODEL_REVISIONS` only pins the four *production* checkpoints (`SPEC.md`
§13 item 4). `sentiment_chunklevel_base_finbert.json`'s `candidate_revision`
(`4556d13015211d73dccd3fdd39d39232506f3e43`) is pinned here for the first
time, fetched live from the HF Hub API the same way `TASKS.md` T-001
pinned the four production models — a real commit SHA, not a placeholder,
but scoped to this one spec file rather than added to `pipeline.py`'s own
dict (it's a comparison arm, never a production model).
