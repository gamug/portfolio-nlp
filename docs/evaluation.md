# Model accuracy evaluation (`news_nlp.eval`)

The pipeline stores `article_sentiment` / `article_entities` / `article_category`
/ `article_summary` and everything downstream (`portfolio-financial-analysis`'s
SEMANTIC score, the knowledge-graph projection) treats them as ground truth.
Nothing measures whether they are right. `news_nlp.eval` fills that gap with an
**LLM-as-judge** evaluation: an LLM scores a sample of the stored predictions
against the source article text, and the aggregate metrics + per-row verdicts go
to MLflow and to two run-log tables in the RESULTS store.

## The judge is a model, not a gold set

There is no hand-labelled corpus. The numbers this produces are
**agreement-with-a-judge**, not truth. A blind spot shared by FinBERT and the
judge inflates the score. Mitigations: the low-confidence bucket (below) surfaces
the rows most likely to be wrong, and every verdict's `rationale` is stored for
manual spot-checking. Treat a *drop* between runs as the signal, not the
absolute value.

## Baseline (2026-09-08)

First full-corpus run (`--sample-size 1000`, all four stages), taken as the
baseline for future `--check-regression` comparisons. Run IDs are the
`eval_run` primary key (RESULTS store); MLflow run ID is the artifact/param
store (`news_nlp_eval/<stage>`).

| stage | n | headline metric | value | eval_run / mlflow |
|---|---|---|---|---|
| `sentiment` | 1000 | `macro_f1_vs_judge` | 0.4006 | 10 / `823579c3` |
| `category` | 1000 | `accuracy_vs_judge` | 0.6890 | 11 / `e40ed7d6` |
| `ner` | 1000 | `micro_f1` | 0.7418 | 12 / `3b41cbec` |
| `c_summary` | 1000 | `mean_faithfulness` | 4.8670 | 13 / `053079fd` |

Per-stage notes from the full metric set (not just the headline):

- **sentiment (weakest stage)** — overall agreement is only 42.3%
  (`agreement_rate`), and even the random bucket is 57.5% (low-conf bucket:
  32.2%). Per-class P/R shows a specific skew, not generic noise: `negative`
  precision 0.19 / recall 0.62 (FinBERT over-calls negative vs. the judge),
  `neutral` precision 0.67 / recall 0.39 (under-predicted). Reads as FinBERT
  skewing negative on full article bodies where the judge leans neutral —
  worth a manual spot-check of disagreed `negative` rows before treating the
  score as "the model is wrong" vs. "the judge's neutrality bar differs."
- **category** — `accuracy_vs_judge_low_conf` (0.833) is *higher* than
  `accuracy_vs_judge_random` (0.4725), backwards from what the sampling
  design expects. Cause: the low-conf bucket is dominated by near-threshold
  `other` calls, and `other` is both the most common label (66% rate on
  model and judge alike) and the best-scoring slug (`acc_other` 0.815). The
  random-bucket accuracy (47%) is the more representative number. Worst
  per-slug accuracy: `product_innovation` 0.14, `partnerships_business_dev`
  0.16, `capital_shareholder_returns` 0.20, `leadership_governance` 0.33 —
  the zero-shot model is mostly guessing on these four.
- **ner** — recall (0.84) well above precision (0.66): the model
  over-predicts. `hallucination_rate` 0.338 — about a third of predicted
  spans are wrong per judge. `PER` is the weakest type (F1 0.66 vs. `LOC`
  0.82 / `ORG` 0.75). `parse_fail_rate` 4.7% (n=953 of 1000 sampled) —
  expected per the error-only judge contract on entity-dense articles (see
  below).
- **c_summary (strongest stage)** — faithfulness 4.87/5 and
  `pct_with_hallucination` only 5.4%, but `mean_coverage` is just 3.02/5:
  summaries are accurate but not very complete, a terse/extractive tendency
  of `distilbart-cnn-12-6` more than a correctness problem.

### Follow-up (2026-09-08): sentiment judge text-scope bug found and fixed

The sentiment bullet above ("skewing negative on full article bodies where
the judge leans neutral") undersold the cause. Investigation found the judge
was shown only the first 6000 chars of `body_text` (`_MAX_BODY_CHARS` in
`src/news_nlp/eval/sampling.py`), while `run_sentiment_stage`
(`src/pipeline.py`) scores the **entire** `body_text` via chunking and a
token-weighted average across all chunks — a text-scope mismatch between what
the model saw and what the judge saw. Fixed: `sentiment` is now judged on the
full, uncapped `body_text`; `category` is unaffected (its cap is correct and
intentional — `run_category_stage` deliberately classifies only the lead
chunk, per `docs/modules/news-nlp.md`).

Empirically, truncation only explains part of the gap (pulled the actual
`judgements.json` for eval_run 10 / mlflow `823579c3`, cross-referenced
against source article length):
- Only 19.2% of the 1000 sampled articles exceeded 6000 chars (median body
  length 3397 chars, max 34884).
- Agreement rate, truncated vs. non-truncated: `positive` 0.276 vs. 0.455,
  `neutral` 0.570 vs. 0.703 (truncation clearly hurt these) — but `negative`
  0.200 vs. 0.192, essentially unchanged by truncation, despite `negative`
  having the worst per-class F1 (0.295) and being the single biggest driver
  of the low macro F1.
- On short (non-truncated) articles, sampled rationales for `negative`
  disagreements show a distinct pattern: the judge applies entity/company-
  focus and net-signal reasoning (e.g. "not about a specific company's
  performance", "mixed positive+negative nets to neutral") that FinBERT's
  whole-article softmax average has no mechanism to replicate. This is a
  deeper, likely larger mismatch than truncation, rooted in the judge
  prompt's per-company framing (`src/news_nlp/eval/prompts/sentiment.md`) vs.
  FinBERT's lack of any entity-scoping or net-signal logic. **Flagged as a
  pipeline-level follow-up (out of scope here) — not implemented.**

The low-confidence bucket's much worse agreement (0.322 vs. 0.575 for random)
was checked for a stratification bug (e.g. the global `ORDER BY score ASC`
skewing toward `negative`) and **ruled out**: `negative` is 38.2% of the
low_conf bucket vs. 39.8% of random vs. 38.8% overall — roughly proportional.
The gap is fully explained by low-confidence (near-tied 3-way softmax) rows
being inherently more error-prone, i.e. the sampling design working as
intended.

`ner` and `c_summary` are suspected of the same full-article-vs-lead-cap
mismatch (`run_ner_stage` and the summary stage's hierarchical reduce both
process the whole article, per `docs/modules/news-nlp.md`), but this has not
been empirically investigated the way sentiment was, and their sampling cap
was left unchanged.

A fresh `--stage sentiment` eval run is needed to get a post-fix number; the
0.4006 macro F1 above should not be compared against future runs without this
context.

### Why recall, not F1, for sentiment negative (2026-09-08)

Digging into the confusion matrix behind the numbers above (judge
`ideal_label` = truth, `article_sentiment.label` = prediction, n=1000, the
same eval_run 10 / mlflow `823579c3` run):

| truth ＼ pred | positive | negative | neutral | row total |
|---|---|---|---|---|
| positive | 102 | 59 | 84 | 245 |
| negative | 11 | 75 | 35 | 121 |
| neutral | 134 | 254 | 246 | 634 |

`negative` precision 0.193 / recall 0.620 — FinBERT casts a wide net for
negative and catches most of the true negatives (recall), but the net is
stuffed with false positives (precision): of the 388 articles FinBERT calls
`negative`, only 19.3% are actually negative per the judge, 65.5% are
actually neutral, 15.2% are actually positive.

For this pipeline's purpose, that's the *less* costly failure mode. Negative
sentiment is a valuable signal for portfolio construction (risk flags,
downgrades, adverse events) — an article that's actually neutral getting
mislabeled `negative` costs a false alarm downstream; an article that's
actually negative getting mislabeled anything else (missed) costs a blind
spot. Missing a real negative is worse than over-flagging a neutral one, so
**`sentiment`'s headline / `--check-regression` gate metric is now
`recall_negative`** (`metrics.HEADLINE["sentiment"]` in
`src/news_nlp/eval/metrics.py`), not the balanced `macro_f1_vs_judge`.
`macro_f1_vs_judge` (and `precision_negative`) are still computed and logged
every run — just no longer what gates a regression — so overall accuracy
stays visible without being the thing that blocks a run.

Retroactively, this run's `recall_negative` was 0.6198 (already logged to
MLflow every run; only which metric is *treated as headline* changed here) —
that's the number a future `--check-regression` run compares sentiment
against going forward, not the 0.4006 macro F1 above.

## What it evaluates

Four per-article stages. `sector_summary` is out of scope — it is deterministic
composition; only its one-sentence intro seed is generative.

| stage | headline metric | also logged |
|---|---|---|
| `sentiment` | `recall_negative`¹ | agreement rate (overall / low-conf / random), `macro_f1_vs_judge`, per-class P/R/F1, mean severity |
| `category` | `accuracy_vs_judge` | macro-F1, per-slug accuracy, model vs judge `other`-rate, mean severity |
| `ner` | `micro_f1` | span micro/macro P/R/F1, per-type F1, hallucination rate, miss rate. Error-only judge contract: it names just the `wrong` predicted spans + `missed` entities (not a verdict per span, which overflows on entity-dense articles); TP/FP/FN are derived from the predicted count. |
| `c_summary` | `mean_faithfulness` | mean coverage / conciseness (1-5), `pct_with_hallucination` |

Every run also logs `n` (rows judged) and `parse_fail_rate` (judge replies that
were not valid JSON after one repair attempt — excluded from the accuracy
numbers).

¹ Not F1 or accuracy: a missed real negative-sentiment article costs more for
portfolio construction than an over-flagged neutral one, so `--check-regression`
gates sentiment on recall specifically — see "Why recall, not F1, for
sentiment negative" above.

## Sampling: 60 % low-confidence + 40 % random

Each run samples `--sample-size` rows per stage (default 80):

- **low-confidence bucket** (`round(size * 0.6)` rows) — the *least-confident*
  stored rows, deterministically, so every run re-checks the true worst case:
  lowest `article_sentiment.score`; category picks within ±0.1 of
  `CATEGORY_CONFIDENCE_THRESHOLD` (0.4) or labelled `other`; lowest per-article
  `MIN(article_entities.score)`; for `c_summary` (no score), articles whose
  sentiment/NER inputs were themselves low-confidence.
- **random bucket** (the rest) — a `--seed`-reproducible uniform draw from the
  remaining predictions. This is where run-to-run variety comes from.

Metrics are reported overall and split `*_low_conf` / `*_random`, so you see both
headline and worst-case accuracy.

## Running it

Needs an OpenAI-compatible chat endpoint (DeepSeek in practice), same as
`portfolio-financial-analysis`. Put in `.env` (see `.env.example`):

```
LLM_API_KEY=...
LLM_MODEL=deepseek-chat
LLM_URL=https://api.deepseek.com
# MLFLOW_TRACKING_URI=./mlruns
```

```bash
uv sync --group eval

# all four stages, 80 rows each, into ./mlruns
uv run cli/news_nlp_eval.py --stage all --sample-size 80

# one stage, reproducible sample
uv run cli/news_nlp_eval.py --stage sentiment --seed 1 --sample-size 40

# fail (exit 1) if a headline metric dropped > 0.05 vs the previous MLflow run
uv run cli/news_nlp_eval.py --stage all --check-regression

uv run mlflow ui            # browse runs at http://127.0.0.1:5000
```

Flags: `--stage` (repeatable; `all` = every stage), `--sample-size`, `--seed`,
`--max-workers` (concurrent judge calls, default 4), `--source-db` /
`--results-db` (override `$SOURCE_DATABASE_URL` / `$DATABASE_URL`),
`--mlflow-uri`, `--check-regression`, `--regression-tolerance` (default 0.05).

## Where results go

- **MLflow** — one run per `(stage, invocation)` in experiment
  `news_nlp_eval/<stage>`. Params (judge model/url, sample size, bucket counts,
  seed, git SHA), metrics, a `judgements.json` artifact (every sampled row: the
  model prediction + the parsed judge verdict) and the `judge_prompt.md` used.
- **RESULTS store** — `eval_run` (one row per invocation: stage, timestamps,
  bucket counts, judge model, `code_version`, `mlflow_run_id`, the metrics blob,
  `status`) and `eval_judgement` (one row per sampled article). DDL in
  `news_nlp.schema`; `init_schema` creates them. `GET /eval/latest` on the
  FastAPI service returns the newest `eval_run` per stage.

## CI

`ci.yml` installs `--group eval` and runs the hermetic `tests/news_nlp/
test_eval_*.py` (the LLM and MLflow are mocked / a tmp file store). The **real**
eval is `.github/workflows/eval.yml` — `workflow_dispatch` + a weekly schedule,
with `LLM_*` repo secrets and `--check-regression`. It needs the actual crawl /
results databases, which live on the working machine, so it is written as a
self-hosted-runner template; the reliable path is `uv run cli/news_nlp_eval.py`
on a local cron with `MLFLOW_TRACKING_URI` pointed at a persistent store.

## Prompts

`src/news_nlp/eval/prompts/{sentiment,category,ner,c_summary}.md` — judge SOPs,
each ending with a strict "reply with ONLY a raw JSON object" instruction.
`category.md` embeds the 10-label taxonomy; keep it in sync with
`news_nlp.taxonomy` / `docs/category-taxonomy.md` if the labels change.
