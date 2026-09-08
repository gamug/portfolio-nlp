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

## What it evaluates

Four per-article stages. `sector_summary` is out of scope — it is deterministic
composition; only its one-sentence intro seed is generative.

| stage | headline metric | also logged |
|---|---|---|
| `sentiment` | `macro_f1_vs_judge` | agreement rate (overall / low-conf / random), per-class P/R/F1, mean severity |
| `category` | `accuracy_vs_judge` | macro-F1, per-slug accuracy, model vs judge `other`-rate, mean severity |
| `ner` | `micro_f1` | span micro/macro P/R/F1, per-type F1, hallucination rate, miss rate. Error-only judge contract: it names just the `wrong` predicted spans + `missed` entities (not a verdict per span, which overflows on entity-dense articles); TP/FP/FN are derived from the predicted count. |
| `c_summary` | `mean_faithfulness` | mean coverage / conciseness (1-5), `pct_with_hallucination` |

Every run also logs `n` (rows judged) and `parse_fail_rate` (judge replies that
were not valid JSON after one repair attempt — excluded from the accuracy
numbers).

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
