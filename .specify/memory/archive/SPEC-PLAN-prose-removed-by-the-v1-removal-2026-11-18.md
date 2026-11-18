# SPEC and PLAN prose replaced by T-220 PR B (2026-11-18), verbatim

The paragraphs that described the v1 chain as current. Replaced, not annotated (INDEX rule 3). The FR rows are in `SPEC-rows-removed-by-the-v1-removal-2026-11-18.md`.

## SPEC section 3, the DB layer bullet (corrections)

```
schema/queries/corrections/taxonomy directly;
```

## SPEC section 3, the fixed-pipeline bullet (`run_pipeline`)

```
- **Stage stack is a fixed pipeline, not a DAG/orchestrator.** Five stages
  run in a fixed order (`run_pipeline`, `src/pipeline.py`) inside one
  process; no task queue, no distributed execution. Justified by corpus
  scale and the "one model resident at a time" VRAM constraint (NR-001).
```

## SPEC section 3, the pending architecture decision (FTI in `pipeline.py`)

```
**Pending architecture decision (§13 item 15, FR-011–FR-013, not yet
built)**: the four ML stages currently ship as independent sets of
module-level functions in `pipeline.py`, never designed against a stated
pattern. `PLAN.md` Work item 10 formalizes a shared Feature/Train/Inference
(FTI) class hierarchy across all four — an internal restructuring, not a
change to the fixed-pipeline/single-process decision above, which stays as
stated. `sector_summary` is explicitly excluded from this hierarchy (it has
no Feature/Train/Inference shape — FR-012) and moves fully into its own
module instead.

```

## SPEC section 4, the code of today (v1)

```
**The code of today (v1)**, until the backfill (T-150): `run_pipeline` runs sentiment, NER and category over every article in one
process; its diagram and reading are in `archive/PIPELINE-superseded-2026-11-10.md` (section 4).

```

## SPEC section 5, the intro sentence

```
All five result
tables key off `article_id INTEGER REFERENCES articles(id)`;
```

## SPEC section 5, the SOURCE schema contract table (`sector_summary` reads)

```
What each stage actually reads from SOURCE, and what `require_source_text`
does (and does not) verify before a run — pulled from
`src/news_nlp/sector_summary/queries.py` and `news_nlp.db`, not assumed:

| Column | Read by | Checked by `require_source_text`? | Behavior if missing/NULL |
|---|---|---|---|
| `id` | every stage (join/FK target) | implicitly (primary key) | — |
| `body_text` | stages 1–4 (sentiment/NER/category/`c_summary`) | ✅ yes — the run aborts if the column is absent or every row is empty/blank (FR-007) | pipeline never starts |
| `ticker`, `company` | stage 5 (`sector_summary`, per-company attribution) | ❌ no | attribution text for that company is whatever the column holds, including `NULL` |
| `gics_sector`, `gics_sub_industry` | stage 5 (buckets articles into `(sector, sub_industry, week)` groups) | ❌ no | a row with either `NULL` is **silently excluded** from every `sector_summary` bucket (`WHERE a.gics_sector IS NOT NULL AND a.gics_sub_industry IS NOT NULL`) — no error, no log line |
| `pub_date` | stage 5 (week-bucketing key) | ❌ no | falls back to `fetched_at` when `NULL` (`COALESCE(a.pub_date, a.fetched_at)`) — not a hard requirement |
| `fetched_at` | stage 5 (fallback only) | ❌ no | only consulted when `pub_date` is `NULL` |

**Only `body_text` is a validated contract.** Everything `sector_summary`
depends on is assumed, not checked — a SOURCE missing `gics_sector`/
`gics_sub_industry` doesn't fail the run, it just quietly produces a smaller
(or empty) `sector_summary` for that period (§13, open question 3).

```

## SPEC section 6, serving, human correction and accuracy evaluation workflows

```
**Serving/query** (FastAPI, no SOURCE needed): a request opens a RESULTS-only
connection (`db.connect`, no `ATTACH`), reads/writes the result tables
directly. `POST /pipeline/run` kicks off an async pipeline run and
`GET /pipeline/status` polls an in-process progress tracker (not persisted
across process restarts).

**Human correction**: a caller `PATCH`es a sentiment/entity/category
endpoint → `news_nlp.corrections.update_*` (Allowlist-checked field names,
`processed_at` refreshed) or `DELETE`s it → the article becomes "pending"
again for that stage's next pipeline run (FR-009).

**Accuracy evaluation** (`cli/news_nlp_eval.py`, separate from the batch
pipeline, needs `$LLM_API_KEY`/`$LLM_MODEL`/`$LLM_URL`): sample
already-processed rows per stage (stratified — a low-confidence bucket plus
a representative random bucket, see `docs/evaluation.md`), send each to the
judge LLM, record `eval_run`/`eval_judgement`, log to MLflow.
`--check-regression` compares against a prior run's `metrics_json` and can
fail the invocation (used as a manual/scheduled gate, not part of `ci.yml`).

```

## SPEC section 6, the v1 CLI clause

```
; the v1 CLI is not changed); a step opens
```

## SPEC section 6, the code of today (v1) sentence

```
 **Code of today (v1)** (`cli/news_nlp_cli.py` ->
`pipeline.run_pipeline`: sentiment, NER, category, then optional summaries, in one process): the steps are in
`archive/PIPELINE-superseded-2026-11-10.md` (section 6) until the backfill (T-150).
```

## SPEC section 7, the sentiment bullet (v1 FinBERT per chunk)

```
- **Sentiment**: FinBERT scored per chunk, aggregated as a token-weighted
  average across the *entire* `body_text` (not just a lead chunk) — a
  deliberate choice, unlike category (see below); the judge is scoped the
  same way as of the 2026-09-08 fix (`docs/evaluation.md`).

```

## SPEC section 7, the category bullet (v1 two-level zero-shot)

```
- **Category**: two-level hierarchical zero-shot NLI — level 1 scores 3
  group hypotheses, level 2 scores only the leaf labels under the article's
  top-2 groups (6 of 9 candidates, not all 9) against
  `CATEGORY_PREMISE_MAX_TOKENS`-bounded title+lead-chunk premises. Chosen
  over a flat 9-way pass because it raised true-positive recall on
  under-served labels (`product_innovation`,
  `partnerships_business_dev`, `leadership_governance`) at an accepted cost
  in false-`other` reduction — see the calibration trade-off recorded in
  `news_nlp/taxonomy.py` and `docs/legacy/category-taxonomy-v1.md`.

```

## SPEC section 7, the `c_summary` gating bullet (v1)

```
- **`c_summary` gating**: only articles with a sentiment row **and** an
  entity scoring `> 0.8` are summarized — a deliberate precision-over-recall
  filter (constitution: AI behavior #4, "decision-support not authoritative
  claims" — don't summarize what the pipeline itself is unsure contains a
  notable entity).

```

## SPEC section 7, the `sector_summary` composition bullet (v1)

```
- **`sector_summary` composition is fully deterministic, not generative**
  (as of 2026-09-14 — previously true for everything except one intro
  sentence): `build_sector_facts` aggregates structured stats
  (`facts_json`); member companies' `c_summary` text is copied verbatim
  and ticker-attributed; `build_sector_intro_seed`'s own templated
  sentence, built purely from aggregate statistics, IS `intro_text` —
  no ticker, company name, or summary text ever reaches it, and no model
  call happens at all, making cross-company blending structurally
  impossible rather than merely policy (FR-005).

```

## SPEC section 11, the eval.yml sentence

```
`.github/workflows/eval.yml` runs the real
   LLM-as-judge evaluation separately and does not block merges.
```

## PLAN D40 (viii) step list, the v1 CLI sentence

```
the v1 `cli/news_nlp_cli.py` is not touched until the backfill (T-150)
```

## SPEC section 8, the v1 pipeline error-handling sentence

```
(There is currently no
  per-article try/except-and-skip in `pipeline.py`; a single malformed row
  failing a stage fails the whole run — see §13 for whether that's the
  right trade-off at scale.)
```

## SPEC section 8, the corrections Allowlist bullet

```
- `news_nlp.corrections`' `Allowlist.check()` raises `ValueError` on any
  field name outside a function's explicit allowed set — a caller cannot
  silently no-op or corrupt an unrelated column via a typo'd field name.

```

## SPEC section 10, the coverage mapping bullet

```
- **Coverage mapping**: `test_schema.py`/`test_db.py`/`test_corrections.py`/
  `test_queries.py`/`test_sector_summary.py`/`test_env.py` are the vendored
  `news_nlp` package's own unit tests (FR-006/007/009); `test_two_tier.py`
  exercises the two-tier contract through `pipeline.run_pipeline` and the
  FastAPI app (FR-007/008); `test_eval_*.py` cover `news_nlp.eval` with the
  LLM and MLflow mocked (FR-010).

```

## SPEC section 10, the accuracy regression bullet

```
- **Accuracy regression** (separate from `pytest`): `news_nlp_eval
  --check-regression` against the recorded baseline (§9) is a manual/
  scheduled gate today, not part of `ci.yml` — see §13 on wiring it in.

```

## SPEC section 10, reproducing and validating the accuracy baseline

```
### Reproducing and validating the accuracy baseline

To re-check a stage against the §9 baseline after a model, prompt, or
threshold change (needs `$LLM_API_KEY`/`$LLM_MODEL`/`$LLM_URL`; no GPU
required — this reads already-stored predictions, it doesn't re-run the
pipeline):

```bash
# one stage
uv run cli/news_nlp_eval.py --stage sentiment --sample-size 100

# every stage the §9 baseline covers, failing if a headline metric dropped
# more than --regression-tolerance (default 0.05) vs. the previous MLflow run
uv run cli/news_nlp_eval.py --stage all --check-regression
```

`--check-regression` compares against **the previous MLflow run** under
`news_nlp_eval/<stage>` (`--regression-tolerance`, default `0.05`) — there is
no separate baseline file to maintain, and none should be invented; the
baseline of record is the run IDs already logged in `docs/evaluation.md`.
This remains a manually-invoked gate (§13, open question 8) — see that
section for the tradeoffs of wiring it into `ci.yml`.

```

## SPEC section 11, the run steps (v1 API and CLI)

```
4. Run `uv run apps/news_nlp_api.py` (long-lived service, `:8003`) and/or
   invoke `uv run cli/news_nlp_cli.py` on whatever cadence the batch run
   needs (no scheduler is wired in today — see §13).
```

## SPEC FR-011, acceptance (`run_pipeline`)

```
`run_pipeline`'s behavior is unchanged and the hermetic tests pass.
```

## SPEC FR-015, acceptance (`--stage <s> --run-name`)

```
The same `--stage <s> --run-name <experiment>` run twice on an unchanged sample makes zero new judge calls the second time (the judge client is never invoked for a key already stored); a new `article_id`, a different `experiment`, or an unchanged `experiment` whose prediction changed judges fresh, and a recorded verdict always matches its own run's prediction.
```

## SPEC NR-005, acceptance (`import pipeline`)

```
`import pipeline` / `import apps.news_nlp_api` succeeds in an environment with only the base `[project]` dependencies installed (no `eval` group) — `news_nlp.eval.run_eval` is imported lazily.
```

## SPEC FR-019, the news_nlp.eval sentence

```
the same metrics are already logged per class by `news_nlp.eval` for sentiment and category.
```

## SPEC section 1, the traceability example

```
(`test_two_tier.py::test_source_readonly  # FR-007`)
```

## SPEC FR-011, the row as it stood on master

```
| **FR-011** | Every ML-based stage (sentiment, NER, category, `c_summary`) is built on a shared Feature/Train/Inference (FTI) class hierarchy (`src/fti.py`); a stage with no real training (category, `c_summary`) has a documented no-op `Train`. | One abstract base per FTI component, a concrete subclass of all three in each of the four stages; `run_pipeline`'s behavior is unchanged and the hermetic tests pass. Full text: `archive/SPEC-condensed-for-the-pipeline-api-2026-11-17.md`. |
```

