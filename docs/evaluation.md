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
full `body_text` in practice — capped only at a generous 100,000-char safety
ceiling (`_SENTIMENT_MAX_BODY_CHARS`, ~2.9x the longest article observed as of
this fix), so a future pathological body can't produce an oversized judge
request that fails outright rather than just losing some tail context.
`category` is unaffected (its cap is correct and intentional —
`run_category_stage` deliberately classifies only the lead chunk, per
`docs/modules/news-nlp.md`).

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

### How much of the sentiment run-to-run swing is noise? (2026-09-08)

An unseeded re-run after the text-scope fix (mlflow `775d89b9`, same
1000-row sample size) landed `recall_negative` at 0.5566, down from 0.6198 —
looked at first like the fix regressed the metric it was supposed to be
gated on. Investigated with three checks, using the fact that the
low-confidence 600 rows are the *same 600 article ids* in both runs
(deterministic `ORDER BY score ASC`, and `article_sentiment` wasn't re-run):

1. **Judge non-determinism is small.** Of the 458 low-conf rows whose article
   body is ≤6000 chars (byte-identical text in both runs — the fix only
   changes what the judge sees on longer articles), the judge's `ideal_label`
   flipped on only 2.4% (11/458) of identical inputs. Not the driver.
2. **The low-conf bucket skews toward exactly the articles the fix touched.**
   23.7% of the low-conf 600 exceed 6000 chars, vs. 12.5% of the random 400 —
   because FinBERT's lowest-confidence scores correlate with longer,
   more-chunked articles (chunk-averaging drags a 3-way softmax toward
   uniform as chunk count grows). So the low-conf bucket's own
   `recall_negative` moved more (0.420→0.348) than the random bucket's
   (0.761→0.717): it's disproportionately exposed to the fix's effect on
   long articles (see the "Follow-up" section above — giving the judge more
   text gave it more material for net-signal reasoning, not less
   disagreement).
3. **The sample just doesn't have enough true negatives for a stable
   estimate.** `negative` is a minority *true* class: only 121 of 1000 rows
   (baseline) / 106 of 1000 (new run) are actually negative per the judge,
   even though FinBERT predicts `negative` on ~39% of rows. A Wilson 95% CI
   on `recall_negative` at those sample sizes is **[0.531, 0.701]** (baseline)
   vs. **[0.462, 0.648]** (new run) — half-width ~0.09 either way, and the two
   intervals overlap substantially. The observed 0.063 point "drop" is not
   distinguishable from sampling noise at the current `--sample-size`.

**What would it take to tighten this?** Halving the CI to ±0.05 needs ~362
true negatives. At the current ~11% true-negative-per-row yield, that means
scaling `--sample-size` to **~3,200** under today's unstratified 60/40
design — expensive in judge calls for a fix that mostly buys precision on
classes you don't need it for. The corpus has ample room to do this more
cheaply instead: `article_sentiment` currently holds 459,112 scored rows,
188,822 (41.1%) of them labelled `negative` — a sample specifically
stratified to pull more from FinBERT's own `negative` predictions (not just
its lowest-confidence rows, which is what `_low_conf_ids` does today) would
raise the true-negative yield per row without inflating `--sample-size`
nearly as much. **Not implemented** — a sampling redesign, flagged here for a
follow-up decision rather than made unilaterally.

**One real (if currently benign) code gap found and fixed along the way:**
`_all_ids()` (`src/news_nlp/eval/sampling.py`) built the random bucket's
candidate pool with no `ORDER BY`, relying on SQLite's incidental table-scan
order for the `--seed`-reproducibility this doc documents and
`test_sampling_is_deterministic_under_seed` asserts. That order isn't
SQL-guaranteed, so the seeded-reproducibility contract wasn't actually
airtight — it happened to hold because the table hadn't been reorganized.
Hardened with an explicit `ORDER BY article_id`. This wasn't the cause of the
swing analyzed above (neither run passed a seed), but matters now that this
doc recommends `--seed` for comparisons.

### Follow-up (2026-09-08): stratified sampling implemented

The section above flagged, but didn't implement, a sampling redesign to fix
the true-negative scarcity behind sentiment's wide `recall_negative` CI. It's
now implemented — see "Sampling" and "Statistical methodology" below for the
design (a `target_negative` stratum, soft-probability-targeted, on top of the
existing `low_conf`/random split) and "Sample-size floor" for updated,
measured-not-just-estimated `--sample-size` guidance. The historical numbers
in the "Baseline" and "How much of the sentiment run-to-run swing is noise?"
sections above predate this and were produced by the old two-bucket design —
left as-is (not rewritten) as the historical record.

### Pilot run and why this error shape is the right one (2026-09-09)

First stratified-sampling run (`--stage sentiment --sample-size 800 --seed 1`,
eval_run 18 / mlflow `c8c69f48`, `code_version` `9bca8da`). Strata drawn (and
their population sizes, from `strata_json`): `low_conf` 160/459,112,
`target_negative` 230/213,604, `target_positive` 77/91,485, `target_neutral`
77/226,253, `representative` 256/458,568.

| class | precision | recall | F1 |
|---|---|---|---|
| negative | 0.359 | **0.783** | 0.493 |
| neutral | **0.733** | 0.526 | 0.613 |
| positive | **0.628** | 0.462 | 0.532 |

(`macro_f1_vs_judge` 0.546, up from `macro_f1_vs_judge_naive_pooled` 0.480 —
both well above the pre-redesign baseline's 0.401.)

**This is the error shape the redesign was aimed at, not an accident**:
excellent recall on `negative`, good precision on `positive` and `neutral`.
Three business reasons this specific trade-off is the one worth having,
tying back to "Why recall, not F1, for sentiment negative" above:

- **Risk aversion drives the negative-recall priority.** A missed negative
  (a real downgrade, lawsuit, or demand-weakness signal the pipeline never
  surfaces) is a blind spot in whatever leans on this data for risk
  flagging — silent, and only discovered after the fact. An over-flagged
  negative (a neutral article mislabeled negative) is a false alarm — visible
  immediately as noise, and cheap to dismiss on a second look. 78.3% recall
  means the pipeline is now catching roughly 4 out of 5 real negative-signal
  articles; the 35.9% precision that comes with casting that wide a net is
  the deliberate, cheaper side of the trade.
- **Confidence in opportunities requires precision, not recall, on
  `positive`.** Unlike negative sentiment (a risk flag you want to see even
  at the cost of noise), positive sentiment more directly informs *acting* on
  an opportunity — chasing a false positive costs more than missing a true
  one, since capital gets deployed on the strength of the signal, not just
  attention. 62.8% precision means when the pipeline says "positive," it's
  right close to two-thirds of the time; that's the number that matters more
  than positive's recall (46.2%) here.
- **Neutral needs to be identified, not just left over.** Neutral is not "we
  had nothing better to say" — it's the pipeline actively recognizing
  administrative/non-material news (policy updates, routine filings, general
  market commentary not about a specific company's performance) so it can be
  triaged out rather than treated as a signal at all. 73.3% precision on
  neutral means a "neutral" label is trustworthy for that filtering job; its
  lower recall (52.6%) mostly reflects negative's wide net pulling some
  genuinely-neutral articles across the boundary — the same deliberate trade
  as above, not a separate problem.

Net: the pipeline is tuned to over-warn on risk and under-claim on reward,
which is the asymmetry a portfolio-construction consumer of this data should
want. `precision_negative`/`recall_positive`/`recall_neutral` are the numbers
to watch for future regressions in the *other* direction (e.g. negative
precision collapsing further, or positive precision dropping) — see
`--check-regression`, still gated on `recall_negative` per `metrics.HEADLINE`,
but these companion numbers are worth eyeballing on every run, not just the
gate.

### Follow-up (2026-09-09): hierarchical category classification

Investigation into the category stage's own worrying per-slug numbers
(`acc_product_innovation` 0.14-0.21, `acc_partnerships_business_dev`
0.16-0.22, `acc_leadership_governance` 0.28-0.33 in the baseline/5000-row
runs above) found the cause was primarily the classifier's flat 9-way
softmax diluting real signal, not unbalanced data or a sampling artifact:
recall on those three slugs stayed low at n=359/170/85 true examples across
two differently-sized runs (rules out small-sample noise), and the model's
own raw score for the correct slug on missed rows averaged only 0.14-0.16 —
barely above the 9-way uniform baseline of 0.111 (rules out threshold
miscalibration as the primary fix; <5% of misses were even close to the 0.4
bar). `run_category_stage` now classifies through `news_nlp.taxonomy.
CATEGORY_GROUPS`' two-level hierarchy instead — see
`docs/category-taxonomy.md`'s "Hierarchical classification" section for the
full design.

This changes `article_category`'s score-column semantics for **future**
pipeline runs: the 9 leaf-slug columns are now populated from 3-way
child-group softmaxes (baseline ~0.333) for whichever slugs' group made an
article's top-2, not a flat 9-way softmax (baseline ~0.111) — raw magnitudes
aren't directly comparable across this boundary. `_CATEGORY_TARGET_THRESHOLD`
(the stratified-sampling `target_<slug>` mechanism above) was raised from 0.2
to 0.35 accordingly — the old value sat *below* the new no-signal baseline,
which would have defeated that stratification's near-miss selectivity for
any slug whose group ran level 2.

The 2026-09-08 baseline table and per-slug notes above predate this change
and stay as the historical record for the old flat-classifier design — not
rewritten. A fresh category eval pilot is needed post-merge for numbers
comparable to the new design (see `docs/category-taxonomy.md`'s "Hierarchical
classification" for the measured-not-guessed threshold caveat).

### Follow-up (2026-09-09): first post-hierarchy pilot and threshold calibration

First eval run under the hierarchical classifier (`--stage category
--sample-size 2800 --seed 1`, eval_run 19 / mlflow `8c470ea1`, against a
100,152-article batch freshly reclassified under `feat/hierarchical-category-
classification`). The redesign's own target confirmed working:

| slug | old (flat 9-way) | new (hierarchical, launch threshold 0.4) |
|---|---|---|
| `product_innovation` | 0.14-0.21 | 0.613 |
| `partnerships_business_dev` | 0.16-0.22 | 0.510 |
| `leadership_governance` | 0.28-0.33 | 0.526 |

But the headline `accuracy_vs_judge` (0.442) came in *below* the old
baseline (0.687-0.711) — `acc_other` collapsed from ~0.80-0.83 (previously
the best-performing class) to 0.215 (the worst): 701 of 1,322
judge-confirmed true-`other` articles got assigned a specific wrong label
instead. Checked before concluding anything: those 701 weren't
near-threshold misses (mean/median winning score 0.661/0.641, only 22.7%
even close to 0.4) — reducing per-decision competition to rescue real
signal for the weak target categories also let spurious signal through for
genuinely generic/ambiguous articles the old, stricter 9-way contest used
to correctly route to `other`. Level 1 was checked and ruled out too: even
correctly-resolved true-`other` articles clear `CATEGORY_GROUP_FLOOR`
comfortably (mean group_score 0.50) — the gap was at level 2's threshold,
not level 1's floor.

Action taken: raised `CATEGORY_CONFIDENCE_THRESHOLD` 0.4→0.6 (full
before/after math in `docs/category-taxonomy.md`'s "Threshold calibration"
section) — checked to recover 43% of the false-`other` losses at an
8-18% cost to the newly-won target-category recall before making the
change, not applied blind. A fresh pilot post-calibration is the next step
to confirm the trade landed as estimated; this run's numbers (both the
per-slug wins and the `accuracy_vs_judge`/`acc_other` collapse) predate the
calibration and stay as the historical record of why it happened, not
rewritten.

## What it evaluates

Four per-article stages. `sector_summary` is out of scope — it is deterministic
composition; only its one-sentence intro seed is generative.

| stage | headline metric | also logged |
|---|---|---|
| `sentiment` | `recall_negative`¹ ² | agreement rate (per stratum), `macro_f1_vs_judge`, per-class P/R/F1, mean severity |
| `category` | `accuracy_vs_judge` ² | macro-F1, per-slug accuracy, model vs judge `other`-rate, mean severity |
| `ner` | `micro_f1` | span micro/macro P/R/F1, per-type F1, hallucination rate, miss rate. Error-only judge contract: it names just the `wrong` predicted spans + `missed` entities (not a verdict per span, which overflows on entity-dense articles); TP/FP/FN are derived from the predicted count. |
| `c_summary` | `mean_faithfulness` ² | mean coverage / conciseness (1-5), `pct_with_hallucination` |

Every run also logs `n` (rows judged) and `parse_fail_rate` (judge replies that
were not valid JSON after one repair attempt — excluded from the accuracy
numbers).

¹ Not F1 or accuracy: a missed real negative-sentiment article costs more for
portfolio construction than an over-flagged neutral one, so `--check-regression`
gates sentiment on recall specifically — see "Why recall, not F1, for
sentiment negative" above.

² Every headline/"also logged" metric marked here has a `_naive_pooled`
companion (e.g. `recall_negative_naive_pooled`, `accuracy_vs_judge_naive_pooled`,
`mean_faithfulness_naive_pooled`) — the old flat pooled computation, kept for
sanity-checking the Horvitz-Thompson reweighting below and for continuity with
pre-redesign runs. `ner`'s metrics have no such companion because they're
mathematically identical to it already (see "Statistical methodology" below).

## Sampling: low-confidence + targeted + representative strata

Each run draws `--sample-size` rows per stage as a **disjoint,
priority-ordered stack of strata** (`src/news_nlp/eval/sampling.py`), each
excluding every earlier-priority stratum's already-drawn ids:

1. **`low_conf`** (`--low-conf-frac` of `--sample-size`, default 0.2) — the
   *least-confident* stored rows, deterministically, so every run re-checks
   the true worst case: lowest `article_sentiment.score`; category picks
   within ±0.1 of `CATEGORY_CONFIDENCE_THRESHOLD` (0.6 as of the 2026-09-09
   calibration — see "Follow-up: hierarchical category classification"
   above) or labelled `other`;
   lowest per-article `MIN(article_entities.score)`; for `c_summary` (no
   score), articles whose sentiment/NER inputs were themselves low-confidence.
   **Diagnostic-only**: deliberately biased toward hard cases, so it's
   excluded from every headline/population-estimate metric (see "Statistical
   methodology" below) — reported only as a `*_low_conf` per-stratum split.
2. **`target_<x>`** (`--target-frac` of the post-`low_conf` budget, default
   0.6; sentiment and category only) — rows clearing a threshold on a *raw
   per-class score*, regardless of which class won the argmax:
   - **sentiment**: `target_negative` (weight 0.6, `negative >= 0.35`),
     `target_positive` (0.2, `positive >= 0.35`), `target_neutral` (0.2,
     `neutral >= 0.35`). `negative` is weighted highest — it's the headline
     class.
   - **category**: one stratum per the 6 worst 2026-09-08-baseline per-slug
     accuracies plus `legal_regulatory`, all thresholded at `>= 0.35` (raised
     from an original 0.2 once the hierarchical category classifier changed
     the leaf-score baseline from ~0.111 to ~0.333 — see the "hierarchical
     category classification" follow-up above), weights summing to 1.0:
     `partnerships_business_dev` 0.20, `labor_human_capital` 0.18,
     `leadership_governance` 0.16, `mergers_acquisitions` 0.14,
     `capital_shareholder_returns` 0.13, `product_innovation` 0.12,
     `legal_regulatory` 0.07.
   - **c_summary** has no discrete classes, so its "targets" instead
     partition on `num_chunks` (a TRUE partition — every row has exactly
     one, so c_summary draws no `representative` bucket at all): `1` (weight
     0.10), `2` (0.30), `>= 3` (0.60) — weighted toward the rare multi-chunk
     tail (~4.6% of the corpus) since `mean_coverage` is the weakest
     c_summary metric and multi-chunk articles pass through more
     hierarchical-reduce steps.
   - **ner** has none: its type imbalance (ORG 55% / PER 25% / LOC 20% of
     spans) is far milder than sentiment's/category's article-level
     imbalance, and there's no secondary per-entity score to target.
   - **Why these thresholds are well below each stage's winning bar** (0.5
     for sentiment's 3-way softmax; `CATEGORY_CONFIDENCE_THRESHOLD` 0.6 for
     category): a distribution over mutually exclusive classes can have at
     most one class exceed 0.5, so a threshold at or above that would
     mathematically exclude every false-negative candidate for that class —
     silently defeating the whole point (catching rows the model *almost*
     called this class but didn't, not just rows it did call this class).
3. **`representative`** (the remainder) — a `--seed`-reproducible uniform
   draw from whatever's left. The only bucket that's a plain, unweighted
   random sample of the population (`low_conf` and `target_<x>` are each
   deliberately non-representative by construction).

Metrics are reported both as a population estimate (`headline_metric`,
Horvitz-Thompson-reweighted across every non-`low_conf` stratum — see below)
and per-stratum (`*_low_conf`, `*_target_negative`, `*_representative`, …),
so you see both the population number and where it's weak.

**Pass `--seed` when comparing two runs.** With no seed, `target_<x>` and
`representative` are fresh OS-entropy draws every time, so two unseeded runs
differ in both *which* articles were judged and (per the LLM) how they were
judged — any delta between them mixes real signal with resampling noise, and
you can't tell how much of each. `low_conf` is already deterministic (same
rows every run, seed or not) since it's a fixed `ORDER BY ... LIMIT`; a shared
`--seed` makes the rest deterministic too, so a before/after comparison (e.g.
across a pipeline or eval-harness change) isolates the change's effect instead
of conflating it with which rows happened to get drawn. See "How much of the
sentiment run-to-run swing is noise?" below for a worked example of how large
that conflation can be.

**Migration note**: a run against pre-redesign code isn't seed-reproducible
against this code even with the same seed — drawing `target_<x>` strata
inserts extra RNG draws before the final `representative` shuffle. `eval_run`
rows from before this redesign have `strata_json = '{}'`.

## Statistical methodology: Horvitz–Thompson reweighting

Naively pooling a deliberately-oversampled `target_<x>` stratum with
`representative` would be just as statistically invalid as the old design's
`low_conf`+`random` pooling (see the "run-to-run swing" section above for why
that mattered). The fix is a standard stratified-sampling ratio estimator.

For any population total (e.g. total true negatives across the whole
corpus), each non-excluded stratum `h` contributes `(N_h / n_h') *
(count within h)`, where `N_h` is that stratum's population size (computed
from the exact SQL result the sample was drawn from, never a second
independently-written count) and `n_h'` is the count of rows *actually
judged* in that stratum (post-parse-failure — a deliberate MCAR assumption:
a judge JSON-parse failure is treated as independent of the row's true
label, so survivors of stratum `h` are still ~a uniform sample of size `n_h'`
from `N_h`; this is an accepted, unverified risk, not solved). A ratio metric
(recall, precision, accuracy, a mean) is the ratio of two such weighted sums.
`low_conf` is always excluded — it's diagnostic by design, never part of a
population estimate.

Implementation: `_ht_sum` / `_ht_ratio` / `_prf_ht` / `_macro_f1_ht` in
`src/news_nlp/eval/metrics.py`. Worked example (also a unit test,
`test_ht_sum_and_ht_ratio_hand_computed_example`): stratum A `N=100,n=4`
(TP=2, FN=1), stratum B `N=900,n=6` (TP=1, FN=0):

```
TP_hat = (100/4)*2 + (900/6)*1 = 200
FN_hat = (100/4)*1 + (900/6)*0 = 25
recall_hat = 200 / 225 = 0.8889          (vs. naive pooled 3/4 = 0.75)
```

`ner` needs no such reweighting: with only `representative` as a non-excluded
stratum, `N_h/n_h` is a constant that cancels identically in any ratio, so
`aggregate_ner` is mathematically unchanged from the old pooled computation —
confirmed by leaving its code untouched rather than adding a no-op HT call.

## Sample-size floor

`n_stratum(class c) = n_true_target(c) / purity_target(c)`, where
`n_true_target` is the Wilson-CI true-example count needed for a target
recall/precision confidence interval (±0.05 needs ~362 true examples at
sentiment's ~55-62% base recall; category's worst slugs need ~200-370), and
`purity_target` is the fraction of a threshold stratum a judge actually
confirms as class `c`. Below are *planning* estimates from population share +
baseline precision, not measured purity — re-solve with a pilot run's actual
per-stratum agreement before locking in a permanent default (a pilot's
`agreement_rate_target_negative` etc., or the `strata_json` blob, gives you
the real `n`/`N` to recompute this from).

- **Sentiment**: `negative >= 0.35` population is 213,604 (46.5% of the
  corpus) vs. argmax-`negative`'s 41.1% baseline precision of 19.3% — purity
  should land a bit below that (looser threshold, more true negatives).
  Planning range 15-19% → `n_stratum ≈ 1,905-2,413` — a real but modest
  ~30-45% reduction vs. the ~3,289 an unstratified design would need (the
  threshold is deliberately loose to keep catching FN candidates, which
  dilutes purity by design).
- **Category**: e.g. `partnerships_business_dev >= 0.2` population is 24,390
  (5.31%, vs. 2.95% true share) → purity roughly 6-12% → `n_stratum ≈
  1,667-6,167`. Because the 7 target strata **share one weighted budget**
  rather than running independently, hitting the single worst slug's own
  ±0.05 target inside one regular run can still need a total `--sample-size`
  in the same range the unstratified design would — an honest result, not
  oversold. **Predates the hierarchical category classifier** (the follow-up
  above) and its `>= 0.2` → `>= 0.35` threshold change — both the population
  count and the threshold in this bullet are stale; re-derive against the new
  classifier's actual score distribution once enough articles have been
  reclassified under it, rather than trusting this number as-is.
- **Recommendation** (two-tier, not one asserted number):
  1. Regular regression-tracked runs: `--sample-size` floor **~1,800-2,200**
     for sentiment, **~2,500-3,000** for category (targets a looser
     ±0.07-0.10 CI — still tight enough to catch a real move against the
     existing 0.05 `--regression-tolerance`).
  2. Occasional deep-dive runs (manual): 5,000+, or a temporary
     single-slug-weighted `_CATEGORY_TARGET_WEIGHTS` override, when chasing a
     specific rare-class regression.
  3. Before locking in either number: run a pilot at `--sample-size 800`
     under this design, read the actual purity per `target_<x>` stratum from
     that run's metrics, and re-solve the formula above with measured (not
     estimated) purity.
  4. `ner` and `c_summary` need no floor increase — `ner`'s stratification is
     unchanged, and c_summary's `target_chunks_ge3` (4.6% of the corpus)
     already gets outsized attention (weight 0.6) at today's sizes.

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

# a regression-tracked run at the recommended sentiment sample-size floor
uv run cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1

# fail (exit 1) if a headline metric dropped > 0.05 vs the previous MLflow run
uv run cli/news_nlp_eval.py --stage all --check-regression

uv run mlflow ui            # browse runs at http://127.0.0.1:5000
```

Flags: `--stage` (repeatable; `all` = every stage), `--sample-size`,
`--low-conf-frac` (default 0.2), `--target-frac` (default 0.6, no-op for
`ner`), `--seed`, `--max-workers` (concurrent judge calls, default 4),
`--source-db` / `--results-db` (override `$SOURCE_DATABASE_URL` /
`$DATABASE_URL`), `--mlflow-uri`, `--check-regression`,
`--regression-tolerance` (default 0.05).

## Where results go

- **MLflow** — one run per `(stage, invocation)` in experiment
  `news_nlp_eval/<stage>`. Params (judge model/url, sample size, per-stratum
  `n_<bucket>` counts, seed, git SHA), metrics, a `judgements.json` artifact
  (every sampled row: the model prediction + the parsed judge verdict) and the
  `judge_prompt.md` used.
- **RESULTS store** — `eval_run` (one row per invocation: stage, timestamps,
  `low_conf_n`/`random_n` — the latter now "every non-`low_conf` stratum
  combined" — `strata_json` (`{bucket: {"population": N_h, "n": n_h}}`, the
  Horvitz-Thompson bookkeeping; `'{}'` for pre-redesign rows), judge model,
  `code_version`, `mlflow_run_id`, the metrics blob, `status`) and
  `eval_judgement` (one row per sampled article, `bucket` now one of
  `low_conf` / `representative` / a stage-specific `target_<x>`). DDL in
  `news_nlp.schema`; `init_schema` creates them (additive migration for
  `strata_json` on a pre-existing table). `GET /eval/latest` on the FastAPI
  service returns the newest `eval_run` per stage.

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
