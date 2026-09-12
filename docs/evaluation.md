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

### Follow-up (2026-09-09): corrected post-calibration numbers, and per-slug precision/recall/F1

eval_run 19 above was judged *before* the `CATEGORY_CONFIDENCE_THRESHOLD`
0.4→0.6 calibration landed, so its stored `article_category` predictions
went stale the moment the threshold changed (28,618 rows in the eval run's
underlying batch flipped label to `other`). Rather than re-spend judge calls
on identical article text, the correction reused the fact that a judge's
verdict (`ideal_slug`, read from the article's *content*) never depended on
the model's prediction: **eval_run 20** (mlflow `2bdecd21`) re-paired eval_run
19's 2,800 stored judge verdicts against the *corrected* `article_category`
labels and re-ran `aggregate_category()` — same judged sample, zero new LLM
calls. Corrected headline: `accuracy_vs_judge` 0.442→**0.487**, `acc_other`
0.215→**0.545** (the specific number the calibration targeted). Several
individual slugs got worse in exchange (`capital_shareholder_returns`
0.407→0.293, `market_analyst_sentiment` 0.610→0.378, `mergers_acquisitions`
0.580→0.481) — the exact 8-18%-recall cost the calibration decision already
priced in, now visible per-slug instead of as an aggregate estimate.

`aggregate_category()` was already computing per-class precision/recall/F1
internally (inside `_macro_f1_ht`/`_macro_f1`, for the macro-F1 average) and
discarding everything but the scalar. Surfaced as real fields —
`precision_<slug>` / `recall_<slug>` / `f1_<slug>` (HT-weighted, +
`_naive_pooled` companions, mirroring `aggregate_sentiment`'s existing
per-class convention) plus `accuracy_ovr_<slug>` (one-vs-rest binary
accuracy: this slug vs. all 9 others collapsed into one negative class) —
because `accuracy_vs_judge`/`acc_other` alone can't tell "the model
over-triggers this slug" apart from "the model is too conservative but
trustworthy when it does fire." (`acc_<slug>`, the old field, is recall by
another name — kept for backward compat; `recall_<slug>` is the same number,
correctly named, proven identical in `test_category_per_slug_precision_
recall_and_ovr_accuracy`.) Re-aggregated as **eval_run 21** (mlflow
`0a18577e`) against the same eval_run 20 data — a metric-surface change, not
a data or prediction change, so again zero new judge calls:

| category | precision | recall | f1 | accuracy (one-vs-rest) |
|---|---|---|---|---|
| `capital_shareholder_returns` | **0.074** | 0.293 | **0.118** | 0.920 |
| `earnings_performance` | 0.585 | 0.508 | 0.544 | 0.961 |
| `labor_human_capital` | 0.561 | 0.651 | 0.602 | 0.958 |
| `leadership_governance` | 0.442 | 0.503 | 0.471 | 0.971 |
| `legal_regulatory` | 0.490 | 0.460 | 0.474 | 0.929 |
| `market_analyst_sentiment` | 0.565 | 0.378 | 0.453 | 0.771 |
| `mergers_acquisitions` | **0.926** | 0.481 | 0.633 | 0.980 |
| `partnerships_business_dev` | 0.466 | 0.449 | 0.457 | 0.963 |
| `product_innovation` | 0.619 | 0.509 | 0.559 | 0.939 |
| `other` | 0.474 | 0.545 | 0.507 | 0.581 |

Two standouts: `capital_shareholder_returns` (precision 0.074 — wrong 93% of
the times the model predicts it; likely confused with its own hierarchy
group-mates `earnings_performance`/`mergers_acquisitions`, not yet root-caused)
and `mergers_acquisitions` (the mirror image — precision 0.926 but recall
only 0.481, i.e. conservative rather than wrong). `other`'s own row is
discussed separately below — it isn't one of the 9 taxonomy slugs, but it's
in `classes` (every value that appears as either a judge verdict or a model
prediction) so `aggregate_category` computes it the same way.

Read `accuracy_ovr_<slug>` with the imbalance caveat baked into its own
docstring in `metrics.py`: it's a one-vs-rest binary call ("is this X or
not"), so a rare slug's accuracy is dominated by true negatives and reads
high (0.92-0.98) almost regardless of how good the model actually is at that
slug — `other`'s own 0.581 is the illustration: it's the *only* row where
the one-vs-rest split isn't lopsided (~40% prevalence vs. 3-15% for a single
specific slug), so its accuracy isn't inflated the same way and actually
tracks how hard the binary call is. Precision/recall are the metrics to read
for real signal; `accuracy_ovr` is a secondary check, not the headline.

### Why precision, not recall, for category (2026-09-09)

Sentiment's headline metric prioritizes recall (`recall_negative`) because a
missed real negative is a blind spot — costly, and there's no cheap fallback
once the article's been scored positive/neutral. Category is the opposite
shape: every article that isn't confidently a specific category already has
a safe fallback — `other`. A model that's too *cautious* about a specific
category (low recall, like `mergers_acquisitions` at 0.481) just leaves some
M&A articles sitting in the uncategorized pile — recoverable, low-cost,
correctable with a threshold retune. A model that's too *eager* (low
precision, like `capital_shareholder_returns` at 0.074) actively tells a
downstream consumer — a dashboard, a category filter, a portfolio-
construction rule keyed off `article_category.label` — that an article is
about capital returns when the judge says it almost certainly isn't. That's
not a gap, it's misinformation with a specific, actionable-sounding label
attached. **Being sure the label is right, when the model commits to one,
matters more here than catching every possible instance of that label** —
the inverse priority from sentiment, for the inverse reason (category has a
safe catch-all to fall back to; sentiment negative does not).

This is why `category`'s headline stays `accuracy_vs_judge`
(`metrics.HEADLINE["category"]`) rather than switching to a recall-style
average: exact-match accuracy penalizes a confidently wrong label exactly as
much as a missed one, so it can't be gamed by under-triggering everything
into `other` (which would tank recall-per-slug but wouldn't show up in a
recall-only metric the way it hurts `accuracy_vs_judge`). `precision_<slug>`
is the diagnostic lens for the business question in this section's title —
"when the model commits to a label, do we trust it" — read stage-wide via
`accuracy_vs_judge` and per-slug via the table above, not a metric to
optimize against `--check-regression` on its own (a model that never fires a
given slug has perfect, meaningless precision on it).

### The `other` bucket: a precision problem of its own (2026-09-09)

The precision-over-recall framing above treats "falls back to `other`" as
the safe failure mode — no false information reaches a downstream consumer.
That's only true if `other` itself is trustworthy, and right now it isn't
quite: `other`'s own precision is **0.474** — of the articles the model
labels `other`, the judge agrees for less than half of them. The other
52.6% are articles the judge says *do* have a real, specific category that
the model suppressed (a direct, now-quantified view of the same 0.4→0.6
threshold trade documented above: raising the bar to fix `acc_other`
necessarily means some genuinely-categorizable articles get pushed into
`other` too — that's exactly what a 0.474 precision reading looks like from
the other side).

Two implications, not fully aligned with each other:

1. **The failure direction still matches the business preference above.**
   A real category article landing in `other` degrades gracefully — a
   consumer sees "uncategorized," not a wrong specific claim. This is the
   same asymmetry the precision-priority argument rests on, and it's working
   as intended.
2. **But `other` can't yet be read as "verified no category."** A consumer
   treating `article_category.label == "other"` as ground truth "this
   article has no relevant business category" is wrong more than half the
   time. Until `other`'s own precision improves, the honest reading of
   `other` is "the model isn't confident enough to commit to a specific
   label" — closer to a low-confidence signal than a negative one. Anywhere
   downstream that filters *out* `other` articles as irrelevant should know
   that filter is discarding a substantial number of real category hits
   along with the true negatives.

No action taken on this yet — flagged as the next natural calibration target
(likely `CATEGORY_GROUP_FLOOR` or a slug-specific threshold rather than
another global `CATEGORY_CONFIDENCE_THRESHOLD` move, since the global lever
was already spent getting `acc_other` from 0.215 to 0.545) once more
post-calibration data accumulates.

### Follow-up (2026-09-10): NER subword-fragmentation root cause and fix

Investigated a pattern the user spotted directly in real data: entities like
the bare digit `"3"` tagged `ORG`, apparently meant to represent 3M. This
turned out to be a real, previously-unidentified contributor to NER's
documented weakness (`hallucination_rate` 0.338, precision 0.66 vs recall
0.84 — Baseline section above) — not the whole gap, but a concrete, fixable
piece of it.

**Root cause**: a training/inference asymmetry in subword handling.
`train_ner.py`'s `make_tokenize_fn` labels only the first WordPiece subword
of each source word and masks every continuation subword to
`IGNORED_LABEL_ID` in the loss — the model gets zero training signal for
continuation-subword predictions. `pipeline.py`'s `merge_bio_predictions`
never accounted for this at inference: it merged spans by every token's own
raw BIO argmax, continuation subwords included. For `"3M"` tokenized as
`["3", "##M"]`, the model often predicted `B-ORG` on `"3"` but something
other than `I-ORG` on the untrained-for `"##M"` position, closing the span
after just `"3"` — with no length/sanity filter anywhere in the write path
to catch the resulting bogus `{entity_type: "ORG", text: "3"}` row.

**Quantified impact** (real data, `article_entities`, 17.6M rows): 2,982
bare single-digit `ORG` entities, of which **`"3"` alone is 1,859 (62%)** —
**628 of those in MMM (3M)-ticker articles, hitting 185 of 316 MMM articles
(59%)**. The same articles correctly tag whole `"3M"`/`"MM"`/`"M"` as `ORG`
459 times too — the model is genuinely inconsistent, not uniformly broken.
Confirmed via SOURCE `body_text` these are literal mid-word splits of real
"3M" mentions. The same bug class (not digit-specific) is independently
corroborated by real LLM-judge `wrong`-verdicts from past NER eval runs:
bogus `"3"` split off `"3Q24"`, bogus `"1"` split off `"1MDB"`, bogus `"20"`
split off `"20 años"`, bogus `"777"` split off `"Boeing's 777 jet"`. Also
affects `L3Harris` (`"L"`/`"L3"` fragments) but not `F5 Inc.`/`Phillips 66`
— specifically triggered by a digit glued directly to letters at a
WordPiece split boundary.

**Fix**: `merge_bio_predictions` is now word-id aware — only a word's first
subword ever opens, closes, or redirects a span; a continuation subword
(same `word_id`) just extends whatever the owning word's first subword
already decided, matching how the model was actually trained. A secondary,
narrow length-≥2 filter was added at write time in `run_ner_stage` as a
last-resort net for whatever the word-id fix doesn't structurally prevent —
not a substitute for it. `excludes_bare_digit`
(`portfolio_common.db.dialect.SqliteDialect`) remains in place on its two
existing call sites (`fetch_pending_company_summaries`,
`fetch_sector_week_entity_stats`) but was never the general fix — it only
ever caught bare single digits on two downstream read-side aggregate
queries, not the write path, not other fragment shapes (e.g. a lone `"L"`),
and not the main article-detail read.

**Scope**: future pipeline runs only, matching this repo's established
precedent (the category hierarchical-classification change was also
future-runs-only). The existing 17.6M-row `article_entities` table is
unchanged; a bulk re-extraction is a separate, not-yet-built follow-up.

This is a *different* issue from the full-article-vs-lead-cap NER
eval-sampling mismatch flagged earlier in this doc as "suspected but
unverified" — that's still open and unrelated to subword fragmentation.

### Follow-up (2026-09-12): NER full-article-vs-lead-cap mismatch confirmed; no post-fix data exists yet

Two findings while starting work on the NER validation task (`PLAN.md` Work
item 3 / `TASKS.md` T-020–T-023).

**The suspected sampling mismatch is real, not just suspected.** Ran
`sample_for_stage(conn, "ner", size=1000, seed=1)` against the real
RESULTS/SOURCE stores (no LLM calls — this only exercises the sampling
code) and compared the judge's `_MAX_BODY_CHARS` (6000) cap against the
*untruncated* `source.articles.body_text` for the same 1000 articles:

- **214/1000 (21.4%)** of sampled articles exceed 6000 chars (median body
  length 3549.5, mean 4800, **max 156,053** — over 4x the longest article
  `_SENTIMENT_MAX_BODY_CHARS`'s 100,000-char safety ceiling was sized
  against as of the 2026-09-08 sentiment fix; NER's population apparently
  has a much longer tail than sentiment's did).
- Of the **67,049** entities `run_ner_stage` predicted across those 1000
  articles, **11,262 (16.8%)** have a `start_char >= 6000` — i.e. the judge
  is structurally incapable of confirming or denying them, because it's
  never shown the text they came from. **209/1000 (20.9%)** of sampled
  articles have at least one such entity.
- This is a more direct problem than sentiment's version of the same bug:
  sentiment's judge saw a truncated body and produced a *worse-calibrated
  score*; NER's `parse_fail_rate`-adjacent error-only judge contract means
  a predicted entity the judge can't see the source text for likely reads
  as an ungrounded/`wrong` verdict (inflating false "hallucination" calls)
  or is silently skipped, while any *real* entity past char 6000 can never
  be named as `missed` (deflating the false-negative count) — both push
  the stage's already-weak `hallucination_rate` (0.338, 2026-09-08
  baseline) and precision/recall numbers in directions that don't reflect
  the model's actual behavior on the untruncated article.
- **Fixed same-day**: raised `ner` into `_UNCAPPED_STAGES`
  (`src/news_nlp/eval/sampling.py`) the same way `sentiment` was fixed
  (2026-09-08 follow-up above) — `run_ner_stage` chunks and predicts over
  the *entire* article, so judging it on a lead-only cap was never a
  faithful comparison, same reasoning as sentiment's fix. Landed **before**
  spending any judge-call budget on a fresh NER eval run (below), so that
  run won't be immediately stale the way the pre-text-scope-fix sentiment
  baseline was. Unlike sentiment, `ner`'s uncapped safety ceiling
  (`_SENTIMENT_MAX_BODY_CHARS`, shared across both stages, not renamed) does
  engage against real data: the 156,053-char article found above exceeds
  it, so that one article's tail gets truncated rather than the request
  failing outright — the intended degradation, not a gap.

**No post-fix NER data exists to evaluate yet, and none can be produced by
just re-running the pipeline.** Checked the real stores directly:
`article_entities`'s newest row (by `id`) is dated **2026-08-19**, and
`article_sentiment`'s newest is **2026-08-17** — both stages haven't run
since, well before the 2026-09-11 `merge_bio_predictions` fix. Checked
whether a normal pipeline re-run would pick up new work: SOURCE
(`urls.db`) articles' max `id` (485,352) is **identical** to RESULTS'
lean-`articles` max `id` — there is no backlog of un-processed articles.
Every article already has a (pre-fix) NER row, and `run_ner_stage` only
processes rows *missing* from `article_entities` (FR-006, idempotent by
design) — so simply invoking `uv run cli/news_nlp_cli.py` again right now
would find zero pending NER work and write nothing new.

This means **T-020 is more tightly blocked on T-022 than originally
scoped** — not "a fresh eval run, then separately decide on backfill
scope," but "no fresh eval run is possible at all until *some* reprocessing
happens," even a small one. `delete_entities_for_article` (`news_nlp.
corrections`) already exists and makes a single article eligible for
reprocessing again (same mechanism `delete_category` uses, FR-009) — a
**targeted reprocessing of a modest sample** (e.g. 1,000-2,000 articles,
not the full 17.6M-row table) is technically available today without
committing to the full-corpus backfill question T-022 was originally
scoped around. Whether to do that, at what size, and whether category's
precedent (existing rows read back unmigrated until reprocessed, SPEC.md
§13 item 6) is an acceptable model for NER too, is still a decision for
the repo owner — not made unilaterally here.

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
