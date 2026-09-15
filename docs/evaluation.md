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

### Follow-up (2026-09-12): T-025 executed — `article_entities` versioned, 20,000-article random resample

The repo owner chose T-025 (targeted resample) over T-022 (full-corpus
backfill), with one refinement: **version the table by renaming it**
rather than reprocessing specific rows in place via
`delete_entities_for_article`. `scripts/resample_ner_2026_09_12.py`
(one-shot, kept in git history per this repo's precedent for this kind of
maintenance script — see `docs/migration-2026-09-01.md`):

1. Renamed the pre-fix `article_entities` (17,666,722 rows / 458,867
   distinct articles) to **`article_entities_v1`** — nothing deleted, a
   durable, queryable snapshot of the pre-fix model's output. Its index
   was explicitly re-homed (`idx_article_entities_v1_article_id`) rather
   than left in place: a real SQLite gotcha found and verified (with a
   throwaway in-memory repro) before touching production data —
   `ALTER TABLE ... RENAME TO` does **not** rename a table's indexes, so
   the original `idx_article_entities_article_id` name would have stayed
   bound to the renamed table, making `init_schema`'s
   `CREATE INDEX IF NOT EXISTS` of that same name against the fresh table
   silently no-op and leave it unindexed.
2. Recreated a fresh, empty `article_entities` via `news_nlp.init_schema`
   (verified indexed, verified empty, before proceeding).
3. Ran `run_ner_stage` over a `random.Random(seed=1)`-seeded, reproducible
   random sample of 20,000 pending articles (`news_nlp.queries.
   fetch_pending_articles`'s new `sample_seed` parameter — every article
   was "pending" against the just-emptied table) under the current, fixed
   code (both the 2026-09-10 word-boundary fix and the 2026-09-12
   uncapped-eval-sampling fix above).

Dry-run validated end to end against a scratch copy of the real store
(50-article sample) before running for real.

**Result**: 714,334 entity rows across **19,988 of the 20,000** sampled
articles (12 articles genuinely predicted zero entities — `write_entities`
writes no row at all for those, a pre-existing, unrelated behavior). Ran in
~15 minutes on the project's GPU (RTX 4050 Laptop, matching `SPEC.md`
NR-001's 6GB VRAM budget). Per-type split: `ORG` 409,575 (57.3%), `PER`
159,912 (22.4%), `LOC` 144,847 (20.3%) — close to the pre-fix corpus's
documented ORG 55%/PER 25%/LOC 20% split (Sampling section below), so the
fix didn't distort the overall type balance.

**T-020 is now unblocked**: 19,988 post-fix articles exist to draw an eval
sample from, where none existed before this follow-up. Running that eval
(`--stage ner`) is the natural next step but was **not** done as part of
this follow-up — it spends real judge-call budget and is a distinct
action from the reprocessing done here.

### Follow-up (2026-09-12): T-020 executed — post-fix NER eval, both fixes confirmed

Ran `uv run cli/news_nlp_eval.py --stage ner --seed 1 --sample-size 8000`
(mlflow run `329f9222`, `news_nlp_eval/ner`, `code_version=18fc30d` — both
the word-boundary merge fix and the uncapped-sampling fix are in). Sample
size is 8x the 2026-09-08 baseline's 1000 and deliberately large relative
to the population: it's a 40% draw of the 19,988-article T-025 resample
pool, not a ~0.2% draw of the full ~459K-article corpus like the baseline
was — that resample is the only place post-fix rows exist yet (see the
T-025 follow-up above). `n_judged=8000`, usable `n=7965` after parse
failures.

| metric | 2026-09-08 baseline (pre-fix, `3b41cbec`) | 2026-09-12 (post-fix, `329f9222`) | Δ |
|---|---|---|---|
| `micro_f1` | 0.7418 | **0.8578** | +0.116 (+15.6%) |
| `micro_precision` | 0.6623 | 0.8403 | +0.178 (+26.9%) |
| `micro_recall` | 0.8429 | 0.8761 | +0.033 (+3.9%) |
| `hallucination_rate` | 0.3377 | **0.1597** | −0.178 (−52.7%) |
| `miss_rate` | 0.1571 | 0.1239 | −0.033 (−21.1%) |
| `macro_f1` | 0.7429 | 0.8729 | +0.130 |
| `f1_ORG` | 0.7454 | 0.8117 | +0.066 |
| `f1_LOC` | 0.8236 | 0.8807 | +0.057 |
| `f1_PER` | 0.6597 | **0.9262** | +0.267 (+40.4%) |
| `parse_fail_rate` | 4.7% | 0.44% | −91% (relative) |
| `mean_entities_per_article` | 43.25 | 32.45 | −25% |

**Reads as the fix working, not noise**: the gain is concentrated in
precision (+26.9%) and `hallucination_rate` (halved), while recall barely
moves (+3.9%) — exactly the shape expected from a fix that removes bogus
spans rather than one that would help the model find new true entities.
`f1_PER`'s outsized jump (+40.4% relative, the largest of the three types)
matches the root cause: person names carry more multi-subword WordPiece
splits than org/location names, so they were hit hardest by the pre-fix
training/inference asymmetry. `parse_fail_rate` collapsing 4.7%→0.44%
independently corroborates the same-day uncapped-sampling fix (T-024) —
the judge is no longer choking on entities past the old 6000-char cap.
`mean_entities_per_article` dropping 25% is the same precision effect
showing up as a raw count (fewer bogus single-token spans written at all).

**Caveat — not a full-corpus result.** This run only speaks to the 19,988
articles reprocessed under the fixed code (T-025). The remaining ~439,000
pre-fix articles (`article_entities_v1`) are untouched and would still
score at the old ~0.74 baseline until reprocessed — T-022's full-corpus
backfill question is unaffected by this result either way. Also note: the
baseline run had no `--seed`, so strata composition isn't reproducibly
comparable row-for-row against this seeded run — the metric-level
before/after above is still valid, just not a row-identical replay.

This clears the last open item in `PLAN.md` Work item 3 / `TASKS.md`
T-020; `SPEC.md` §9's NER row is updated below.

### Follow-up (2026-09-13): sentiment — three designs measured; fine-tuning wins

`PLAN.md` Work item 4 (`SPEC.md` §13 item 1). Three candidate fixes for
FinBERT's lack of per-company/net-signal reasoning and (as this follow-up
found) domain/vocabulary staleness were prototyped and real-data validated
against the production DB and the real LLM judge this week, each on its
own branch so it stands on its own evidence:

1. **Entity-scoped chunk-weighting** (`feat/sentiment-entity-scoped`,
   PR #42): score each ~510-token chunk, weight chunks naming the
   article's own `company`/`ticker` over everything else.
   `recall_negative` 0.856 vs. a 0.783 pre-change pilot — a real
   improvement on this project's stated priority metric, but
   `precision_negative` stayed weak (0.376), essentially unchanged from
   every prior sentiment design measured.
2. **Title-only scoring** (`feat/sentiment-title-only`, PR #43): score
   just the headline, no aggregation. `recall_negative` 0.533 — did not
   beat the chunk-level design; real disagreement transcripts showed
   headlines that read factually neutral over a strongly directional
   body, and idiom/second-order reasoning single short spans can't
   support.
3. **Fine-tuning FinBERT itself** (this branch,
   `feat/finbert-financial-news-finetune`): rather than continue tuning
   the *aggregation* around a model whose training data predates most of
   the corpus it now scores, fine-tune the model. `ProsusAI/finbert` was
   trained on [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
   — ~4,840 sentences of **2014 news about Nordic (OMX Helsinki) listed
   companies**. Real disagreement transcripts from (1) and (2) above
   already showed the cost directly: the base model missing "crushed" as
   a positive idiom ("Amazon and Alphabet crushed earnings"), and no
   grounding for instruments/vocabulary that barely existed in 2014
   (cryptocurrency).

**Training data**: 5,000 sentences, LLM-labeled (`deepseek-chat`,
temperature 0, investor/price-impact framing — the same framing Financial
PhraseBank's own annotators used), drawn from real article `body_text`
already sampled across every prior `--stage sentiment` eval run (a pool
the stratified sampling design already skewed toward covering all three
classes). 100% labeling success. Class split: 938 positive (18.8%), 893
negative (17.9%), 3,169 neutral (63.4%) — real financial news skews
neutral, consistent with every other finding in this document.
80/10/10 train/validation/test, stratified per label.
`scripts/label_sentiment_sentences_2026_09_13.py` has the full
methodology; **known limitation, not solved here: these are LLM-generated
labels (silver-standard), not human-annotated ground truth** — the same
caveat this document applies to every judge-derived number, now also
applying to training data, not just evaluation.

**Training**: continued fine-tuning from the `ProsusAI/finbert` checkpoint
(not vanilla BERT — a domain refresh, not a fresh retrain), 4 epochs,
lr 2e-5, `src/train_sentiment.py`. Held-out sentence-level test set
(n=498): accuracy 0.813, macro F1 0.774 (F1 positive 0.798 / negative
0.667 / neutral 0.858).

**The real test — substituted into the best aggregation scheme found so
far** (entity-scoped chunk-weighting from PR #42), same 2,000-article
pool, same LLM judge:

| metric | pilot (pre-change) | chunk-level, base FinBERT (PR #42) | title-only (PR #43) | **chunk-level, fine-tuned FinBERT** |
|---|---|---|---|---|
| `recall_negative` | 0.783 | **0.856** | 0.533 | 0.812 |
| **`precision_negative`** | 0.359 | 0.376 | 0.484 | **0.505** |
| `f1_negative` | 0.493 | 0.523 | 0.507 | **0.623** |
| `macro_f1_vs_judge` | 0.546 | 0.625 | 0.620 | **0.737** |
| `agreement_rate` | 0.483 | 0.585 | 0.633 | **0.697** |
| `agreement_rate_representative` | 0.555 | 0.700 | 0.763 | **0.852** |
| `recall_positive` | 0.462 | 0.520 | 0.507 | **0.790** |
| `precision_neutral` | 0.733 | 0.871 | 0.803 | **0.935** |
| `mean_severity` | 0.570 | 0.481 | 0.410 | **0.350** |
| `parse_fail_rate` | — | 0.0 | 0.0 | 0.0 |

**Fine-tuning is the strongest result of this entire investigation, and
not narrowly** — `recall_negative` dips slightly from the base model's
0.856 (still comfortably above the 0.783 pre-change pilot and this
project's documented healthy range), but every other headline metric
improves substantially, most notably `precision_negative` (+13 points),
the single most persistent weak point across every design measured for
this stage since the 2026-09-08 baseline. This is a broad-based
improvement, not one metric traded against a loss everywhere else — the
opposite of the sentence-level-vs-chunk-level trade-off found earlier in
this investigation.

**Honest limitation, checked directly, not glossed over**: a quick manual
spot-check after publishing the model still shows it mislabeling "Amazon
and Alphabet crushed earnings" as negative — the exact idiom gap this
fine-tune targeted. The aggregate metrics improved substantially; this
specific case did not resolve. 5,000 sentences, one training pass, is not
guaranteed to have covered every gap the base model had, and the
LLM-labeled training data may itself be sparse or inconsistent on rare
idioms like this one.

**Methodological caveat, disclosed not hidden**: the training labels
(DeepSeek) and the LLM judge scoring this comparison are not
independently sourced — both ultimately trace to the same class of tool.
This doesn't invalidate the pipeline-level result (the judge scores whole
articles holistically; the training data is isolated sentences — a real
task/granularity difference, not the same function scoring itself twice),
but it means "improved agreement with this judge" is not the same claim
as "objectively more accurate," and should be read with that in mind.

Published: [`gamug/FinBERT-financial-news`](https://huggingface.co/gamug/FinBERT-financial-news)
(full model card with training data, procedure, and both evaluation
results). `SPEC.md` §13 item 1 and §9's sentiment row are updated below,
noting all three candidates and this recommendation; which (if any) is
actually wired into `SENTIMENT_MODEL`/`run_sentiment_stage` remains the
repo owner's decision, not made unilaterally by any of the three
branches.

### Follow-up (2026-09-13): the "crushed earnings" idiom gap — diagnosed, mined, fixed, measured

The honest limitation disclosed just above ("crushed earnings" still
mislabeled negative post-fine-tune) wasn't left as an accepted gap.
Diagnosis first, before any fix: mining the eval-run article pool (11,322
articles) for the exact idiom family (`crushed/smashed/trounced/
clobbered/routed/walloped/demolished/hammered` + earnings/estimate/
guidance/consensus context) found only **75 hits** — the original
5,000-sentence random draw happened to include almost none of them by
chance, a coverage gap, not a labeling error (the few "beat/topped/
surpassed" idioms that *did* make it into training were already labeled
correctly).

That mining pass also ruled out the cheap fix: the same verb family
flips polarity depending on *what* is being crushed —
`"Nvidia stock got crushed"` (negative, the stock/company is the object)
vs. `"Meta crushed its earnings estimates"` (positive, an estimate/target
is the object). A lexicon/regex override would get the negative case
wrong, so `scripts/mine_idiom_sentences_2026_09_13.py` widened the search
to the **full ~480k-article source corpus** (not just the 11k-article eval
pool) and LLM-labeled 900 more sentences with an explicit instruction
covering both directions. Class split: 487 negative, 316 positive, 97
neutral — confirming the negative ("got crushed") sense is actually more
common in this corpus than the positive ("crushed estimates") sense, the
opposite of what a naive "crush = positive idiom" rule would assume. 100
of the 900 were held out entirely from training as an **idiom probe** —
never trained on, reserved purely to measure the fix directly rather than
infer it from aggregate metrics moving. A manual spot-check of 20 probe
labels (including a negation case, `"Instead of CSX getting crushed, the
stock actually went higher"` → correctly positive) checked out.

The remaining 800 were merged into the 5,000-sentence base draw (5,900
total) and `src/train_sentiment.py` retrained from the base
`ProsusAI/finbert` checkpoint from scratch (not a second fine-tuning pass
on top of v1, to avoid double-fine-tuning drift).

**Idiom probe, direct before/after** (v1 = the previously-published model,
scored on this same held-out probe; v2 = this update):

| metric | v1 (pre-fix) | **v2 (this update)** |
|---|---|---|
| Accuracy | 0.750 | **0.870** |
| Macro F1 | 0.664 | **0.759** |
| Recall — positive | 0.710 | **0.903** |
| Recall — negative | 0.797 | **0.932** |

The exact original case now scores correctly: `"Amazon and Alphabet
crushed earnings."` → **positive** (0.935 confidence); the negative sense
is still caught correctly too: `"The stock got crushed after the
disappointing guidance."` → **negative** (0.994 confidence).

**Downstream pipeline re-check — did fixing this regress anything else?**
Re-ran the exact same integration test (chunk-level entity-scoped
aggregation, identical 2,000-article pool, seed=1, same LLM judge,
eval_run 30, mlflow `8e4aa2d3`):

| metric | fine-tuned v1 | **fine-tuned v2, idiom-fixed** |
|---|---|---|
| `recall_negative` | 0.812 | 0.808 |
| `precision_negative` | 0.505 | **0.513** |
| `f1_negative` | 0.623 | **0.628** |
| `macro_f1_vs_judge` | **0.737** | 0.731 |
| `agreement_rate` | 0.697 | **0.701** |
| `agreement_rate_representative` | 0.852 | **0.866** |
| `recall_positive` | 0.790 | **0.801** |
| `precision_neutral` | 0.935 | **0.936** |
| `mean_severity` | 0.350 | **0.341** |

Every v1→v2 delta on the full 2,000-article real-traffic sample is within
±0.01 — noise, not a trade-off. The targeted fix held on the specific
failure class without costing anything measurable on the broader
distribution. Model card and Hub weights updated in place at
[`gamug/FinBERT-financial-news`](https://huggingface.co/gamug/FinBERT-financial-news)
(same repo, new commit — not a new model name, since this is a fix to the
same model, not a new candidate).

**Still open, disclosed not hidden**: `neutral` performance on this
specific idiom probe stayed weak (f1 0.39→0.47) — expected, since only
97/900 mined sentences were neutral and this probe isn't representative
of the pipeline's overall neutral-heavy traffic (see the downstream
table's `precision_neutral` for neutral performance on real traffic,
which is strong). This idiom family was one specific, manually-discovered
gap; the same "silver-standard LLM labels, not exhaustively verified"
caveat from the base fine-tune still applies, and other undiscovered
vocabulary gaps of this shape likely still exist.

### Follow-up (2026-09-13): diagnosing the remaining precision gap, testing title-only against it, and selecting a production candidate

The idiom fix above closed one gap; `precision_negative`/`precision_positive`
still sat around 0.50-0.65 on the fine-tuned chunk-level design — a real
concern (near-random-feeling on a 3-class problem), investigated directly
against eval_run 30's disagreement data rather than assumed:

**Confusion-matrix diagnosis**: of 1,072 directional predictions, 51.8%
were correct, only 7.6% were genuine positive↔negative flips, and
**40.6% were directional calls on articles the judge scored neutral** —
almost the entire precision problem is "calls something directional that
isn't," not "calls negative when it's actually positive." Reading the
judge's own rationale text for all 435 such cases and keyword-classifying
them: 50.6% are multi-company/market-wrap roundups with no single-company
focus, 38.4% are mixed-signal pieces where competing facts should net to
neutral, and only ~11% combined are preview/speculative or immaterial-news
patterns a sentence-level relabeling could plausibly address.

**Three fix attempts, all tested on real data, all rejected before
building anything into the pipeline** (reported here because each was a
real, falsifiable hypothesis, not because negative results are
interesting for their own sake):
- *Confidence/margin threshold*: ruled out — only 7% of predictions have
  a thin top1-vs-top2 margin; the model is confidently wrong on most
  errors (median margin 0.90), so a threshold gate would barely fire.
- *Subject-coverage gate* (down-weight when the subject company's share
  of chunk-weight is thin): ruled out — false alarms and correct
  predictions have nearly identical subject-mass-share distributions
  (0.623 vs 0.631 mean); this corpus's articles are just entity-dense in
  general, so coverage isn't discriminative.
- *Zero-shot "realized event vs. speculative/roundup" materiality gate*
  (reusing the category stage's `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`):
  ruled out — top-label agreement with "realized result" was 51.7% for
  false alarms vs. 53.3% for correct predictions, and every threshold in
  a sweep from 0.15 to 0.4 caught false alarms at almost exactly the same
  rate it wrongly suppressed correct ones (e.g. at 0.2: 25.0% caught vs.
  19.2% wrongly suppressed). This framing doesn't separate the two groups
  at all — a genuine negative result, not a calibration miss.

**Why none of these worked, structurally**: the dominant failure
(roundup / mixed-signal, ~89% of the 435 false-alarm cases) isn't a
sentence-classification error — a chunk reading "XYZ Corp shares fell 5%"
inside a multi-company roundup is being read correctly *as a sentence*.
The judge's neutral call reflects a document-structure fact ("this piece
isn't dedicated to one company") or a cross-sentence composition fact
("these two claims should net against each other") that no per-chunk
label, confidence threshold, or off-the-shelf zero-shot NLI pass over
title+lead text captures. This is the same "net-signal reasoning" gap
flagged at the very start of this investigation — still present after
fixing vocabulary (fine-tuning) and entity scope (PR #42's weighting).

**Title-only + the fine-tuned model, tested as a genuine alternative**:
since aggregation across multiple, possibly-irrelevant chunks is the
mechanism producing false alarms, title-only scoring (PR #43's own
design, not yet merged) sidesteps aggregation entirely — one short,
single-topic forward pass, no chunk weighting. Re-tested with the
fine-tuned model (not just base FinBERT, which is what PR #43 originally
measured) on the same 2,000-article pool:

| | chunk-level, base FinBERT (PR #42, eval_run 27) | title-only, base FinBERT (PR #43, eval_run 28) | **chunk-level, fine-tuned — selected** (eval_run 30) | title-only, fine-tuned (eval_run 31) |
|---|---|---|---|---|
| precision / recall / f1 — **positive** | 0.687 / 0.520 / 0.592 | 0.586 / 0.507 / 0.544 | 0.647 / **0.801** / **0.716** | **0.670** / 0.528 / 0.591 |
| precision / recall / f1 — **negative** | 0.376 / **0.856** / 0.523 | 0.484 / 0.533 / 0.507 | 0.513 / 0.808 / **0.628** | **0.619** / 0.574 / 0.595 |
| precision / recall / f1 — **neutral** | 0.871 / 0.674 / 0.760 | 0.803 / 0.815 / 0.809 | **0.936** / 0.777 / **0.849** | 0.842 / **0.897** / 0.869 |
| `macro_f1_vs_judge` | 0.625 | 0.620 | **0.731** | 0.685 |
| `agreement_rate` | 0.585 | 0.633 | 0.701 | **0.746** |
| `mean_severity` (lower better) | 0.481 | 0.410 | 0.341 | **0.287** |

The fine-tuned model transfers to title-only much better than base
FinBERT did (recall_negative 0.533→0.574, precision_negative
0.484→0.619, agreement_rate 0.633→0.746) — but it's a real trade-off
against chunk-level, not a win: title-only wins on precision (both
directional classes) and on every aggregate calibration metric
(agreement, severity), while chunk-level wins on recall for *every*
class — positive 0.801 vs 0.528, negative 0.808 vs 0.574, a roughly
15-27-point gap each way. There is no dominant design here; this is a
genuine point on a precision/recall frontier, and picking one is a
values decision this document states explicitly rather than resolves by
default.

**Decision (2026-09-13): chunk-level, fine-tuned FinBERT is selected as
the production candidate.** Rationale — this pipeline is deliberately
**pessimistic**: for a monitoring signal that feeds a portfolio-analysis
SEMANTIC score and a knowledge graph, a missed real story (a false
negative — good or bad news silently filed as neutral) is a blind spot
downstream consumers have no way to recover from, while a false alarm
(lower precision) is a story a human reviewer or a downstream aggregation
step can still discount or average away. Chunk-level's **0.801 recall on
positive and 0.808 on negative** — both classes, not just one — mean it
surfaces the large majority of real upside and downside stories.
Title-only's 0.528 / 0.574 recall on those same two classes means it
*misses roughly four in ten* of exactly the stories this system exists to
catch. That is the same "recall over precision" argument this document
already made for the negative class alone (`docs/evaluation.md`'s
2026-09-08 "Why recall, not F1" section) — generalized here to both
directional classes as the deciding criterion, precisely because both
recalls are strong under the chunk-level design and neither is under
title-only. The remaining precision cost (0.647 positive / 0.513
negative) is a real, diagnosed, and disclosed limitation — not an
unexamined one — and is judged the acceptable side of this trade-off for
a "surface it, let downstream discount false alarms" monitoring signal,
rather than a "stay silent by default" one.

### Follow-up (2026-09-13, same day): the selected design merged into the pipeline

`src/pipeline.py`'s `SENTIMENT_MODEL` now points at
`gamug/FinBERT-financial-news`, with `run_sentiment_stage` doing the
chunk-level entity-scoped weighting (`_text_mentions_subject`/
`_sentiment_chunk_weights`) cherry-picked from PR #42's final revision —
`db.fetch_pending_sentiment_articles` (also from PR #42) supplies the
`(company, ticker)` pair each chunk's weight is computed against. This is
no longer just a recorded recommendation: it's the code every pipeline
run now actually executes.

Verified two ways before treating this as done: the full hermetic test
suite (225 tests — up from 214, PR #42's own `test_sentiment_pipeline.py`
and `test_schema.py` additions came along with the cherry-pick), ruff,
and mypy all pass; and, separately, a live smoke test against real
production data — `run_sentiment_stage(conn, limit=3, sample_seed=999)`
against the actual `nlp.db`/`urls.db` — loaded the fine-tuned model on
CUDA and correctly scored 3 previously-unscored articles, including the
exact "A" (Agilent)/"ON" (ON Semiconductor) ticker-ambiguous names this
investigation's earlier ticker-collision bug was about. `SPEC.md` §13
item 1, §9, and FR-001 are updated to reflect the merge; the repository
artifact's "Gaps"/"Plan" sections are updated too.

### Follow-up (2026-09-14): sentiment training data rebalanced — a real trade, not a strict win

`PLAN.md` Work item 9 / `SPEC.md` §13 item 14. The 5,800-sentence training
pool behind `gamug/FinBERT-financial-news` (5,000 base draw + 800 merged
idiom-augment sentences) was 3,256 neutral (56.1%) / 1,321 negative
(22.8%) / 1,223 positive (21.1%) — never a deliberate target, a byproduct
of drawing sentences from the eval harness's confidence-stratified
sampling pool (stratified on prediction confidence, not label ratio) on
top of real financial news skewing neutral/factual. Nothing in the
original training procedure corrected for it: `train_sentiment.py` picked
the best checkpoint by macro F1 (equal per-class weight, but only at
*evaluation* time) and stratified the train/validation/test split to
match the source distribution, not rebalance it.

**Fix**: `scripts/rebalance_sentiment_data_2026_09_14.py` downsamples
`neutral` to 1,321 — the size of the larger minority class (`negative`) —
keeping every `positive`/`negative` sentence untouched. Which 1,321 of the
original 3,256 `neutral` sentences survive is not a random cut: each is
ranked by cosine similarity to the `neutral` class's own TF-IDF centroid
(scikit-learn, already a transitive dependency), and the most
representative (closest to centroid) are kept, the most atypical/outlier
ones dropped. Result: 1,321 / 1,321 / 1,223 (34.2% / 34.2% / 31.6%) — a
genuine three-way balance. Published as v2 of
[`gamug/FinBERT-financial-news-data`](https://huggingface.co/datasets/gamug/FinBERT-financial-news-data)
(`scripts/publish_finbert_financial_news_dataset_rebalanced_2026_09_14.py`);
`idiom_probe` (100 rows) is untouched in both versions — its role is
measuring against real, unfiltered idiom-family traffic, not a
class-balance concern.

`train_sentiment.py` was extended (not replaced) to prefer this rebalanced
pool when present (`BALANCED_DATA_PATH`), same procedure/hyperparameters
as before (`ProsusAI/finbert` base, lr 2e-5, 4 epochs, seed 42) — a
data-quality fix, not an architecture or hyperparameter change. Retrained
and measured against the currently-published model (referred to below as
v2; the rebalanced retrain as v3):

**Held-out sentence-level test set**

| metric | v2 (published, unbalanced data, n=579) | **v3 (rebalanced data, n=386)** |
|---|---|---|
| Accuracy | 0.798 | 0.777 |
| Macro F1 | 0.779 | 0.774 |
| Precision — positive | — | 0.795 |
| Recall — positive | — | 0.762 |
| F1 — positive | 0.775 | **0.778** |
| Precision — negative | — | 0.756 |
| Recall — negative | — | **0.917** |
| F1 — negative | 0.726 | **0.829** |
| Precision — neutral | — | 0.789 |
| Recall — neutral | — | 0.652 |
| F1 — neutral | **0.838** | 0.714 |

Negative F1 improves substantially (0.726→0.829) — the class that wasn't
touched by rebalancing, but benefits from the model no longer being
pulled toward the now-shrunk neutral majority. Neutral F1 drops
(0.838→0.714), an expected, direct cost of training on 1,935 fewer
neutral examples, not a surprise.

**Idiom probe (n=100, held out of training, unchanged between v2/v3) —
where the real cost shows up**

| metric | v2 (published, pre-rebalance) | **v3 (rebalanced)** |
|---|---|---|
| Accuracy | 0.870 | 0.830 |
| Macro F1 | 0.759 | **0.583** |
| F1 — positive | 0.889 | 0.848 |
| F1 — negative | 0.917 | 0.902 |
| F1 — neutral | 0.47 | **0.0** |

**Neutral F1 on this probe collapses to 0.0 (precision and recall both
0.0) in v3** — the model made zero correct `neutral` predictions on this
specific slice. This probe is only 10% neutral by design (10/100 rows —
it targets the crushed/smashed/hammered idiom family, which skews
negative/positive, not neutral), so it's a small-n reading, not a broad
claim about v3's neutral performance generally — but it's a real,
measured, disclosed regression, consistent with training on 40% fewer
neutral examples overall, not glossed over.

**Not yet measured**: the downstream, production-pipeline evaluation
(entity-scoped, chunk-level aggregation against real article traffic,
LLM-judge) that validated v2 — that needs a full `--stage sentiment` eval
run against live production data, a separate, larger step from this
retrain.

**Disposition**: this is a trade, not a strict improvement — v3 fixes the
disclosed class imbalance and improves negative F1 substantially, at a
real cost to neutral performance most visible on the idiom probe.
`src/pipeline.py`'s `MODEL_REVISIONS` still pins v2's commit SHA; v3 is
published as an available checkpoint on the Hub, not silently adopted
into the production pipeline — adopting it is a separate decision, to be
made with the downstream-pipeline numbers in hand, not before.

### Follow-up (2026-09-15): a second rebalancing approach — class-weighted loss, no data discarded

Same problem as the follow-up above (56.1%/22.8%/21.1% training-pool
imbalance), a different fix: instead of downsampling `neutral` (discarding
1,935 sentences), keep the full original 5,800-sentence pool and weight
each class's contribution to the loss inversely to its frequency —
`compute_class_weights` in `train_sentiment.py`, computed from the
*train* split's own label counts (4,642 rows): `neutral` 0.594,
`negative` 1.464, `positive` 1.581. `WeightedLossTrainer` (a `Trainer`
subclass overriding `compute_loss` with a weighted `CrossEntropyLoss`)
applies them; `--weighted` on `train_sentiment.py` selects this path,
writing to separate output paths so it doesn't overwrite the downsampled
retrain (v3) above.

Because this trains on the full original pool, it evaluates on the exact
same test set (n=579) and idiom probe (n=100) as the currently-published
model (v2) — a cleaner, more directly comparable reading than v3's
smaller (n=386) rebalanced-pool test set.

**Held-out sentence-level test set (n=579, same set as v2)**

| metric | v2 (published) | v3 (downsampled, n=386 — not directly comparable) | **v4 (class-weighted, n=579)** |
|---|---|---|---|
| Accuracy | 0.798 | 0.777 | **0.796** |
| Macro F1 | 0.779 | 0.774 | **0.778** |
| F1 — positive | 0.775 | 0.778 | 0.770 |
| F1 — negative | 0.726 | **0.829** | 0.729 |
| F1 — neutral | **0.838** | 0.714 | 0.834 |

Unlike v3, v4 doesn't meaningfully move any class — every number sits
within ~0.01 of v2's. Negative F1 ticks up marginally (0.726→0.729), not
the substantial jump v3 got (→0.829), but neutral doesn't pay for it
(0.838→0.834, essentially flat) the way it did in v3 (→0.714).

**Idiom probe (n=100, held out of training, same 100 rows in all three)**

| metric | v2 (published) | v3 (downsampled) | **v4 (class-weighted)** |
|---|---|---|---|
| Accuracy | 0.870 | 0.830 | **0.850** |
| Macro F1 | 0.759 | 0.583 | **0.745** |
| F1 — positive | 0.889 | 0.848 | 0.875 |
| F1 — negative | 0.917 | 0.902 | 0.891 |
| F1 — neutral | 0.47 | **0.0** | **0.471** |

**This is the number that matters most**: v4's idiom-probe neutral F1
(0.471) lands essentially on top of v2's (0.47) — the catastrophic
collapse to 0.0 that made v3 a real regression simply doesn't happen here.
Class weighting corrects the training signal without ever removing the
1,935 neutral sentences v3 discarded, so the model never loses whatever
it was those sentences taught it about harder, less-typical neutral
cases — visible directly in this probe's neutral precision (0.571) /
recall (0.4), both far above v3's 0.0/0.0.

**Reading both experiments together**: v3 (downsample) is a real trade —
a substantial negative-F1 win purchased with a real, measured neutral
regression. v4 (class-weighted) is closer to a free lunch on these two
eval sets — small, mixed movement in every direction, but nothing broken.
Neither has been measured against the downstream, production-pipeline
LLM-judge evaluation (the number that actually validated v2) — that
remains the open step before adopting either. `src/pipeline.py`'s
`MODEL_REVISIONS` is untouched by this experiment either way.

### Follow-up (2026-09-15, same day): a complete, consistent metric set for all three candidates — accuracy_ovr added, v2's precision/recall backfilled

Both follow-ups above compared v2/v3/v4 with a real gap: v2's own model
card only ever published F1 per class on its test set, never
precision/recall — those cells read "not recorded" rather than a number.
Constitution AI behavior #12 (added this session, per direct request) now
requires every classification-stage evaluation to report the same
complete metric set per class — precision, recall, F1, and one-vs-rest
accuracy (`accuracy_ovr_<class>`, same formula/naming as
`news_nlp.eval.metrics.aggregate_category`'s `accuracy_ovr_<slug>`) — plus
overall accuracy/macro F1, computed the same way for every candidate in a
comparison rather than mixing older, differently-sourced numbers with
freshly-computed ones.

`make_compute_metrics()` (`train_sentiment.py`) gained `accuracy_ovr_<label>`.
`scripts/evaluate_sentiment_candidates_2026_09_15.py` then re-evaluated all
three candidates — v2 loaded fresh from the Hub at its pinned revision (not
re-read from its old model card), v3/v4 from their local saved
checkpoints — each on its own already-established test set, with this same
metric function, eval-only (no retraining). v2's freshly-computed numbers
match its model card's old F1 figures to within ~0.001 (same model, same
test set, confirms nothing drifted) and now also carry real
precision/recall/`accuracy_ovr` it never had before.

**Held-out sentence-level test set**

| | v2 (published, n=579) | v3 (downsampled, n=386) | v4 (class-weighted, n=579) |
|---|---|---|---|
| **Overall accuracy** | 0.798 | 0.777 | 0.796 |
| **Macro F1** | 0.779 | 0.774 | 0.778 |
| Positive — precision | 0.748 | 0.795 | 0.746 |
| Positive — recall | 0.803 | 0.762 | 0.795 |
| Positive — F1 | 0.775 | 0.778 | 0.770 |
| Positive — accuracy_ovr | 0.902 | 0.863 | 0.900 |
| Negative — precision | 0.710 | 0.756 | 0.724 |
| Negative — recall | 0.742 | **0.917** | 0.735 |
| Negative — F1 | 0.726 | **0.829** | 0.729 |
| Negative — accuracy_ovr | 0.872 | 0.870 | 0.876 |
| Neutral — precision | **0.858** | 0.789 | 0.848 |
| Neutral — recall | **0.818** | 0.652 | 0.822 |
| Neutral — F1 | **0.838** | 0.714 | 0.834 |
| Neutral — accuracy_ovr | **0.822** | 0.821 | 0.817 |

**Idiom probe (n=100, held out of training, same 100 rows for all three)**

| | v2 (published) | v3 (downsampled) | v4 (class-weighted) |
|---|---|---|---|
| **Overall accuracy** | **0.870** | 0.830 | 0.850 |
| **Macro F1** | **0.759** | 0.583 | 0.745 |
| Positive — precision | 0.875 | 0.800 | 0.848 |
| Positive — recall | 0.903 | 0.903 | 0.903 |
| Positive — F1 | **0.889** | 0.848 | 0.875 |
| Positive — accuracy_ovr | 0.93 | 0.90 | 0.92 |
| Negative — precision | 0.902 | 0.873 | 0.883 |
| Negative — recall | **0.932** | **0.932** | 0.898 |
| Negative — F1 | **0.917** | 0.902 | 0.891 |
| Negative — accuracy_ovr | 0.90 | 0.88 | 0.87 |
| Neutral — precision | 0.571 | 0.0 | 0.571 |
| Neutral — recall | 0.4 | 0.0 | 0.4 |
| Neutral — F1 | 0.471 | **0.0** | 0.471 |
| Neutral — accuracy_ovr | 0.91 | 0.88 | 0.91 |

`accuracy_ovr` reads flatter and higher than precision/recall/F1 across
the board here, exactly the caveat constitution #12 states — every class
is a small minority within its own binary framing (e.g. "neutral" is only
10% of the idiom probe), so "predict not-this-class" alone already scores
well on this metric. Read it alongside precision/recall, not instead of
them, same as `accuracy_ovr_<slug>`'s existing caveat for category.

No new decision follows from this — same disposition as both follow-ups
above: neither v3 nor v4 is published to the Hub, `MODEL_REVISIONS` is
untouched, and the downstream production-pipeline LLM-judge evaluation
remains the open step before adopting either.

### Follow-up (2026-09-15, same day): v4's downstream production-pipeline eval — the number that actually validated v2, now run for v4 too

The one open step named in both follow-ups above. Real article traffic,
entity-scoped chunk-level aggregation (the actual `run_sentiment_stage`
code path, not sentence-level scoring in isolation), LLM-judge — the same
methodology that validated v2 in the first place
(2026-09-13 four-candidate comparison).

**Mechanics** (kept off the real, shared `nlp_.db`/`nlp_use.db` entirely
by copying it first): `nlp_.db` copied to a scratch `nlp_use.db`, never
touching the real file. `scripts/resample_sentiment_v4_2026_09_15.py`
versioned that copy's `article_sentiment` (preserved as
`article_sentiment_v2_published_snapshot_2026_09_15`, nothing deleted) and
scored a fresh sample with v4 by monkeypatching
`pipeline.SENTIMENT_MODEL`/`MODEL_REVISIONS` to point at v4's local
checkpoint **in-process only** — `src/pipeline.py` on disk was never
edited, so the actually-pinned production model was never at risk.
2,500 real articles scored with v4 in ~107s (chunk-level, CUDA). Then
`cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1`
(the documented sample-size floor) judged that fresh sample —
`eval_run` 35, `mlflow_run_id` `8e78ac51b06e417e97cdb7cd40c03738`, both
inside the scratch copy only. Real time: ~11 minutes for 2,000 judge
calls.

**Caveat on comparability**: this is v4's own fresh eval run against its
own freshly-drawn sample, not a re-judging of the exact same article
instances v2's original 2026-09-13 run used (that run predates this
session and its raw sample isn't reproducible after the fact) — same
sampling design/seed convention, same judge, same aggregation code, but a
different draw. This is the same shape of comparison every earlier
candidate round in this project used (each of the four 2026-09-13
candidates, and NER's/category's own resample rounds, each got its own
eval run against its own sample) — not a new methodological gap introduced
here.

**Complete per-class set, both runs** (constitution AI behavior #12,
extended the same day to cover this downstream/LLM-judge methodology, not
just the offline one — `accuracy_ovr_<class>` added to
`aggregate_sentiment`, same formula/naming as `aggregate_category`'s
`accuracy_ovr_<slug>`). v2's original 2026-09-13 run (`eval_run` 30) had
never had every cell published — precision_positive/f1_positive,
recall_neutral/f1_neutral, and accuracy_ovr for any class were sitting in
its own stored `eval_judgement` rows but were never pulled into
`docs/evaluation.md`'s table. Both runs' complete metrics, including v2's
now-backfilled ones, were recomputed via `aggregate_sentiment` straight
from `eval_run`/`eval_judgement` (`strata_json` for population weights) —
**no new judge calls for either**, since both were already fully judged
and stored; this is a pure re-aggregation with the updated metric
function.

**Overview**

| metric | v2 (published, 2026-09-13 run) | **v4 (class-weighted, this run, n=2000)** |
|---|---|---|
| `agreement_rate` | **0.701** | 0.674 |
| `macro_f1_vs_judge` | **0.731** | 0.724 |
| `mean_severity` (lower is better) | **0.341** | 0.369 |

**Per class**

| | v2 — positive | v2 — negative | v2 — neutral | **v4 — positive** | **v4 — negative** | **v4 — neutral** |
|---|---|---|---|---|---|---|
| Precision | 0.647 | 0.513 | **0.936** | 0.638 | 0.507 | 0.933 |
| Recall | **0.801** | 0.808 | 0.777 | 0.777 | **0.832** | 0.764 |
| F1 | **0.716** | 0.628 | **0.849** | 0.701 | 0.630 | 0.840 |
| `accuracy_ovr` | **0.878** | 0.882 | **0.811** | 0.870 | 0.877 | 0.803 |

**Reading this**: v4 delivers on the one metric this pipeline is actually
built around — negative recall, its stated priority — a real gain
(0.808→0.832), consistent with the sentence-level test-set signal that
class weighting nudges the model away from the old neutral-majority pull.
But it's a trade here too, same as every earlier result in this work item:
`agreement_rate`/`mean_severity` both get worse, and every single per-class
cell in the table above — not just negative recall's mirror image — moves
in v2's favor except that one recall figure. This is a narrower, more
one-directional win than the sentence-level comparison suggested: v4 isn't
"about the same with one clear improvement" downstream, it's "one
real, specific improvement bought at a small cost nearly everywhere else."

**Disposition — unchanged**: v4 is still not published to the Hub, and
`src/pipeline.py`'s `MODEL_REVISIONS` is still untouched, still pinning
v2. With the downstream number now in hand (the one thing missing before),
adopting v4 would mean deliberately trading `agreement_rate`/
`mean_severity` for `recall_negative` — a real decision with a real cost
on both sides, not a default one this evaluation makes on its own. The
scratch copy (`nlp_use.db`) is left as-is, not deleted, in case the exact
judged rows need re-inspecting; the real `nlp_.db`/`nlp.db` were never
opened for writing at any point in this follow-up.

### Follow-up (2026-09-15, same day): a third approach — swap the base checkpoint (`nlpaueb/sec-bert-base`), rejected before a downstream eval

Neither rebalancing approach (v3 downsample, v4 class-weighted) moved
`precision_negative` on the metric that actually matters — the downstream,
production-pipeline number, stuck at 0.505/0.513/0.507 across v1/v2/v4
(see the three follow-ups above). The 2026-09-13 follow-up's
confidence/margin-threshold finding (only 7% of predictions have a thin
top1-vs-top2 margin; median margin 0.90 on errors) already ruled out
"the model is hesitant" as the cause — it's *confidently* wrong, which
argues for a training-signal or base-checkpoint problem, not a
calibration one. Tried swapping the base checkpoint from `ProsusAI/finbert`
to `nlpaueb/sec-bert-base` (already this project's NER base, domain-pretrained
on 260,773 SEC 10-K filings with its own 30k-subword financial vocabulary
— Loukas et al. 2022, arXiv:2203.06482) as a real, falsifiable candidate
fix for vocabulary/subword fragmentation, via `train_sentiment.py --base-model
nlpaueb/sec-bert-base` (new flag, this follow-up). Same procedure/
hyperparameters/data as v2 (original unbalanced pool, 4 epochs, same
splits) for a clean base-model-only comparison — `MODEL_REVISIONS` untouched.

**Complete per-class set — held-out sentence-level test set (n=579, same set as v2/v4)**

| | v2 (published) | v3 (downsampled) | v4 (class-weighted) | **v5 (sec-bert-base)** |
|---|---|---|---|---|
| **Overall accuracy** | 0.798 | 0.777 | 0.796 | 0.765 |
| **Macro F1** | 0.779 | 0.774 | 0.778 | 0.732 |
| Positive — precision | 0.748 | 0.795 | 0.746 | 0.760 |
| Positive — recall | 0.803 | 0.762 | 0.795 | 0.648 |
| Positive — F1 | 0.775 | 0.778 | 0.770 | 0.699 |
| Positive — accuracy_ovr | 0.902 | 0.863 | 0.900 | 0.883 |
| Negative — precision | 0.710 | 0.756 | 0.724 | 0.688 |
| Negative — recall | 0.742 | **0.917** | 0.735 | 0.667 |
| Negative — F1 | 0.726 | **0.829** | 0.729 | 0.677 |
| Negative — accuracy_ovr | 0.872 | 0.870 | 0.876 | 0.855 |
| Neutral — precision | **0.858** | 0.789 | 0.848 | 0.795 |
| Neutral — recall | **0.818** | 0.652 | 0.822 | 0.849 |
| Neutral — F1 | **0.838** | 0.714 | 0.834 | 0.821 |
| Neutral — accuracy_ovr | 0.822 | 0.821 | 0.817 | 0.793 |

**Complete per-class set — idiom probe (n=100, held out of training, same 100 rows for all four)**

| | v2 (published) | v3 (downsampled) | v4 (class-weighted) | **v5 (sec-bert-base)** |
|---|---|---|---|---|
| **Overall accuracy** | **0.870** | 0.830 | 0.850 | 0.820 |
| **Macro F1** | **0.759** | 0.583 | 0.745 | 0.683 |
| Positive — precision | 0.875 | 0.800 | 0.848 | 0.893 |
| Positive — recall | 0.903 | 0.903 | **0.903** | 0.806 |
| Positive — F1 | **0.889** | 0.848 | 0.875 | 0.847 |
| Positive — accuracy_ovr | 0.93 | 0.90 | 0.92 | 0.91 |
| Negative — precision | **0.902** | 0.873 | 0.883 | 0.857 |
| Negative — recall | **0.932** | **0.932** | 0.898 | 0.915 |
| Negative — F1 | **0.917** | 0.902 | 0.891 | 0.885 |
| Negative — accuracy_ovr | 0.90 | 0.88 | 0.87 | 0.86 |
| Neutral — precision | 0.571 | 0.0 | 0.571 | 0.333 |
| Neutral — recall | 0.4 | 0.0 | 0.4 | 0.3 |
| Neutral — F1 | 0.471 | **0.0** | 0.471 | 0.316 |
| Neutral — accuracy_ovr | 0.91 | 0.88 | 0.91 | 0.87 |

**Disposition after the offline gate — rejected pending a downstream check**: unlike v3/v4,
this candidate loses to v2 on nearly every sentence-level/idiom-probe metric, most visibly
neutral F1 on the idiom probe (0.471→0.316). v3 and v4 each earned the expensive downstream
production-pipeline eval (~11 minutes, 2,000 judge calls) by winning cleanly somewhere on this
cheaper gate first; v5 doesn't clear it the same way. Base-checkpoint vocabulary doesn't look
like the fix for `precision_negative`'s stuck-ness at the sentence level — the aggregation-level
multi-company/mixed-signal misattribution identified in the 2026-09-13 follow-up (89% of the
435 false-alarm cases) remains the more likely structural cause there.

**Run anyway, at the user's explicit request** ("the metrics seem more solid than the previous
model except for the neutral ones... I have a good feeling") — the downstream eval below tells
a more nuanced story than the offline gate suggested.

### Follow-up (2026-09-15, same day): v5's downstream production-pipeline eval, run despite the offline rejection above

Same mechanics as v4's downstream eval (`scripts/resample_sentiment_v5_2026_09_15.py`, modeled
on `scripts/resample_sentiment_v4_2026_09_15.py`): a **fresh** scratch copy of the results DB
(`nlp_use_v5.db`, copied from `nlp_.db` — deliberately not v4's own `nlp_use.db`, so this run
can't collide with or be confused with v4's already-scored/versioned state there), production
`article_sentiment` versioned and recreated empty, `pipeline.SENTIMENT_MODEL` monkeypatched
in-process only (`src/pipeline.py` on disk untouched) to v5's local checkpoint. 2,500 real
articles scored, then `cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1`
(the documented floor) — `eval_run` 35 (in `nlp_use_v5.db`'s own independent sequence, not the
same row as v4's `eval_run` 35 in its own scratch copy), `mlflow_run_id`
`ed0e9ff7b4574bf98569bd141ac566a1`.

**Complete per-class set, all three candidates, same downstream methodology**

| | v2 (published) | v4 (class-weighted) | **v5 (sec-bert-base)** |
|---|---|---|---|
| **agreement_rate** | 0.701 | 0.674 | 0.689 |
| **macro_f1_vs_judge** | 0.731 | 0.724 | 0.717 |
| **mean_severity** (lower is better) | 0.341 | 0.369 | 0.352 |
| Positive — precision | 0.647 | 0.638 | 0.620 |
| Positive — recall | 0.801 | 0.777 | 0.728 |
| Positive — F1 | 0.716 | 0.701 | 0.670 |
| Positive — accuracy_ovr | 0.878 | 0.870 | 0.865 |
| Negative — precision | 0.513 | 0.507 | **0.526** |
| Negative — recall | **0.808** | **0.832** | 0.760 |
| Negative — F1 | 0.628 | 0.630 | 0.622 |
| Negative — accuracy_ovr | 0.882 | 0.877 | **0.906** |
| Neutral — precision | **0.936** | 0.933 | 0.915 |
| Neutral — recall | 0.777 | 0.764 | **0.814** |
| Neutral — F1 | 0.849 | 0.840 | **0.861** |
| Neutral — accuracy_ovr | 0.811 | 0.803 | **0.814** |

**A more nuanced result than the offline gate predicted**: `precision_negative` — the specific
metric this whole experiment was built to move, stuck at 0.505/0.513/0.507 across v1/v2/v4 —
actually ticks up with v5 (0.513→0.526), and `accuracy_ovr_negative` (0.906) and neutral
F1/recall/accuracy_ovr are all v5's best of the three. It's not a clean win, though:
`recall_negative` drops to 0.760 (worse than both v2 and v4, and this pipeline's stated
priority metric), and `agreement_rate`/`macro_f1_vs_judge` both land worse than v2 (though
better than v4 on both).

**Disposition — not adopted**: despite the real, specific movement on `precision_negative`,
v5 doesn't beat v4 on the metric that decided v4's adoption (`recall_negative`), and its
`agreement_rate` sits between v2 and v4 rather than beating either outright. The user's decision
(2026-09-15) was to adopt **v4** for production — see the follow-up documenting that below.
Model saved locally to `models/sec-bert-base-financial-sentiment` (not published to the Hub);
offline metrics in `data/sentiment_finetune/test_metrics_sec_bert_base.json`, downstream
metrics in `nlp_use_v5.db`'s `eval_run` 35 / MLflow run `ed0e9ff7b4574bf98569bd141ac566a1`.
`src/pipeline.py`'s `MODEL_REVISIONS` is untouched by this experiment, and the real
`nlp_.db`/`nlp.db` were never opened for writing at any point in either v5 follow-up.

### Follow-up (2026-09-15, same day): Work item 9 decided — v4 adopted for production

With all three candidates now measured downstream (v2 baseline, v4 class-weighted, v5
base-checkpoint-swap — see the three follow-ups above), the user made the adoption call this
work item had been blocked on since 2026-09-14 (T-073): **v4 (class-weighted loss) is the
version this repo's pipeline pins**, chosen for the real `recall_negative` gain (0.808→0.832,
this pipeline's stated priority metric) despite the `agreement_rate`/`mean_severity` cost
disclosed in that follow-up's table. v3 (downsampled) and v5 (base-checkpoint swap) remain
documented, measured candidates, not adopted — v3 for its idiom-probe neutral collapse, v5 for
not beating v4 on `recall_negative` despite its own real `precision_negative` gain.

**Not yet live — the publish itself is blocked, not the decision.** The publish script
(`scripts/publish_finbert_financial_news_v4_2026_09_15.py`, model card carries the full offline
+ downstream comparison in one-vs-rest precision/recall/F1 form) is written and ready, and
`src/pipeline.py`'s `MODEL_REVISIONS` pin move is prepared to land in the same change once it
runs — but Claude Code's auto-mode classifier denies the Hub push itself as a "Create Public
Surface" action without explicit user permission (a Bash permission rule, or the user running
the script directly). `gamug/FinBERT-financial-news` still serves v2 until that step completes;
unlike v3's publish, this one is intended to be wired into the pipeline the same PR ships it in,
not left as an available-but-unpinned checkpoint, once unblocked. `sector_summary`'s pre-fix
rows and the ~439K
pre-pin-era `article_entities` rows (Work items 3/6's own non-blocking backfill items) are
unaffected by this change; existing `article_sentiment` rows are not retroactively
reprocessed with v4 — same precedent as Work item 1's checkpoint-pinning ("pinning going
forward is enough," `PLAN.md` non-goals) — a full-corpus resentiment backfill is a separate,
not-yet-scoped decision.

### Follow-up (2026-09-15, same day): sentiment's downstream eval narrowed to one-vs-rest metrics only

Presenting sentiment's results across this work item's several follow-ups (chat summaries, the
repository artifact, this doc) repeatedly mixed two different framings in the same report —
per-class one-vs-rest numbers (`precision_negative`, `recall_negative`, ...) alongside
aggregate, blended-across-all-three-classes numbers (`agreement_rate`, `macro_f1_vs_judge`,
`mean_severity`) — and that mixing was a real, repeated source of confusion (a table showing
"precision 0.647 / recall 0.801 / F1 0.716" for one class read as inconsistent with a different
number shown minutes earlier from a different evaluation set, and an aggregate metric sitting
next to per-class ones in the same table was misread as another class). At the user's explicit
request, `news_nlp.eval.metrics.aggregate_sentiment` (the downstream, MLflow-tracked LLM-judge
harness) now computes **only** one-vs-rest metrics for sentiment: `precision_<class>`,
`recall_<class>`, `f1_<class>`, `accuracy_ovr_<class>` (each HT-weighted and naive-pooled), plus
`n`/`parse_fail_rate` run bookkeeping. `agreement_rate`, `agreement_rate_<bucket>`,
`macro_f1_vs_judge`(`_naive_pooled`), and `mean_severity` are no longer computed for sentiment at
all — not just hidden from a report.

**Scoped to sentiment only** — `aggregate_category`/`aggregate_ner`/`aggregate_c_summary`/
`aggregate_sector_intro` are unchanged, still reporting their full complete metric set including
aggregate/overall numbers, per constitution AI behavior #12. `HEADLINE["sentiment"]` is
unaffected (`recall_negative` was already a per-class metric, not an aggregate one), so
`--check-regression` keeps working exactly as before.

**Constitution amended** (AI behavior #12, this session): the "complete metric set" principle
now explicitly carves out sentiment as one-vs-rest-only for its downstream methodology — see
`.specify/memory/constitution.md`, version bumped for the redefinition (a MAJOR change per this
project's own governance rule, since it narrows what #12 requires for one stage, not a new
addition). Every table in this document *before* this follow-up that shows
`agreement_rate`/`macro_f1_vs_judge`/`mean_severity` for sentiment is a historical record of a
run made under the old aggregation code and stays as-is — those numbers were real, computed
values at the time, not retroactively wrong; they're just no longer what a *future* sentiment
run will produce. `eval_run` rows already recorded in the database keep their full stored
`metrics_json` (including the old aggregate fields) regardless of this code change — only future
runs are affected.

Tests: `tests/news_nlp/test_eval_metrics.py`'s two sentiment tests that asserted
`agreement_rate`/`macro_f1_vs_judge` were updated (one renamed
`test_sentiment_per_class_f1_ht_and_naive_pooled`, asserting those keys are now *absent*; the
other's `agreement_rate` assertion replaced with an equivalent per-class `f1_positive` check).
Full suite (231 tests), ruff, and mypy all green.

### Follow-up (2026-09-14): c_summary full-article-vs-lead-cap mismatch confirmed and fixed

Started `PLAN.md` Work item 6 (summarization eval validation) by checking
the same suspicion already confirmed for sentiment (2026-09-08) and NER
(2026-09-12): does `c_summary`'s eval judge see the whole article the
pipeline actually summarized, or just the lead capped at `_MAX_BODY_CHARS`
(6000 chars)? Code inspection alone already pointed at yes — `pipeline.
run_company_summary_stage` -> `hierarchical_summarize_batch` chunks and
reduces over the *entire* `body_text` (via `build_company_summary_input`,
which concatenates the full, untruncated body), while `c_summary` was not
in `_UNCAPPED_STAGES` — but this project's standing rule is to measure
before fixing, not infer from code shape alone.

Measured directly against real data (`article_summary` joined to
`source.articles.body_text`, all 458,641 rows — no LLM calls, pure
sampling-layer arithmetic):

- **45,867 / 458,641 (10.0%)** of `article_summary` rows have `body_text`
  longer than the judge's 6000-char cap. `body_text` length: p50=2,763,
  p90=6,001, p95=7,499, p99=13,538, **max=156,053** (the same tail NER's
  2026-09-12 follow-up found, since both stages draw from the same
  article population).
- **Every one of those 45,867** is also a multi-chunk summary
  (`num_chunks > 1`) — i.e. this is exactly the population where
  `hierarchical_summarize_batch`'s reduce pass synthesizes content from
  more than one chunk, content a 6000-char judge view can never fully
  show.
- A broader **131,944 / 458,641 (28.8%)** have `num_chunks > 1`, but the
  other 86,077 of those are multi-chunk despite a body *under* 6000
  chars — that's the summarizer's own token budget (dense text hitting
  BART's ~1000-token leaf-chunk limit before 6000 characters), a
  separate phenomenon from the judge-cap mismatch, not evidence of it.
  The clean, judge-relevant figure is **10.0%**, not 28.8%.
- Smaller in magnitude than NER's 21.4% (2026-09-12), but the same
  structural bug: a real, double-digit-percent slice of the population
  where the eval judge is scoring the model against a truncated view of
  the text it actually processed.

**Fixed same-day**: added `c_summary` to `_UNCAPPED_STAGES`
(`src/news_nlp/eval/sampling.py`), the same fix shape as sentiment
(2026-09-08) and NER (2026-09-12) — it inherits the shared
`_SENTIMENT_MAX_BODY_CHARS` (100,000) safety ceiling, which does engage
for the 156,053-char tail article the same way it already does for NER.
Added `test_c_summary_uses_full_body_text_since_the_2026_09_14_fix`
(`tests/news_nlp/test_eval_sampling.py`), mirroring the existing
sentiment/NER tests of the same shape. Full hermetic suite still passes.

Not yet done (remaining Work item 6 steps): decide whether/how to address
`c_summary`'s weak `mean_coverage` (3.02/5, pre-fix baseline); re-run the
`c_summary` eval post-fix and record the result here; add a
faithfulness-only eval path for `sector_summary`'s `intro_text` (currently
zero coverage).

### Follow-up (2026-09-14, same day): post-fix c_summary re-run — coverage up, a previously-invisible hallucination gap revealed

Re-ran `c_summary` (`--stage c_summary --sample-size 1000 --seed 1`,
`code_version` `6312be9` — includes the uncap fix above) against the real
stores (`eval_run` 34, mlflow `72cf167d`). Strata drawn: `low_conf`
200/458,641, `target_chunks_1` 48/326,590, `target_chunks_2` 144/110,911,
`target_chunks_ge3` 288/20,940, `representative` 320/457,961.

| metric | 2026-09-08 baseline (pre-fix, old sampling design) | 2026-09-14 (post-fix, HT) | 2026-09-14 (naive pooled) |
|---|---|---|---|
| `mean_faithfulness` | 4.867 | 4.781 | 4.583 |
| `mean_coverage` | 3.02 | **3.639** | 3.304 |
| `mean_conciseness` | (not in baseline headline) | 4.568 | 4.362 |
| `pct_with_hallucination` | 0.054 | **0.085** | 0.154 |

Two things changed since the 2026-09-08 baseline at once, not one — an
honest confound, same shape as NER's 2026-09-12 before/after: (a) this
fix (judge now sees the whole article), and (b) the `num_chunks`-tiered
target stratification (`_CSUMMARY_CHUNK_TIERS`) was implemented the same
day as the 2026-09-08 baseline but *after* it was recorded, so the
baseline used the old, un-stratified low_conf/random design. The
headline deltas above can't be cleanly attributed to the uncap fix
alone.

What the per-stratum breakdown shows regardless of that confound —
`num_chunks >= 3` articles (`target_chunks_ge3`, exactly the population
this fix targets) are the weak spot on **both** metrics, more sharply
than the aggregate suggests:

| stratum | mean_faithfulness | mean_coverage |
|---|---|---|
| `target_chunks_1` (single-chunk) | 4.938 | 3.833 |
| `target_chunks_2` | 4.542 | 3.229 |
| `target_chunks_ge3` | **4.229** | **2.861** |

**`mean_coverage` genuinely improved** (3.02 → 3.64 HT) — plausible
mechanism: capped at 6000 chars, the judge could previously only check
the summary against the article's lead, so real coverage of later
material (which the hierarchical reduce does draw on) had no visible
ground truth to confirm — this fix lets the judge actually credit it.

**`pct_with_hallucination` went up, not down** (0.054 → 0.085 HT / 0.154
naive) — the opposite direction a pure "judge sees more, catches more
context" story would predict if the model were unchanged and only
overlooked-hallucinations were surfaced from behind the old cap. Plausible
mechanism: a hallucinated detail drawn from post-cap content previously
had no ground truth for the judge to check it against either, so it likely
read as unverifiable rather than confirmed-wrong under the old cap; the
uncap fix lets the judge actually confirm real fabrications it couldn't
see before. **Not yet independently verified** — no rationale-text audit
of the newly-flagged hallucination cases has been done this pass (unlike
the sentiment diagnosis's confusion-matrix/rationale audit); flagged here
as a plausible but unconfirmed mechanism, per this project's evidence bar.

**Treat `eval_run` 34 (this run) as the new post-fix `c_summary` baseline**
for regression tracking going forward — the 2026-09-08 numbers predate
both the uncap fix and the stratification redesign and are no longer a
fair `--check-regression` comparator, the same disposition NER's
pre-2026-09-12 baseline was given.

**Work item 6 step 2 (the `mean_coverage` decision), informed by this
data**: the fresh HT `mean_coverage` (3.64/5) is meaningfully better than
the stale 3.02/5 figure the item was originally scoped against, and the
per-stratum breakdown shows the residual weakness concentrated in
`num_chunks >= 3` articles (2.86/5, 4.6% of the corpus) rather than
spread evenly.

### Follow-up (2026-09-14, same day): a bigger output-length budget was tried and rejected

Tested the obvious next lever — `SUMMARY_MIN_OUTPUT_TOKENS`/
`SUMMARY_MAX_OUTPUT_TOKENS` (56/142, `src/pipeline.py`) are literally
`distilbart-cnn-12-6`'s own stock CNN/DailyMail generation defaults,
never retuned for longer, denser financial-news articles — before
touching the full corpus.

**Matched-pair experiment, not a fresh independent sample**: took 200
articles already judged in `eval_run` 34 (70/70/60 across the
`num_chunks` 1/2/3+ tiers), regenerated their summaries with
`min=100, max=220` — everything else byte-for-byte the same production
code path (`hierarchical_summarize_batch`, same model, same chunking) —
and re-judged the new summaries with the same judge/prompt, so the
comparison is the same articles, old vs. new settings, not two
different random draws. (One data wrinkle surfaced and fixed along the
way: reconstructing each article's original summarization input needed
the sentiment label/score that was live *when that summary was
generated* — `article_sentiment_v1`, the pre-fine-tune-swap table with
full 459,112-row coverage — not the current `article_sentiment` table,
which only holds 7,520 rows re-scored under the new fine-tuned model so
far.)

| metric | old (56/142) | new (100/220) | Δ |
|---|---|---|---|
| `mean_faithfulness` | 4.595 | 4.070 | −0.53 |
| `mean_coverage` | 3.315 | 3.515 | +0.20 |
| `mean_conciseness` | 4.380 | 3.725 | −0.66 |
| `pct_with_hallucination` | 15.0% | **35.5%** | more than doubled |

Per-tier coverage: `chunks=1` 3.814→**4.143** (crosses the 4/5 target),
`chunks=2` 3.143→3.357, `chunks>=3` (the actual weak spot) barely moves:
2.933→2.967.

**Rejected.** A bigger output budget does nudge coverage up, but mostly
on the single-chunk summaries that were already closest to fine — it
does almost nothing for the `chunks >= 3` population that's actually
driving the weak aggregate, and it more than doubles the hallucination
rate while dragging faithfulness and conciseness down with it. That
population's problem isn't output length; it's information loss
compounding across multiple summarize-of-summaries reduce passes
(`_reduce_pass`) — a bigger per-pass token ceiling doesn't undo hops of
lossy compression that already happened. This is the same discipline
this project applied to the sentiment precision investigation: test the
obvious lever on real data before adopting it, and report a negative
result plainly rather than force a metric up at a hidden cost.

**Decision (2026-09-14): accept `mean_coverage` as a deliberate
completeness-vs-correctness trade — no further fix attempted.** Every
measured lever for raising coverage moves the same knob the same
direction: more output allowed/forced means more room for the model to
pad with plausible-sounding invented detail alongside (or instead of)
genuinely-extracted fact, which is exactly what the rejected experiment
above showed happening (`pct_with_hallucination` more than doubling).
`mean_faithfulness` (4.78/5) is the stage's strongest metric and the one
that matters most for a *summary* specifically — a reader relies on a
summary to say only true things, more than to say every true thing. An
incomplete-but-accurate summary is a bounded, honest gap: a reader gets
less than the full article but nothing they read is wrong. A
complete-but-sometimes-fabricated one is worse: a reader has no way to
tell which parts are real without going back to the source, which
defeats the point of summarizing at all.

This mirrors, in the opposite direction, the same trade-off shape
sentiment's 2026-09-13 "pessimist model" decision made: sentiment
chose to risk more false alarms (lower precision) rather than risk
missing a real signal (lower recall), because a missed signal is an
unrecoverable blind spot downstream while a false alarm is visible and
discountable. Summarization's asymmetry runs the other way — a
fabricated detail is the unrecoverable failure mode here (a reader
can't tell it's fabricated), while an omitted detail is the visible,
recoverable one (the reader can tell the summary is thin and go read
the source). Recall was worth the false-alarm cost for sentiment;
completeness is not worth the fabrication cost for c_summary.

**What this decision does NOT close off**: a narrower fix scoped only
to the reduce pass itself (e.g. a larger `max_input_tokens` per chunk,
to cut the number of lossy reduce hops `chunks >= 3` articles go
through, rather than a bigger *output* budget on every pass) was never
tested and remains a legitimate future candidate if `chunks >= 3`
coverage (2.86/5) is later judged unacceptable on its own. This
decision is about not chasing the output-length lever further, not
about the coverage gap being permanently untouchable.

### Follow-up (2026-09-14, same day): sector_summary's intro_text eval added — a real, sizable faithfulness gap found

Work item 6 step 4. `sector_summary`'s `intro_text` sentence — the one
model-generated piece of an otherwise deterministic composition (see
`docs/modules/news-nlp.md`) — had **zero** evaluation before this. Added
a dedicated, narrower eval path rather than reusing `c_summary`'s: a new
`sector_summary` stage in `news_nlp.eval` (`_sector_summary_items` in
`sampling.py`, `SectorIntroVerdict`/`judge_sector_summary` in
`verdicts.py`/`judges.py`, `aggregate_sector_summary` in `metrics.py`,
`prompts/sector_summary.md`), judging `intro_text` for faithfulness only
against its own `facts_json` grounding — no coverage/conciseness scoring
(doesn't meaningfully apply to one stats-only sentence), never raw
article/company text (that's not what the model saw).

**T-054 (population check) first**: 3,628 `sector_summary` rows total,
one per `(gics_sector, gics_sub_industry, week)`, `intro_text` 180-383
chars, `facts_json` up to ~10K chars. Small enough to judge **the full
population every run**, not a sample — so this stage skips the
low_conf/target/representative stratification machinery entirely
(`sample_for_stage`'s `sector_summary` branch ignores `--sample-size`/
`--seed`/`--low-conf-frac`/`--target-frac`). Needs no SOURCE store either
— pure RESULTS-store composition, unlike every other stage.

**Full-population run** (`eval_run` 34 in the leveled `nlp_.db`, `n`
3628, `code_version` `2c18e07`):

| metric | value |
|---|---|
| `mean_faithfulness` | 4.05 / 5 |
| `pct_with_hallucination` | **42.2%** |

That hallucination rate is not noise — it's a real, systematic pattern,
confirmed by reading actual flagged rows rather than trusting the
aggregate number alone:

- **64% of flagged rows (985/1532)** are a fabricated source attribution
  distilbart-cnn tacks onto the sentence: `"...according to CNN.com's
  weekly Newsquiz"`, `"...according to analysts"`, `"...according to the
  latest article from 2 companies"` — none of which exist anywhere in
  `facts_json`. Reads as the model pattern-completing a news-summary
  sentence shape (articles often end with a source attribution) onto a
  purely statistical seed sentence it was never trained to summarize.
- Most of the rest are a **self-contradiction** pattern: the sentence
  states the correct percentages, then appends a second, wrong
  percentage for the same category — e.g. `"Sentiment was 0% positive,
  100% negative, 0% neutral, and 0% negative"` (grounding says 100%
  negative) or `"...50% negative, and 0% neutral, and 50% neutral"`
  (grounding says 0% neutral). Reads as a short-input generation
  artifact (repetition/degeneration), not a factual reasoning error.
- Faithfulness score distribution: `1`→5, `2`→560, `3`→758, `4`→238,
  `5`→2067 rows — a real bimodal split, not a uniform "slightly off"
  pattern: the majority (2067, 57%) are clean, but a large minority sits
  at 2-3 (1318, 36%), consistent with the two patterns above being
  frequent but not universal.

**Root cause, not yet fixed**: `sshleifer/distilbart-cnn-12-6` is a
*news-article* summarizer, repurposed here to turn a short, templated
stats sentence (`build_sector_intro_seed`, `src/news_nlp/sector_summary/
composition.py`) into prose — a task shape it was never trained on. This
is a newly-discovered, real gap distinct from `c_summary`'s (which
summarizes real article text, not a synthetic stats seed) — recording it
here as a finding, not deciding a fix in the same pass this eval path
was built, matching this project's step-by-step discipline (measure and
document first, decide the fix as its own step). See `SPEC.md` §13 for
the new open question this creates.

### Follow-up (2026-09-14, same day): confirmed the hallucination pattern is data-independent, then fixed it with a deterministic template

Before designing a fix, checked whether the 42.2% hallucination rate was
somehow an artifact of `sector_summary`'s existing rows predating the
sentiment-model swap (all dated 2026-08-29, built from the pre-fine-tune
sentiment data) rather than a property of the generation step itself.
Regenerated the full `sector_summary` table against fully-current
sentiment data and re-ran the eval (`eval_run` 34 again, n=3444, same
`code_version`):

| | pre-regeneration | post-regeneration (fresh sentiment) |
|---|---|---|
| `mean_faithfulness` | 4.05/5 | **3.83/5** |
| `pct_with_hallucination` | 42.2% | **50.2%** |
| fake-attribution share of hallucinations | 64% | 70% |

**Got worse, not better** — confirming the pattern is independent of the
underlying sentiment data, exactly as the root-cause diagnosis predicted:
regenerating with correct sentiment couldn't have fixed a bug in how the
*model paraphrases its own input*, because the input's factual content
was never the problem.

**Fixed the same day.** `build_sector_intro_seed`'s own output (`src/
news_nlp/sector_summary/composition.py`) is already a complete,
fully-grounded sentence — the summarization step was never adding
information, only degrading a sentence that was already correct.
`run_sector_summary_stage` (`src/pipeline.py`) no longer runs that seed
through `SUMMARY_MODEL` at all: `intro_text` is now the seed itself
(through `clean_generated_text` for whitespace normalization only),
verbatim. Zero hallucination risk **by construction**, not by
mitigation — the same "structural guarantee over probabilistic
mitigation" principle this stage's cross-company-blending design already
used (see `build_sector_intro_seed`'s own docstring). This also drops
the model load/GPU dependency for this stage entirely — `run_
sector_summary_stage` no longer touches `AutoTokenizer`/
`AutoModelForSeq2SeqLM` at all, and `sector_summary.model_name` now
records `"deterministic-template"` instead of `SUMMARY_MODEL`.

Verified against real production data: deleted and regenerated 5 real
`sector_summary` rows, output correct and clean — e.g. `"This week, the
Construction Materials sub-industry within Materials saw 1 article(s)
across 1 company, primarily about other. Sentiment was 0% positive, 100%
negative, and 0% neutral."` No fabricated attribution, no contradictory
percentage, by design (there's no generation step left to introduce
one). Full hermetic suite green, `test_summary_pipeline.py`'s
`run_sector_summary_stage` tests rewritten to assert the model is never
loaded, whether or not there's work pending.

**`SECTOR_SUMMARY_FORMAT_VERSION` bumped (2→3)** the same day
(`src/news_nlp/schema.py`) — this is the project's existing, designed
self-heal mechanism (`docs/modules/news-nlp.md`): a row below the
current version is treated as stale and regenerates via `INSERT OR
REPLACE` the next time `run_sector_summary_stage` runs, no separate
backfill script needed. The existing 3,444 rows (all built by the old
model-paraphrase path) will self-heal to the deterministic template the
next time the stage is run against the full corpus — not done as part
of this change itself (that's a real, deliberate production run, left
to the repo owner), but the mechanism is now armed and requires no
further code.

## What it evaluates

Four per-article stages, plus `sector_summary`'s own one-sentence
`intro_text` (added 2026-09-14) — the rest of `sector_summary` stays out
of scope, since it's deterministic composition (§9/§13, `SPEC.md`).

| stage | headline metric | also logged |
|---|---|---|
| `sentiment` | `recall_negative`¹ ² | agreement rate (per stratum), `macro_f1_vs_judge`, per-class P/R/F1, mean severity |
| `category` | `accuracy_vs_judge` ² | macro-F1, per-slug accuracy, model vs judge `other`-rate, mean severity |
| `ner` | `micro_f1` | span micro/macro P/R/F1, per-type F1, hallucination rate, miss rate. Error-only judge contract: it names just the `wrong` predicted spans + `missed` entities (not a verdict per span, which overflows on entity-dense articles); TP/FP/FN are derived from the predicted count. |
| `c_summary` | `mean_faithfulness` ² | mean coverage / conciseness (1-5), `pct_with_hallucination` |
| `sector_summary` | `mean_faithfulness` ² | `pct_with_hallucination` — faithfulness-only (no coverage/conciseness), full population every run, not a sample (see the 2026-09-14 follow-up above) |

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
  5. `sector_summary` has no floor at all — `--sample-size` is ignored for
     it, every run judges the full 3,628-row population (2026-09-14
     follow-up above).

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
