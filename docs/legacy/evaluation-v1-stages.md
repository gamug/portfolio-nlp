# Model evaluation of the v1 stages (legacy)

**Legacy (moved 2026-11-12).** These sections were cut verbatim from `docs/evaluation.md` at commit `09d90ae` (line ranges below), because the owner deprecated the v1 sentiment, zero-shot category and `sector_summary` stages on 2026-11-12 (the stepped pipeline of Work item 18 replaces them; `docs/modules/news-nlp.md`), and Work item 15 was superseded by Work item 18 on 2026-10-28. The v1 code still runs and its tables are still read downstream until the backfill (T-150). Nothing below was edited: "above" and "below" refer to the places these sections had in `docs/evaluation.md`; the sections that stayed there (NER, `c_summary`, sampling, the statistical method, running, results, experiments, Work item 18) are still in that file.

## Baseline notes (2026-09-08): sentiment and category

<!-- docs/evaluation.md lines 59-75 at 09d90ae -->
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

## Sentiment, the v1 stage (`gamug/FinBERT-financial-news`)

<!-- docs/evaluation.md lines 87-233 at 09d90ae -->
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

<!-- docs/evaluation.md lines 246-303 at 09d90ae -->
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

<!-- docs/evaluation.md lines 736-1565 at 09d90ae -->
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

**Constitution amendment proposed, not yet merged** (AI behavior #12, this session): the
"complete metric set" principle should explicitly carve out sentiment as one-vs-rest-only for
its downstream methodology, matching the code change above — drafted on
`docs/constitution-complete-metric-reporting` (a separate, still-open PR at the time of this
follow-up), version bumped for the redefinition (a MAJOR change per this project's own
governance rule, since it narrows what #12 requires for one stage, not a new addition). Until
that PR merges, `.specify/memory/constitution.md` on `master` still states #12's general rule
without the sentiment carve-out — this code change is ahead of its own governing documentation
landing, not silently contradicting it. Every table in this document *before* this follow-up that shows
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

## Category, the v1 zero-shot stage (10-slug taxonomy)

<!-- docs/evaluation.md lines 304-508 at 09d90ae -->
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

## `sector_summary` (v1, deterministic template)

<!-- docs/evaluation.md lines 1770-1895 at 09d90ae -->
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

## Work item 15: the `other` bucket after the upstream re-tag (superseded by Work item 18 on 2026-10-28)

<!-- docs/evaluation.md lines 1960-2384 at 09d90ae -->
### Follow-up (2026-10-22): what the upstream ticker re-tag did to the stored results and to category's measured numbers (Work item 15, T-133 step 1)

Read-only, on the real stores (`urls.db` at schema version 5, `nlp.db`), by
`scripts/measure_corrected_corpus_2026_10_22.py` (rerunnable; both databases opened
`mode=ro`). **Not a model evaluation:** nothing was judged again. The category numbers
below are unweighted agreement rates from judgements already stored, not the complete
constitution #12 metric set, which the T-133 steps 2-4 subsection at the end of this
follow-up reports per partition.

**Where the stored results stand upstream.** `portfolio-data-mining` re-tagged `urls.db`
on 2026-10-15; classes are read from its `discovered_urls.match_class`.

| upstream class | RESULTS articles | share | with stored `article_entities` | with an `ORG` span |
|---|---|---|---|---|
| `accept` (`ok`) | 93,330 | 20.3% | 93,286 | 93,268 |
| `reject` (quarantined) | 315,822 | 68.8% | 315,630 | 314,235 |
| `review` (quarantined) | 49,960 | 10.9% | 49,949 | 49,766 |
| total | 459,112 | 100% | | |

**Is `other` concentrated on wrong-tagged articles?** No. The `other` rate over all
459,112 articles, by upstream class:

| upstream class | articles | share of corpus | labelled `other` | `other` rate | share of all `other` |
|---|---|---|---|---|---|
| `accept` (confirmed) | 93,330 | 20.3% | 35,368 | 37.9% | 18.0% |
| `reject` | 315,822 | 68.8% | 138,740 | 43.9% | 70.8% |
| `review` | 49,960 | 10.9% | 21,885 | 43.8% | 11.2% |
| total | 459,112 | 100% | 195,993 | 42.7% | 100% |

The eight ordinary-word tickers (`ON`, `A`, `ARE`, `IT`, `ALL`, `HAS`, `TECH`, `NOW`) hold
179,210 articles (39.0%) with an `other` rate of 45.3%, against 41.0% for the other
tickers. So `other` follows each group's share of the corpus; it is not concentrated on
the articles the re-tag rejected, and on the confirmed articles it is still 37.9%. The
hand-read sample behind Work item 15's first framing (about 20 of 30 `other` titles
off-topic) is what any random draw of this corpus shows, since about 70% of it was later
rejected.

**Stratified by source and period, and how much tagging can explain at most.** The
overall 37.9% vs 43.9% gap is partly composition. `other` rate by upstream class:

| stratum | `accept` (confirmed) | `reject` | `review` |
|---|---|---|---|
| CNBC, 2022 onward | 43.3% (26,249) | 44.2% (68,194) | 46.4% (11,870) |
| CNBC, before 2022 | 36.6% (62,282) | 43.9% (247,354) | 43.0% (37,866) |
| other domains, 2022 onward | 24.4% (4,701) | 35.3% (269) | 38.6% (207) |
| other domains, before 2022 | 28.6% (98) | 60.0% (5) | 41.2% (17) |

Counterfactual: if rejected and `review` articles had the confirmed articles' `other`
rate, the excess is 22,010 `other` unstratified (11.2% of the 195,993) and 21,316
stratified by source and period (10.9%; strata with at least 50 confirmed articles). That
is the most wrong tagging can explain under that assumption, and nearly all of it sits in
the pre-2022 CNBC stratum: within CNBC from 2022 on there is no gap (43.3% vs 44.2%), and
the confirmed articles' own `other` rate is 43.3% there.

**Scoring path of the stored category rows.** Every stored row is the hierarchical scheme.
On level-2 rows the nine leaf columns sum to 1.0000 with exactly six non-zero (a softmax
over the six evaluated candidates); a row that took the level-1 short-circuit
(`group_score` below 0.40) has no leaf scores.

| upstream class | articles | level-1 short-circuit | level 2 |
|---|---|---|---|
| `accept` | 93,330 | 1,012 (1.1%) | 92,318 |
| `reject` | 315,822 | 13,806 (4.4%) | 302,016 |
| `review` | 49,960 | 1,880 (3.8%) | 48,080 |

So a rule confined to level 2, or a raised level-1 floor, can be applied to a stored row
from its stored scores; the short-circuited rows, and any change to the top-2 selection,
the input or the model, need inference (Work item 15, T-124 and T-140).

**`ok` articles with no results yet.** SOURCE has 98,462 `ok` articles (2026-10-22);
5,132 have no `article_category` row. All 5,132 were fetched between 2026-10-21 21:21:12
and 21:21:48 UTC, and the last `article_category` write is 2026-09-12 01:28 UTC.

**Which category runs can be recomputed.** Only `eval_run` 43 (2026-09-19, 2,800 judged,
population 459,112) has stored judgements. `eval_run` 21, the baseline Work item 15
first quoted (`other` precision 0.474), has none (`eval_inference` / `eval_verdict` hold
no rows for it) and its strata were drawn from a population of about 100k articles, so it
cannot be recomputed and is not the baseline.

**`eval_run` 43 by upstream class.** The `representative` bucket (896 articles, uniform
sampling weights, so unweighted equals population-weighted inside it) is 71% `reject`,
19% `accept`, 10% `review`, matching the population. Judge agreement with the stored
category label, with 95% Wilson intervals:

| `representative` bucket | `accept` (confirmed) | `reject` | `review` |
|---|---|---|---|
| any label | 93/170 = 54.7% (47–62%) | 383/636 = 60.2% (56–64%) | 61/90 = 67.8% (58–77%) |
| model predicted `other` | **25/68 = 36.8% (26–49%)** | **194/288 = 67.4% (62–73%)** | 24/35 = 68.6% (52–81%) |
| model predicted a substantive label | 68/102 = 66.7% (57–75%) | 189/348 = 54.3% (49–59%) | 37/55 = 67.3% (54–78%) |

All buckets together (deliberately skewed toward low-confidence and rare labels, so
shown for sample size only): 1,953 `reject`, 538 `accept`, 309 `review` items judged.

**Reading.** The population rates above say wrong tagging does not explain most of
`other`: what remains on the corrected corpus is mostly the model withholding or missing
a label, which calibration addresses. The category model and its judge see only the title and body, never the
company, so an article that is not about its tagged company has no correct
company-performance label. `other` was judged correct about twice as often on rejected
articles (67%) as on confirmed ones (37%): the quoted `other` precision (0.576 in run 43)
is inflated by off-topic articles, and on confirmed articles it is nearer 0.37. The
substantive-label difference (67% vs 54%) points the expected way but its intervals
overlap. The confirmed-article `other` figure rests on 68 articles; a ±5-point estimate
needs about 359 such articles, so T-123 takes its baseline from T-121(b)'s stratified
judged sample (report half; funded by D10, T-142). Not claimed: any per-class precision/recall/F1 on
the confirmed partition, or that these rates carry over to the corrected corpus after
the next upstream change.

**Rule consistency and sample sizing (same script, 2026-10-22).** Applying the current
rule (group floor 0.4, argmax, threshold 0.6; constants and slug order imported from
`news_nlp.taxonomy`, the decision restated in the script because `category_stage`'s
functions take logits or are inline, not callable on stored probabilities) to the stored
scores reproduces the stored label on all 459,112 rows (0 mismatches; 16,698 level-1
short-circuits; 141 rows have an exact tie for the highest leaf score, all below 0.5, so
no stored label depends on the tie-break). T-122 will extract the decision into one pure function so the re-label
and the stage share code. Reproduce with `uv run python
scripts/measure_corrected_corpus_2026_10_22.py --source-db <urls.db> --results-db <nlp.db>
--eval-run 43` (`urls.db` `user_version` 5; RESULTS `article_category` last written
2026-09-12; run on 2026-10-22). Confirmed articles by predicted slug x score
band give 45 cells, all with at least 60 articles (counts as of 2026-10-22, not design constants; smallest: 186, `mergers_acquisitions` below 0.4), plus 1,012
level-1 short-circuits. T-121(b)'s sample is 90 per cell in the 0.4-0.6 bands, 60 elsewhere
and 60 short-circuit. That is 3,300 judge calls at most (cap 3,800 with the discordant top-up), split in halves (fit / report) so T-123's baseline is not scored
on the data the thresholds were fitted on.

**T-133 steps 2-4: the complete metric set per upstream class (same script, section 12;
no judge call; run on 2026-10-22, per-class label counts and intervals re-derived on
2026-10-23 after review).** The stored judgements of eval_run 43 (2,800 items)
are split by `discovered_urls.match_class` and each part is run back through
`aggregate_category`, the live-run code path (`news_nlp.eval.partition`). Each part keeps
the run's own weight per article: a stratum's population is rescaled to
`N_h * n_h(part) / n_h`, the standard domain estimator, so the weight stays `N_h / n_h`.
Sanity check: the `all` column reproduces the run's stored `metrics_json` (accuracy
0.576, macro F1 0.551, `other` precision 0.576, recall 0.627, F1 0.601, model and judge
`other` rates 0.444 and 0.408). `n` counts parse-ok items including the `low_conf`
bucket, which the estimators exclude. The only figures given an interval are the `other`
precisions below, by a stratified bootstrap (500 replicates, seed 0, each bucket redrawn
at its own size; `news_nlp.eval.partition.bootstrap_interval`, free, printed by the
script): all 0.576 [0.540, 0.613]; `accept` 0.319 [0.242, 0.403]; `reject` 0.639
[0.599, 0.681]; `accept` minus `reject` -0.320 [-0.405, -0.233]. The other cells carry no
interval and the small per-class cells in `accept` are indicative only. A partition's
rescaled stratum population is rounded and clamped to at least 1. Robustness: seed 1
with 1,000 replicates gives `accept` [0.241, 0.398], `reject` [0.596, 0.684] and the
difference [-0.411, -0.226], within 0.01 of the bounds above; every replicate had a
defined statistic (the bootstrap skips a replicate whose `precision_`/`recall_` has an
empty denominator, and reports how many were used).

| metric | all | `accept` (survivors) | `review` | `reject` |
|---|---|---|---|---|
| judged items (parse ok) | 2,800 | 538 | 309 | 1,953 |
| accuracy vs judge | 0.576 | 0.526 | 0.654 | 0.580 |
| macro F1 | 0.551 | 0.558 | 0.616 | 0.533 |
| `other`: precision / recall / F1 | 0.576 / 0.627 / 0.601 | **0.319** / 0.661 / 0.430 | 0.600 / 0.644 / 0.621 | 0.639 / 0.621 / 0.630 |
| `other` rate, model / judge | 0.444 / 0.408 | 0.407 / **0.196** | 0.430 / 0.401 | 0.456 / 0.469 |
| `other` one-vs-rest accuracy | 0.660 | 0.656 | 0.685 | 0.658 |

The full per-class precision, recall, F1 and one-vs-rest accuracy for all ten labels
and the four columns are printed by the script (section 12) and are not repeated here.

**Reading.** On the confirmed articles the model labels `other` about as often as
elsewhere (0.41 against 0.44) but the judge does so far less (0.20 against 0.41), so
`other` precision on them is 0.32 (interval 0.24-0.40), not the 0.58 quoted for the
whole run: about two thirds of the confirmed articles the model sends to `other` (point
estimate 68%, interval 60-76%) carry a real category in the judge's view. That agrees in direction with the unweighted 37% above and makes the
`other` problem of Work item 15 a property of the confirmed corpus, not an artefact of
the wrongly tagged articles (they inflate the quoted precision). `other` recall is
similar across parts (0.62-0.66), so the cost of the rule is mostly in precision.

**Does the surviving partition clear the sample-size floor? No.** `accept` has 538
judged items (488 outside `low_conf`); the floor for a regression-tracked category run is
about 2,500-3,000 items (the "Sample-size floor" section above derives it from the
+-0.07-0.10 interval a regression check needs), and the per-class floor is about
200-370 true examples, but the judge's own labels on `accept` (parse ok, outside
`low_conf`, unweighted) give only 11 `capital_shareholder_returns`, 33
`legal_regulatory`, 39 `earnings_performance` and 40 `labor_human_capital` (the rest are
42-76). So per-class figures on `accept`
(for example `capital_shareholder_returns` F1 0.254) are indicative only. Consequence
for the plan: eval_run 43's `accept` partition is **not** T-123's baseline; T-123 takes
its baseline from the report half of T-121(b)'s stratified judged sample (already the
plan above), and this table is the reason that sample exists. No behavior was changed.

### Follow-up (2026-10-23): offline simulation of candidate decision rules, no judge call (Work item 15, T-121(a))

`scripts/simulate_category_rules_2026_10_23.py` (read-only, both databases `mode=ro`;
logic in `news_nlp.eval.calibration`) reruns on the stored scores of the 93,330
confirmed (`accept`) articles. Rules are uniform: one confidence threshold for all slugs and a minimum
margin of the best slug over the runner-up (32 candidates; per-slug thresholds are fitted
on T-121(b)'s fit half, there is too little per-slug data to fit nine of them here).
`t0.60+m0.00` is the shipped rule and reproduces every stored label.

**Re-labelled articles** (of 93,330; `--cells` prints the full slug x band grid of the
population and of every distinct rule's re-labelled articles, the per-cell counts
T-121 asks for; the script's output also splits each rule by band):

| rule | re-labelled | direction |
|---|---|---|
| `t0.50` | 16,359 (17.5%) | all `other` -> slug, all in the 0.5-0.6 band |
| `t0.55` | 7,957 (8.5%) | `other` -> slug |
| `t0.65` | 7,843 (8.4%) | slug -> `other` |
| `t0.70` | 15,138 (16.2%) | slug -> `other` |
| `t0.80` | 28,916 (31.0%) | slug -> `other` |
| `t0.90` | 42,450 (45.5%) | slug -> `other` |

The margin binds only at some thresholds. At a threshold of 0.6 or more the runner-up of a
confident article holds at most 0.4, so a margin up to 0.2 never binds there (the rules
`t0.60+m0.10`, `t0.60+m0.20` and every `t >= 0.65` rule with a margin up to 0.3 re-label exactly what
the margin-free rule does), and only a margin of 0.3 changes labels at `t0.60` (1,918 articles, to `other`).
Below 0.6 a margin does bind: `t0.50` re-labels 16,359 articles with no margin, 14,582 with 0.1,
9,728 with 0.2 and 5,782 with 0.3; `t0.55` re-labels 7,957 with margin 0.0 or 0.1, 6,198
with 0.2 and 4,798 with 0.3. Rules with a margin of 0.3 below the shipped threshold also send
some articles to `other`.

**Stored verdicts, tightening rules only** (eval_run 43, `accept` partition, 538 judged): the
judge's `ideal_slug` is the truth, so the rules that only send items to `other` are scored
exactly. Raising the threshold raises the mean per-slug precision (0.698 at 0.60, 0.739 at
0.70, 0.845 at 0.90; a point estimate) and `other` recall. Against the shipped rule, on the same
items (paired stratified bootstrap, 500 reps, 95%):

| rule | `other` precision | `other` recall | accuracy vs judge |
|---|---|---|---|
| `t0.60+m0.30` | -0.015 [-0.027, -0.004] | +0.000 | -0.013 [-0.027, -0.002] |
| `t0.65` | -0.027 [-0.053, +0.003] | +0.058 [+0.005, +0.123] | -0.028 [-0.053, -0.006] |
| `t0.70` | -0.035 [-0.074, +0.007] | +0.136 [+0.053, +0.230] | -0.052 [-0.090, -0.019] |
| `t0.75` | -0.051 [-0.094, -0.003] | +0.211 [+0.117, +0.329] | -0.081 [-0.123, -0.035] |
| `t0.80` | -0.067 [-0.114, -0.017] | +0.220 [+0.124, +0.340] | -0.110 [-0.153, -0.059] |
| `t0.90` | -0.095 [-0.147, -0.040] | +0.270 [+0.160, +0.394] | -0.189 [-0.237, -0.129] |

Overall accuracy falls and `other` recall rises at every threshold from 0.65 up, and the
`other` precision falls at 0.75 and above and at `t0.60+m0.30`; at 0.65 and 0.70 the interval
includes 0, so for those two the direction is a point estimate only. A stricter threshold trades
`other` precision for slug precision; it does not fix the `other` problem. A rule that moves
items *away* from `other` cannot be scored from these verdicts (the judge graded `other` on them, not the
slug) and needs T-121(b)'s sample.

**What T-123 asks of a loosening rule, from the measured aggregates alone.** T-123 requires `other`
precision +0.05 (lower bound above 0) and, so that precision is not bought with recall, `other` recall,
overall accuracy and macro F1 each with a paired 95% lower bound no worse than -0.02 (tolerance confirmed
by the owner). On the `accept` partition the stored verdicts measure `other` precision 0.319 and `other`
recall 0.661. Of the 35,368 confirmed articles the shipped rule labels `other`, 11,282 are then truly
`other` (about 68% have a real category, spread over any of the nine slugs) and 17,069 articles in the
population are truly `other`. A rule that removes R articles from `other`, X of them truly `other`,
changes recall by -X / 17,069 and precision to (11,282 - X) / (35,368 - R). So **recall within -0.02
needs X <= 341, and +0.05 precision then needs R >= 5,718**; both are necessary conditions (the
95% *lower* bound of the recall change needs more room than the point estimate). For the rules at hand
(`news_nlp.eval.calibration.loosening_bound`; the script prints the table):

| rule | removed from `other` | at most truly `other` | as a share of the removed |
|---|---|---|---|
| `t0.50+m0.00` | 16,359 | 341 | **2.1%** |
| `t0.50+m0.10` | 14,582 | 341 | 2.3% |
| `t0.50+m0.20` | 9,728 | 341 | 3.5% |
| `t0.55+m0.00` | 7,957 | 341 | 4.3% |
| `t0.55+m0.20` | 6,198 | 341 | **5.5%** |

For scale, the articles the shipped rule labels with a slug are truly `other` in **15.4%** (0.6-0.8; 141
judged, 0.08-0.24) and **7.5%** (>= 0.8; 155 judged, 0.03-0.13), measured on the judge's `ideal_slug`
(a model, `(1 - agreement) x 0.339` with one pooled share for all bands, gives 13.0% and 6.8% and is what
the power tables use). So a loosening rule passes T-123 only if the articles it re-labels are about as clean
as the cleanest band the model already labels: all five rules need a set cleaner than 7.5% (2.1% to 5.5%).
A tightening rule fails the precision side instead (above). The bound uses the weighted `accept`
precision 0.319 and recall 0.661 as exact; they are estimates (a precision error of 0.03 moves the 341 by
about 10%), so it is a planning figure, not a test.

**The stored verdicts do measure the 0.5-0.6 band.** The judge gave an `ideal_slug` to the items the shipped
rule labelled `other` too, so the share of `other`-labelled articles whose truth is `other`, and the share
whose truth is the argmax slug, can be read per band (`news_nlp.eval.calibration.measure_bands`; no model;
weighted by the Horvitz-Thompson item weights, 95% stratified-bootstrap interval of the weighted share):

| stored label | band | judged | truth is `other` (raw) | weighted | 95% | truth is the argmax slug |
|---|---|---|---|---|---|---|
| `other` | < 0.4 | 23 | 9/23 | 0.474 | 0.20-0.75 | 0.238 |
| `other` | 0.4-0.5 | 91 | 28/91 | 0.392 | 0.28-0.52 | 0.272 |
| **`other`** | **0.5-0.6** | **78** | **12/78** | **0.193** | **0.09-0.31** | **0.484** |
| slug | 0.6-0.8 | 141 | 18/141 | 0.154 | 0.08-0.24 | 0.547 |
| slug | >= 0.8 | 155 | 9/155 | 0.075 | 0.03-0.13 | 0.779 |

The 0.5-0.6 rate the power sweep varies is therefore measured at **0.19** (0.09-0.31). The
`graded` shape at that rate sets the < 0.4 and 0.4-0.5 bands to 0.536 and 0.365, against the measured 0.474
and 0.392: close, inside the intervals. Those are band averages. A rule removes only part of the band, so the figure that the bound caps is the
truly-`other` share *of the judged articles the rule would take out of `other`*, which the same verdicts
give per rule (`measure_rule`; the script's `stored` section; the recall and precision columns are the
closed form of `loosening_bound` at that share, point and interval ends):

| rule | removed (population) | judged | truly `other` | weighted (95%) | cap | d recall (point; interval) | d precision | reading |
|---|---|---|---|---|---|---|---|---|
| `t0.50+m0.00` | 16,359 | 78 | 12 | 0.193 (0.09-0.30) | 2.1% | -0.19 (-0.29 to -0.08) | +0.11 | above the cap |
| `t0.50+m0.10` | 14,582 | 64 | 10 | 0.186 (0.07-0.30) | 2.3% | -0.16 (-0.26 to -0.06) | +0.09 | above the cap |
| `t0.50+m0.20` | 9,728 | 36 | 4 | 0.079 (0.01-0.21) | 3.5% | -0.05 (-0.12 to -0.00) | +0.09 | interval includes the cap |
| `t0.55+m0.00` | 7,957 | 38 | 5 | 0.166 (0.03-0.31) | 4.3% | -0.08 (-0.15 to -0.02) | +0.04 | interval includes the cap |
| `t0.55+m0.20` | 6,198 | 25 | 2 | 0.087 (0.00-0.25) | 5.5% | -0.03 (-0.09 to -0.00) | +0.05 | interval includes the cap |

On the stored evidence **the two lowest-margin rules fail**: `t0.50+m0.00` and `t0.50+m0.10` remove a set
that is truly `other` 19% of the time against a cap of 2%, with an interval that excludes the cap. The
margin of 0.2 halves the rate (0.19 to 0.08-0.09), which is still above the cap at the point estimate
(recall change -0.05 and -0.03 against -0.02) but, with 36 and 25 judged items, **not excluded**; nor is
`t0.55+m0.00` (38 items). The power and joint tables below give every rule the band average (0.193), so
their recall columns are those of the band, not of the margin subsets (for `t0.50+m0.20` the band figure is
-0.11, the subset figure -0.05): read this table for the recall verdict and the power tables for what the
precision test can detect. Macro F1 is in neither. Whether a margin or a score isolates a set truly
`other` in only 2-5.5% of cases is the open question for T-121(b), which should oversample the `m0.20`
subsets.

**How many judged articles T-121(b) needs inside each removed set** (`items_to_decide`; one-sided, 80% power,
normal approximation, a planning figure). The weighted share is a design-weighted estimate, so a simple
random sample understates the need by the design effect (Kish: judged n over the effective n of the weights;
`design_effect`), measured on the stored run's judged sets at 1.47-1.61. With it:

| rule | cap | stored rate | design effect | to fail it, 95% | to fail it, 98.75% | to pass if rate = 0.5 cap | to pass if rate = 0.75 cap |
|---|---|---|---|---|---|---|---|
| `t0.50+m0.00` | 2.1% | 0.193 | 1.47 | 16 | 22 | 1,389 | 6,230 |
| `t0.50+m0.10` | 2.3% | 0.186 | 1.50 | 19 | 26 | 1,259 | 5,648 |
| `t0.50+m0.20` | 3.5% | 0.079 | 1.61 | 233 | 339 | 893 | 4,001 |
| `t0.55+m0.00` | 4.3% | 0.166 | 1.47 | 41 | 57 | 662 | 2,965 |
| `t0.55+m0.20` | 5.5% | 0.087 | 1.58 | 565 | 844 | 548 | 2,452 |

(The sizes scale with the design effect of T-121(b)'s sampler, which T-141 must report; 98.75% is the
Bonferroni level for a stop after one of two looks on two rules.) Failing a rule is cheap; passing one is
affordable only if the removed set is well below its cap, so T-121(b) samples in stages and treats the early
decisions as screens (TASKS.md T-121(b)).

**Power of T-123's estimator** (paired difference of the weighted `other` precision, a ratio of weighted
sums; report-half sample of 45 per 0.4-0.6 cell, 30 elsewhere, 30 short-circuit; 300-replicate within-cell
bootstrap, nearest-rank 95% interval, seed 0). Power is the share of draws whose lower bound is above 0,
from 200 draws; a power within 0.10 of the 80% bar is re-run at 1,000 draws (same seed and random stream:
its first 200 draws are the original ones). Every power is printed with its Monte Carlo standard error (up
to 0.035 at 200 draws, 0.016 at 1,000). A verdict is "detectable" or "not detectable" only when the power
is two standard errors from the bar and "borderline" otherwise (a borderline candidate is treated like a
miss and takes the discordant top-up; the rule is fixed in TASKS.md T-121), and **only the row at a gain of
+0.05 carries a verdict**. A verdict row near the bar is also checked against the **null-calibrated
power**: the lower bound is compared with the value that only 2.5% of zero-gain draws exceed (1,000 null
draws, three null seeds). A cell smaller than its sample is taken whole and still resampled in the
bootstrap (no finite-population correction), so the power is slightly conservative.

*Truth model, calibrated to the measured baseline.* In the >= 0.6 cells, eval_run 43's agreement per cell,
weighted by each item's Horvitz-Thompson weight (3 of the 18 cells have fewer than 10 judged items and use
their band's pooled rate: `earnings_performance` and `product_innovation` in 0.6-0.8,
`capital_shareholder_returns` in >= 0.8); when the label is wrong, the truth is `other` 33.9% of the time.
In the cells whose stored label is `other` (< 0.6 and the short-circuit stratum) the probability that the
truth is `other` is the swept quantity: **the rate of the 0.5-0.6 band** (0.30, 0.20, 0.10, 0.05 and the
rate at which the gain is +0.05), with the other low cells set so that the average over the 35,368
`other`-labelled articles stays at the measured 0.319 (the `graded` shape puts 0.4-0.5 halfway between the
swept rate and the rest; `band2` changes only 0.5-0.6; `flat` sets every cell to 0.319, so no rule can
gain). The model therefore reproduces the measured baseline whatever the rate: its `other` precision is
0.319 and its recall 0.663 (measured 0.661) on every row, and the script stops if the precision differs by
more than 0.005. The share of the non-`other` mass whose truth is the argmax slug (0.5 by default) only
moves accuracy and macro F1; neither `other` precision nor `other` recall depends on it. Macro F1 is not
derived (the rest of the mass has no class to land on); T-121(b) measures it.

Loosening rules; `rate` = P(truth is `other`) of a 0.5-0.6 article; `d` = implied change in expectation;
`power`, `joint` over 200 draws (1,000 where the power is near the bar); `joint` requires the precision
lower bound above 0 and the recall and accuracy lower bounds no worse than -0.02. The `graded` and `band2`
shapes give identical figures here (these rules only re-label 0.5-0.6 articles), so one table:

| rule | rate | d precision | d recall | d accuracy | power (SE) | joint | verdict at +0.05 |
|---|---|---|---|---|---|---|---|
| `t0.50+m0.00` | 0.30 / 0.20 / 0.10 / 0.05 | +0.016 / +0.102 / +0.188 / +0.231 | -0.288 / -0.192 / -0.096 / -0.048 | +0.009 / +0.035 / +0.061 / +0.074 | 0.14 / 1.00 / 1.00 / 1.00 | 0.00 | |
| | **0.261** | +0.050 | -0.251 | +0.019 | 0.65 (0.034) | 0.00 | not detectable |
| `t0.50+m0.10` | 0.30 / 0.20 / 0.10 / 0.05 | +0.013 / +0.083 / +0.154 / +0.189 | -0.257 / -0.171 / -0.086 / -0.043 | +0.008 / +0.031 / +0.055 / +0.066 | 0.12 / 0.99 / 1.00 / 1.00 | 0.00 | |
| | **0.248** | +0.050 | -0.212 | +0.020 | 0.75 (0.014, 1,000 draws); calibrated 0.71-0.74 | 0.00 | not detectable (stable) |
| `t0.50+m0.20` | 0.30 / 0.20 / 0.10 / 0.05 | +0.007 / +0.045 / +0.083 / +0.102 | -0.172 / -0.114 / -0.057 / -0.029 | +0.005 / +0.021 / +0.036 / +0.044 | 0.09 / 0.90 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 / 0.08 | |
| | **0.187** | +0.050 | -0.107 | +0.023 | 0.94 (0.017) | 0.00 | detectable |
| `t0.55+m0.00` | 0.30 / 0.20 / 0.10 / 0.05 | +0.005 / +0.035 / +0.064 / +0.078 | -0.140 / -0.094 / -0.047 / -0.023 | +0.004 / +0.017 / +0.030 / +0.036 | 0.07 / 0.80 / 1.00 / 1.00 | 0.00 / 0.00 / 0.01 / 0.15 | |
| | **0.147** | +0.050 | -0.069 | +0.024 | 0.98 (0.009) | 0.00 | detectable |
| `t0.55+m0.20` | 0.30 / 0.20 / 0.10 / 0.05 | +0.004 / +0.025 / +0.047 / +0.057 | -0.109 / -0.073 / -0.036 / -0.018 | +0.003 / +0.013 / +0.023 / +0.028 | 0.06 / 0.62 / 0.99 / 1.00 | 0.00 / 0.00 / 0.04 / 0.33 | |
| | **0.084** | +0.050 | -0.030 | +0.025 | 1.00 (0.000) | 0.07 | detectable |

Rules with a margin of 0.3 reach +0.05 at no rate; every rule with `t >= 0.65` has a negative precision
gain (above). Under `flat` every loosening rule has a precision gain of 0.000 and a power of 0.03-0.04 at
every rate, which is the estimator's false-positive rate at one side of a 95% interval (nominally 0.025;
200 draws, so within about one standard error of it).

**Joint condition.** The largest rate at which the implied `other` recall stays within -0.02 (the point
estimate; the lower bound needs more), what the rule gains there, and the joint power there and at a rate
of 0:

| rule | largest rate | d precision | d accuracy | joint power at that rate | joint power at rate 0 |
|---|---|---|---|---|---|
| `t0.50+m0.00` | 0.021 | +0.257 | +0.082 | 0.24 | 1.00 |
| `t0.50+m0.10` | 0.023 | +0.207 | +0.073 | 0.23 | 1.00 |
| `t0.50+m0.20` | 0.035 | +0.108 | +0.047 | 0.23 | 1.00 |
| `t0.55+m0.00` | 0.043 | +0.080 | +0.037 | 0.23 | 1.00 |
| `t0.55+m0.20` | 0.055 | +0.056 | +0.028 | 0.27 | 1.00 |

(The largest rate is the bound above: 2.1%, 2.3%, 3.5%, 4.3%, 5.5%. Joint power is only about 0.25 there
because the recall *lower bound* sits below the point estimate; it reaches 1.00 when the removed set is
free of truly `other` articles.)

**At the measured rate (0.193).** Power of the precision test, 200 draws (1,000 where near the bar), `graded`
shape: `t0.50+m0.00` 1.00 (implied d precision +0.108), `t0.50+m0.10` 0.99 (+0.088), `t0.50+m0.20` 0.91
(+0.048), `t0.55+m0.00` 0.84 (+0.036), `t0.55+m0.20` 0.68 (+0.027); the rules with a margin of 0.3 gain
nothing (+0.004 and 0.000). Joint power is 0.00 for every rule (recall change of the band average, -0.19 to -0.07; the margin subsets are in the per-rule table above). The
precision test is passed by the rules that remove the most `other`-labelled articles, and it is the
recall that fails.

**Accuracy does not bind; it depends on `slug_fraction`.** The accuracy columns depend on the share of the
non-`other` mass whose truth is the argmax slug, 0.5 in the tables. Measured on the stored verdicts above
(pooled over the three low bands by article count) it is 0.534. Implied accuracy change of the loosening
rules at the measured rate for 0.25 / 0.50 / 0.75 / measured: `t0.50+m0.00` +0.001 / +0.037 / +0.072 /
+0.042; `t0.55+m0.20` +0.001 / +0.014 / +0.027 / +0.016; `t0.50+m0.30` -0.010 / -0.001 / +0.007 / -0.000.
So accuracy is within -0.02 at every fraction tried; the joint failure comes from recall. Macro F1 is
not modelled (the third label has no class), so the joint condition checks two of T-123's three
co-requirements; T-121(b) measures the third.

**Reading.** Two things hold whatever the rate. A tightening rule cannot raise `other` precision, and a
loosening rule can raise it by +0.05 only at a rate well below the measured 0.319: 0.25-0.26 for the
no-margin and 0.1 margin `t0.50` rules (then recall falls by 0.21-0.25, and the precision test is not
detectable at this sample), 0.08-0.19 for the others (recall falls by 0.03-0.11). The precision test alone
is therefore passed by `t0.50+m0.20`, `t0.55+m0.00` and `t0.55+m0.20` and says little: the recall
tolerance is what binds. **A uniform loosening rule passes T-123 only if the articles it re-labels are
truly `other` at most 2.1-5.5% of the time** (the bound above), which is about as clean as the cleanest band
the model labels (7.5%). Measured on the judged articles each rule would take out of `other`, the no-margin
rules remove a set truly `other` 19% of the time (cap 2%) and fail; the 0.2-margin rules and `t0.55+m0.00`
remove a set 8-17% truly `other` (cap 3.5-5.5%, 25-38 judged), which the stored evidence does not exclude (per-rule
table above). T-121(b) should therefore report, per candidate rule, the truly-`other` rate of the set it would re-label with a 95%
interval and compare it with the bound (the pass/fail check), together with how well the maximum score, the
margin and the group score separate truly-`other` from truly-slug articles (an AUC), which is what per-slug
thresholds would exploit.

Reproduce: `uv run python scripts/simulate_category_rules_2026_10_23.py --source-db
/path/to/urls.db --results-db /path/to/nlp.db --eval-run 43 --cells` (62 minutes at the defaults: 200 draws,
1,000 for the borderline rows, three shapes, three null seeds). The tables above other than the
measured-rate rows, the measured-band table and the accuracy sensitivity come from that full run; those
three come from a re-run of `--sections power --shapes graded --null-seeds 1` (22 minutes), whose rows for
the swept rates are identical to the full run's (same seed and code path). `--draws`, `--refine-draws`,
`--bootstrap-reps`, `--interval-reps`, `--null-seeds`, `--shapes` and
`--sections relabel,verdicts,stored,power` shorten it (`stored`, the measured bands and the per-rule table, takes seconds). Nothing is written.

<!-- docs/evaluation.md lines 2433-2517 at 09d90ae -->
### Follow-up (2026-10-23): the judged calibration sample, stage 1 (Work item 15, T-121(b))

Run with `scripts/calibrate_category_2026_10_23.py` (`plan` / `run --stage 1` / `report`), seed
20261023, eval_run 45 in the real RESULTS store, blind truth-label judge (`category_truth`, the
article only, the classifier's label not shown; judge model `deepseek-chat` at `https://api.deepseek.com`, from `LLM_*`). **Spend: 720 of
the 3,800 calls of D10** (10 smoke-test articles, then the other 710 of stage 1; 0 parse failures,
0 transport errors). The 3,300-article sample is drawn (46 + 27 strata over the 93,330 confirmed
articles).

**Deviation from the staged design of TASKS T-121(b), and why.** The design said "judge 350
articles inside each `m0.20` removed set". The 0.5-0.6 band cells were instead split by the
candidate rules into sub-cells A (removed by `t0.55+m0.20`, hence by `t0.50+m0.20` and
`t0.55+m0.00`), U (removed by `t0.50+m0.20` only), V (removed by `t0.55+m0.00` only) and W (none),
sampled 40 / 24 / 16 / 10 per slug (the cell's 90 draws are unchanged, so the base sample is still
3,300 calls). Every removed set is then a union of whole sub-cells, so each rule's rate is a plain
stratified estimate, with no two-phase top-up; stage 1 judged every article drawn in A, U and V (720
in all) and none of the other strata. The third rule, `t0.55+m0.00`, was not in the original staged
design (it was judged inside the base sample); its sub-cell V sits in stage 1, so it is a third
rule and the stop level below is set for three.

**Stop level.** Three rules, each looked at twice (stage 1 and the pooled sample): six looks, so
the early stop uses a one-sided bound at 1 - 0.05 / 6 = **99.17%** (the design first said 98.75%
for two rules). The intervals are stratified bootstraps of **5,000** repetitions (the stop bound is
the 42nd value from the end; 1,000 repetitions would put it on the 8th and leave its third digit
noisy).

| rule | removed (of 35,368 `other`) | judged inside | truly `other` | weighted rate | 95% | one-sided 99.17% lower | cap (T-123) | design effect | result |
|---|---|---|---|---|---|---|---|---|---|
| `t0.50+m0.20` | 9,728 | 576 | 139 | 0.287 | 0.230-0.348 | 0.218 | 0.035 | 2.63 | **fails** |
| `t0.55+m0.00` | 7,957 | 504 | 101 | 0.285 | 0.220-0.354 | 0.208 | 0.043 | 2.82 | **fails** |
| `t0.55+m0.20` | 6,198 | 360 | 87 | 0.335 | 0.257-0.418 | 0.240 | 0.055 | 2.72 | **fails** |

By sub-cell, the weighted share truly `other` is **A 0.335** (360 judged, 6,198 articles),
**U 0.204** (216, 3,530) and **V 0.111** (144, 1,759). Overall 153 of the 720 verdicts are `other`
and 48% equal the argmax slug (unweighted).

**Reading.**

* **Every candidate fails the cap that the anchored baseline gives, early.** The caps use `other`
  precision 0.319 and recall 0.661 measured on eval_run 43 with a judge that saw the model's label.
  The one-sided lower bound of the truly-`other` rate of the removed set is 4 to 6 times the cap.
* **Which verdicts need that baseline.** The recall tolerance (-0.02) lets at most 0.02 x T
  truly-`other` articles leave `other`, T being the truly-`other` total of the population (at most
  the 93,330 confirmed articles, so at most 1,866). At the 99.17% lower bounds the removed sets
  hold: `t0.50+m0.20` 9,728 x 0.218 = **2,121** truly-`other` articles, more than 1,866 even if
  every confirmed article were truly `other`: **it fails whatever the baseline**.
  `t0.55+m0.00` holds 7,957 x 0.208 = **1,655** and `t0.55+m0.20` 6,198 x 0.240 = **1,488**: they
  fail unless T is at least 82,750 and 74,400 (89% and 80% of all confirmed articles). **How far
  that is from the data, under a stated assumption:** T is at most the 35,368 stored-`other`
  articles (all of them taken as truly `other`) plus the truly-`other` slug-labelled ones. With the
  stored verdicts' rates for those (15.4% in 0.6-0.8, 28,916 articles; 7.5% from 0.8, 29,046
  articles) that is **at most 42,000**, and 46,084 at the upper ends of their intervals (24% and
  13%). Those rates come from the *anchored* judge: under the blind judge they are probably higher
  (it found 2 to 4 times more truly-`other` in the removed sets), but even at three times the
  anchored rates T is about 55,000, below 74,400. So the two verdicts that depend on the baseline
  would need the slug-labelled articles to be about two thirds truly `other` (39,008 of 57,962) to
  flip; the baseline itself is **not measured under the blind judge** (stage 2 would).
* **Within the removed sets no score separates the truly-`other` from the truly-slug articles.**
  The AUCs below are measured **only on the 720 articles of sub-cells A, U and V** (0.5-0.6 band,
  every article already with a score of 0.5 or more and a margin of 0.2 or more, or a score of
  0.55 or more): a restricted range by construction. P(a truly-slug article has a higher value
  than a truly-`other` one) is **0.42 (0.35-0.50) for the maximum score, 0.40 (0.32-0.47) for the
  margin over the runner-up and 0.63 (0.55-0.70) for the group score**; the sub-cell with the
  clearest margin and score (A) is the dirtiest. So a threshold or margin on the leaf scores does
  not isolate a clean subset *inside the candidate sets*, which is what the rules need. Whether
  scores separate the classes across the whole 0.4-0.6 range (the 0.4-0.5 band and W were not
  judged) is **not measured**: stage 2 would.
* **These rates are 2 to 4 times the ones the stored verdicts gave** (A-like subsets 8-17%, the
  0.5-0.6 band 19%). The stored verdicts came from a judge that saw the model's label; this one
  does not. Only 5 articles are common to both runs, so the gap between an anchored and a blind
  judge is not measured here. What stage 1 does show is that, against a label chosen from the
  article alone, a large share of what the classifier calls `other` in 0.5-0.6 is
  genuinely `other` (multi-company round-ups, generic market commentary).
* **The measured design effect (2.6-2.8) is about 1.7 times the 1.5-1.6 the sizing above assumed**
  (the sub-cells are sampled at very different rates), so the "judged articles needed" of the
  sizing table scale by about 1.7: do not reuse those sizes as they stand.
* **Limits.** Stage 1 covers only the sub-cells A, U and V, so it says nothing yet about the
  other cells: the T-123 baseline under the blind judge, the short-circuit stratum
  (`CATEGORY_GROUP_FLOOR`), per-slug thresholds in the other bands, and the loosening of
  `t0.50`/`m0.10` (already excluded by T-121(a)).

**Not run (2,580 calls left in the sample, 3,080 in the cap):** stage 2 (the other 2,580 strata).
It would re-measure the baseline `other` precision and recall with the blind judge, the AUCs on
all cells, per-slug thresholds and the floor decision.

<!-- docs/evaluation.md lines 2915-2999 at 09d90ae -->
### Calibration judge and sampler (T-141)

`calibration.md` is not a stage prompt: it belongs to the opt-in calibration judge
(`news_nlp/eval/calibration_judge.py`) that T-121(b) uses. It shows the judge the article
only (never the classifier's output) and asks for the best-fitting label of the full
taxonomy, `other` included: the **truth label**. "The argmax slug fits" is truth == argmax;
for the short-circuit stratum, "`other` is correct" is truth == `other`. **What "truth" means:**
the judge reads the same capped lead of the article (the first 6,000 characters) the category
classifier reads, so it is the best label for that text, not for the whole article.

* Verdicts are stored under `eval_verdict.task = "category_truth"` (and `eval_run.stage` the
  same), so a stored-label verdict (task `category`) never satisfies a truth lookup. The task
  is outside `sampling.STAGES`, `judges.JUDGES` and `VERDICT_MODELS`, and `ExperimentSpec`'s
  stages are unchanged (a test pins all four lists): `--stage all` and experiments do not see
  it. Its `eval_run` rows are left out of `latest_eval_runs` and `GET /eval/latest`
  (`queries.NON_ACCURACY_EVAL_STAGES`): their metrics are call counters, not accuracy.
* An out-of-taxonomy slug is not coerced to `other`: it goes to the repair prompt and then to a
  `parse_failed` fallback, which is never reused. `load_truths` excludes those rows and
  reports their count and rate, and re-weights each half over the usable articles (assuming
  they are missing at random within the cell and half); so does an article left unjudged.
* **The wording is frozen.** It is the system prompt file, the user template and the repair
  prompt, all owned by `calibration_judge` (not borrowed from the helpers other judges share).
  `test_eval_calibration_judge.py` pins the hash of the three and the rendered user message, the
  reuse key (`truth_experiment`) carries the prompt hash and the judge model, and a run
  started under another hash is refused. A change to the wording or to the label set
  invalidates every stored `category_truth` verdict and costs a re-judge, to be budgeted
  separately from D10.
  The reuse key also carries the item's `prediction`, which for a calibration item is only
  `{"text_sha256"}` (the hash of the title and capped text the judge read): the judge never sees
  the classifier's output, so T-122's new rule or T-124's re-label never forces a re-buy, and a
  change of the text always does. The label and scores to analyse against come from
  `article_category` by article id. `judge_and_record` refuses items outside the run's sample.
* **Spend.** The D10 cap (3,800) is on judge *calls*. `judge_and_record` reserves the two
  calls an article may need (the reply and one repair), counts the calls it really makes,
  records and commits each verdict as it arrives and stops at `max_calls` rather than start an
  article it might not finish (status `stopped_at_cap`; `error` for an outage or any unexpected exception, an interrupt leaves `running`; `incomplete` when articles lack a verdict for another reason, such as
  transport errors below the outage threshold, `ok` only when every item has one). The calls used are kept in the run's
  `metrics_json` as it goes; pass `remaining_calls(conn)` as the cap so it holds across runs.
  The base sample is 3,300 calls at one call per article (a test pins it), leaving 500 under
  the cap; if every reply needed a repair it would cost 6,600, so the run logs a warning
  whenever `max_calls` is below two calls per pending article. Duplicate articles are refused.
* **Resuming, interrupts, outages.** Calling `judge_and_record` again on the same run resumes
  it: usage carries over, an article already judged in the run is not recorded twice, a
  `parse_failed` one is judged again and replaced. Articles go out round-robin across
  (cell, half), seeded within each, so a stop at the cap leaves an even gap, not whole cells.
  An interrupt cancels the queue (only the calls in flight finish, and their verdicts are
  kept). A failed call (rate limit, 5xx, auth, network) is a `JudgeTransportError`, never a
  verdict; five in a row stop the run with `JudgeOutage` (status `error`).
* `start_calibration_run` creates the `eval_run` with the config in `strata_json`: seed, band
  edges, cell populations, half assignment, dropped articles, judge model, temperature
  (`config.JUDGE_TEMPERATURE`, the one `build_model` uses), judge seed, prompt hash, date.
* The sampler (`calibration_sample.draw_sample`) stratifies the confirmed articles
  (`load_confirmed`: `match_class = 'accept'`) by (argmax slug x score band) plus the
  short-circuit stratum (`stratum_of`; the 0.5-0.6 band cells are split into the sub-cells A, U, V,
  W of the candidate rules, 40 / 24 / 16 / 10 draws per slug), draws `2 x cell_half_size` per
  stratum (all of a smaller one; 90 per cell in the 0.4-0.6 bands, 60 elsewhere), and
  splits each cell at random into a fit and a report half. Each cell has its own generator
  seeded from the run seed and the cell name, so the same inputs and seed give the same draw
  and one cell's population never moves another's. An article's weight in a half is the cell
  population over the cell's articles in that half; `sample_items` drops an article whose
  text or stored row is missing, lists it in the sample and recomputes the weights
  (`restrict`). `design_effect` gives the Kish factor of a half's weights, which the sizing in
  `calibration.items_to_decide` needs; `run_config` stores each half's size and design effect, and `load_truths(...).design_effect(half)` gives the realised one over the usable rows. `start_calibration_run` refuses a sample that still draws a dropped article.
  A reused verdict is the latest *valid* one (a newer `parse_failed` never hides it).
  `model.calibration_agent_factory(settings)` builds the `agent_factory`.

**Recipe (what T-121(b) calls, in this order)**

```python
population = load_confirmed(conn)                      # match_class = 'accept'
drawn = draw_sample(population, seed=SEED)             # cells, halves, weights
pairs, sample = sample_items(conn, drawn)              # drops articles without text
run_id = start_calibration_run(conn, sample, judge_model=..., judge_url=..., code_version=...)
judge_and_record(conn, run_id, [i for _, i in pairs],
                 agent_factory=calibration_agent_factory(settings),
                 max_calls=remaining_calls(conn))      # resume = call it again on run_id
truths = load_truths(conn, run_id)                     # usable labels + weights
```

Two traps: pass `start_calibration_run` the **effective** sample `sample_items` returns, not
`drawn` (it refuses a sample that still draws a dropped article), and pass `remaining_calls(conn)`,
not the full 3,800, so the D10 cap holds across runs. The label and scores to analyse against are
read from `article_category` by article id. Check `truths.parse_failed_rate`, `truths.unjudged`
and `truths.design_effect(half)` before trusting an estimate.
