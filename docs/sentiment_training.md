# Sentiment training: the experiments after the pair trainer (PR #154)

**Status: written 2026-11-10.** What was tried to improve the sentiment "pair" model after PR #154's grid, in
the order it was tried, with the figures. **It opens with the history** (how the tags were made, the databases, the
assumptions, the metrics, and the story in seven steps); the sections after it are the detail. Every run is in MLflow (`mlruns/`, git-ignored, local) under the
experiment named in each section. The scripts that produced the figures are `scripts/*_2026_11_10.py`
(exploratory, run once; they carry a file-level `ruff: noqa`). Tests cover the gold file (counts, slices, no shared key or
body), `merge_marked`, `build_all_data`, the `--final-all` and `--models` options, and that the headline figures of this
page are the recorded runs of the export below; they do **not** cover the 13 experiment scripts themselves. The MLflow store is local and git-ignored: the metrics of every cited run are exported to
[sentiment_training_results.csv](sentiment_training_results.csv)
(`scripts/export_sentiment_training_results_2026_11_10.py`); the relevance experiment was run twice on 2026-11-10 (the
rerun reproduced the first figures exactly), so each of its runs appears twice there, with its own start time; the `recorded` column is `yes` for the earliest run of each
name, the one this page quotes.
Nothing here is pinned or published: the
pin is decided only on the 180-text gold (T-148c4) and only with the owner's word.

## The history: how we arrived at the final model

This section tells the story in order, with the databases, the assumptions and the metrics of each step. The sections
after it are the detail and the figures; the numbers here are the ones they give. "The final model" is, at the end of
the story, **the sentiment pair model trained on every tagged row, and, on top of it, a relevance model combined with it
(a "stack")**; the first is built and recorded, the second is proposed and cross-validated, and nothing is pinned.

### How the tags were made (read this before any figure)

What the owner tagged is **not the sentiment of the sentences. It is what a text says about one company.** Every row is
a text (a summary of an article, a few sentences) **and a company** (`company`, `ticker`); the mark is the price impact
an investor would read on **that company** (owner's rule, 2026-11-08, `docs/gold_datasets.md` set 6):

- If the company is in the facts the text presents, the mark is the sentiment about it: positive, negative or neutral.
- If the text gives **no fact about the company** (the company is absent, only mentioned in passing, or the facts are about
  another company or the market), the mark is **neutral**, even when the text is strongly positive or negative about
  something else. A positive sentence about another company does not make the text positive for this row.
- The title is context, not ownership: it helps to know whether a text speaks about the company, but a sentence is the
  company's because of what it says, not because the headline names the company.
- Rows with no company (sector, market and economy texts, 26 + 20 of them) are tagged for that scope; the pair model
  cannot read them and leaves them out.

So "neutral" is **a mixture of two different things**: (a) the text states facts about the company that have no
direction for its price, and (b) the text states no fact about the company. A model has to learn *relevance to the
company* as well as *polarity*, and the weak class of this story (neutral) is where the two meet. Two limits of the
record: the rule was given on 2026-11-08, after the first 450 marks were in, and whether it was followed on every row is
not recorded (the owner's notes show the same practice, for instance "the company appears but the real fact is not
about the company"; 37 of the 405 later rows carry a note, several of this kind); and no different instruction was recorded for the 405
later tags, which are read as made under the same rule. What the earlier sentence labels are (below) is **not** this.

### The databases

| Set | Where | What it is | Size | Labelled by and how | Used for |
|---|---|---|---|---|---|
| Sentence pool (silver) | `data/sentiment_finetune/`, git-ignored | sentences from the project's own articles | 5,800 merged, 3,865 after balancing (1,321 negative / 1,321 neutral / 1,223 positive) | an LLM (DeepSeek), one sentence at a time, price-impact view, **not tied to a company**; silver, not human | trained the pinned model; the **replay** of every pair run |
| 180-text gold | `gold/sentiment_gold.csv` | the pipeline's own texts: summaries and company texts (120) and extractor texts (60) | 180 | the owner, company-anchored | the exam of D32 and the pin; never tuned on; **not scored in this story** (T-148c4) |
| Training set 1 | `gold/sentiment_train.csv` | summaries, 450 in 3 batches of 150, **150 per class the pinned model predicted** | 450 (232 positive / 113 negative / 105 neutral); 424 read as a pair | the owner, company-anchored | train (341) and validation (83) |
| Training set 2 | `gold/sentiment_train_2.csv` | summaries, the same design, 135 per predicted class | 405 (178 / 116 / 111); 385 read as a pair | the owner, company-anchored | all of it, by cross-validation (the 270/135 slice split is the assistant's, T-148c3f) |
| Replay as used | derived | the pool's sentences that **name their article's company**, read as pair with that company | 925 eligible of 3,865; 341 drawn | the LLM labels of the pool | mixed into every training set |

The draws balance the classes the *pinned model predicted*, not the owner's classes (the owner never saw the
prediction), so the owner's classes are not balanced and the figures need weights (below). No text of the gold or of the
held-out set, by body hash or headline, is in any training set or in the replay.

### The assumptions

1. **The company anchor can be learned from the pair `company + text`.** The pinned model never saw the company; the pair
   gives it the anchor of the tagging rule. (Tested only together with the fine-tuning: the trained pair model beats the pinned text-only model on the same texts,
   0.737 against 0.648. The pinned model read as a pair, **without** training, agreed no better than read as text, 61.3%
   against 60.9% on the validation slice, so the pair input alone does nothing.)
2. **A labelled sentence that names the company is a fact about it.** The replay keeps the pool's sentences that name
   the company (925 of 3,865: 2,915 do not name it) and reads them for that company. The labels are the LLM's
   sentence-level reading, not the anchored rule, so a sentence that names company A and states a fact about B is
   mislabelled for A. **Not tested.**
3. **Replay keeps the sentence-level knowledge** while the model learns summaries (341 sentences: half of the grid's
   examples, 341 of 682, and 30% of the model on all rows, 341 of 1,150). A larger replay (809) was tested and was not better (0.723 against 0.737).
4. **"The company is not named, so neutral" is the rule's practical consequence.** Tested as a hard gate, a flag, a
   threshold and a set of cues: it does not help, because three quarters of the neutral texts **do** name the company
   (151 of 202).
5. **The tags of two draws, by two prediction-balanced designs, can be pooled** with weights per draw and stratum.
6. **Cross-validation over the tags stands in for a held-out score** when there are about 800 rows (the owner's call,
   "crossvalidation is better given the lack of data"). It is an estimate of the recipe, not of one model, and it is
   not an exam: the 180-text gold is.
7. **Relevance ("is the text about this company") is what neutral lacks.** Read from the neutral misses (analyst or
   broker actions, the company as a comparator or an incidental mention, tags that look wrong); a reading of 30 texts,
   not a count. The relevance model's labels are derived from the tags (a direction means relevant, neutral means not),
   so they carry the same mixture.

### The metrics

- **Agreement (accuracy)** was the headline of D32: 80% on the 180-text gold, and negative precision and recall of at
  least 70%. It is kept as a description.
- **From 2026-11-09 (D37, owner): precision, recall and ROC OVR per class**, nine figures; the **score** (selection) is
  their mean. ROC OVR is the AUC of one class against the other two, from the model's probability of the class.
- **Weighted.** The pool is not the sample: each text weighs *pool size of its predicted class / texts drawn from it*, per
  draw (first draw pools 1,878 negative / 1,582 neutral / 2,540 positive, second 1,895 / 1,606 / 2,499). The figures
  estimate the pool the draw came from.
- **Intervals and gains:** bootstrap within strata (400 resamples in the cross-validation and the stack); a gain is **paired** (the same texts under two models)
  and read as `better`, `not better` or `not decided` by its interval.
- **Three evaluation protocols**, which are **not comparable with each other**: the 83 validation texts of training set 1
  (a seed moves the score by 0.02 to 0.04), the 809-row 5-fold cross-validation twice (out-of-fold), and the nested
  version of it for any rule fitted on predictions (gate, stack).

### The story

**1. The starting point.** The pinned `gamug/FinBERT-financial-news` is FinBERT fine-tuned on the silver sentence pool,
and it scores chunks of a text and weighs them by the entity. On the 180-text gold, the owner's company-anchored marks,
it agreed 60.4% [52.8, 67.6] against the 80% bar (D32). The recipe's condition to fine-tune on summaries was met (D26,
D35). *Data: pool, 180 gold. Assumption: none yet. Metric: agreement.*

**2. The owner tags 450 summaries; PR #154 builds the trainer.** The recipe (D35) fixed before any mark was read:
the pair `company + summary`, from the pinned model, the 341 replay sentences, a grid of 27 runs (epochs 2, 3, 4;
learning rate 1e-5, 2e-5, 3e-5; 3 seeds) chosen on the validation slice with a tie rule. It chose 2 epochs and 3e-5
(score 0.646 to 0.668 over the grid), and the final run (seed 4) scored **0.678 [0.580, 0.768]** on the 83 validation
texts: neutral recall only 0.20 to 0.44 across the seeds, negative precision under the bar (46%), the paired gain of the
score over the pinned model `not decided` (only negative recall was `better`). *Data: set 1 and the replay. Assumptions 1, 2, 3. Metric: D37 on 83 texts.* The model was not
good enough on the class the owner cared about, and the 83 texts could not say why.

**3. Is it the company? Four cheap tests on the 83 texts.** The owner suspected the model did not give the company its
weight, and proposed "company not named, so neutral". Tested as a hard gate (neutral recall 0.27 to 0.52, but neutral
precision 0.47 to 0.38, score 0.678 to 0.664), as a flag in the input (0.673 against 0.671), as weights on the
neutral and negative probabilities (0.699 against 0.671, not decided in 3 of 4 seeds), and with more data (the
learning curve: +0.005 per quarter of the rows). *Assumption 4. Metric: score on 83 texts.* None fixed neutral, and
the matcher of "named" was strict, so these were weak tests.

**4. More tags, and cross-validation.** The owner tagged 405 more summaries (second draw, same design, same rule): 855
rows, 809 read as a pair. A validation slice of 83 or even 135 texts is too small to judge a recipe, so the recipe is
judged by **5-fold cross-validation over the 809 rows**, twice, out-of-fold. The recipe scored **0.737** against 0.648
for the pinned model on the same texts (paired gain +0.089), with neutral precision 0.634 and **neutral recall still
0.442**. *Data: sets 1 and 2. Assumptions 5, 6. Metric: weighted D37 on 809 texts.*

**5. Looking at what neutral is.** The same tests on the larger evaluation (a mention flag at two levels, a neutral
prior fitted on out-of-fold predictions) gave nothing (score 0.737 to 0.740; 0.74), and the error analysis counted
where the company is named: of the 202 neutral texts, 151 name it in the summary, and the model recalls 56% of
those and 11% of the ones named only in the body. So **the miss is not "the company is absent"**: the model hears a
company and a polarity and does not tell whether the text is *about* the company. *Assumption 7.*

**6. A relevance model.** A zero-shot reading of the category stage's NLI model ("this text is mainly about
<company>") already ranks relevant above neutral texts (AUC 0.743, the best of nine variants; the sentiment model's own
ranking is 0.795). Fine-tuned on the tags the same way (relevant = the owner marked a direction), with the same
folds, its out-of-fold AUC is **0.90**. Combined with the sentiment model, nested: a **gate** (neutral when relevance is
under a threshold) scores **0.789** and a **stack** (logistic regression on the sentiment log-probabilities and the
relevance) **0.814**, with neutral precision 0.787 and recall 0.686 and negative precision and recall near 0.75. A
shuffled-relevance control (20 shuffles) adds at most +0.005 to the stack, so the gain is the relevance information's.
*Metric: nested weighted D37 on 809 texts.*

**7. Where it stops.** The sentiment pair model on every row (`--final-all`, one run, seed 4) is the model to
propose; the stack is not built (no final relevance model, nothing in `sentiment_stage`), its evaluation is not
strictly nested, and **no figure of this story is held-out**. The pin is decided only on the 180-text gold with the
owner's word (T-148c4), by the rule of D35 (3), (4), (7). The amendment of D35 to train on all rows is for the owner
to accept ("Amendment to D35", below); **once it is accepted, T-148c3g trains the model on the whole dataset** (see there).

## The recipe and the metric (unchanged)

Pair input `[CLS] company [SEP] summary [SEP]`, base `gamug/FinBERT-financial-news`, 2 epochs, lr 3e-5,
341 replay sentences in every training set (they exclude every marked body). The metric is D37: per-class
one-vs-rest precision, recall and ROC OVR, **weighted** (each text weighs pool size / drawn count of its
draw and predicted-class stratum), the selection score is the mean of the nine. Intervals are bootstrap
within strata; gains are paired on the same texts.

## The problem

On the 83 validation texts the pair model scored 0.65-0.70 and neutral recall was 0.25-0.44. The owner's
reading: the model does not give the company's presence the weight it should, and most of the neutral
news is "the company is not what the news is about".

## What did not work

**The rows are not comparable with each other**: the first five were judged on validation texts (the 83 of the first file, and for `more_data` also the 135 of the second) (small, noisy slices: a seed moves the score by 0.02-0.04), the last two on the 809-row cross-validation. Read each result against its own baseline in the third column.

| Experiment (MLflow) | What | Evaluated on; compared with | Result |
|---|---|---|---|
| `sentiment_pair_company_gate` | Hard rule "company not named in the summary -> neutral", on the 83 validation texts | 83 validation texts (450-row recipe); the same model without the gate | Neutral recall 0.27 -> 0.52, neutral precision 0.47 -> 0.38, score 0.678 -> 0.664 (final model). Not better. The matcher is strict (`_text_mentions_subject` misses aliases), so the groups are noisy |
| `sentiment_pair_company_yes_no` | Segment A says "company named in text / not named" so the model can learn the prior | 83 validation texts, 4 seeds; the grid's 0.671 (no flag) | Seeds 1-4 mean 0.673 against 0.671 without the flag. No effect |
| `sentiment_pair_neutral_threshold` | Weights on the neutral/negative probabilities (`argmax(p*w)`), fitted on the 83, out-of-fold | 83 validation texts, out-of-fold on the 83, 4 seeds; the same models without the weights (0.671) | Mean 0.699 against 0.671, paired gain not decided in 3 of 4 seeds |
| `sentiment_pair_learning_curve` | 25/50/75/100% of the training rows | 83 validation texts, 3 seeds per fraction; 100% of the 367 rows = 0.668 | 0.649 / 0.657 / 0.663 / 0.668: more data helps a little (about +0.005 per 25%) |
| `sentiment_pair_more_data` | Add the new 270 rows (old 83 / new 135 validation) | the old 83 and the new 135 validation texts, 4 seeds; trained on the first file only (0.671 / 0.680) | Old validation 0.671 -> 0.695, new 0.680 -> 0.690 |
| `sentiment_pair_company_cues_cv` | A mention flag (V1 yes/no, V2 levels, alias-aware via `triage.CompanyResolver`) in the pair, cross-validated | 809-row cross-validation, out-of-fold; the no-cue CV model (0.737) | Score 0.737 / 0.737 / 0.740 (none / V1 / V2), paired gain +0.000 [-0.015, +0.011]; neutral precision up 0.05, neutral recall down 0.06 |
| `sentiment_pair_cv_threshold` | The neutral prior on the cross-validated predictions, nested | 809-row cross-validation, nested; the no-cue CV model (0.737) | Nested score 0.74 against 0.737; neutral recall 0.53 but negative recall 0.72 and precision 0.71. Not useful |

## More data: 405 new tags and cross-validation

`data/sentiment_eval_to_tag.csv` (405 rows; the owner tagged 178 positive / 116 negative / 111 neutral) was
migrated to `gold/sentiment_train_2.csv` (slice train blocks 1-18 = 270, validation blocks 19-27 = 135,
batch "2"; scripts `draw_sentiment_train_2_2026_11_10.py`, `migrate_sentiment_train_2_to_gold_2026_11_10.py`; the
working files keep the draw's first name, `data/sentiment_eval_*`, but are training data). **The 270/135 slice
split was chosen by the assistant, not by the owner.** The draw balances the *predicted* class (135 each), so
the owner's classes are not balanced (second file 178/116/111, first file 232/113/105).
450 + 405 = 855 rows are tagged; 46 (26 + 20) are sector, market and economy tags with no company,
which the pair model cannot read (`reads_as_pair`) and which stay with the text-only model, so the CV has 809.
With about 800 labelled rows a fixed validation slice is too small, so the recipe is judged by **5-fold
cross-validation over the 809 rows read as a pair** (2 repeats, folds stratified by (draw, stratum), seeds
20261109+repeat, the same folds in every later experiment; out-of-fold predictions only).

`sentiment_pair_cv_all_data` against the pinned text-only model on the same texts:

| | CV | pinned | paired gain |
|---|---|---|---|
| score | 0.737 [0.716, 0.764] | 0.648 | +0.089 [+0.068, +0.114] |
| macro precision | 0.692 | 0.576 | +0.116 |
| macro recall | 0.674 | 0.597 | +0.077 |
| macro ROC OVR | 0.845 | 0.771 | +0.074 |
| neutral precision / recall | 0.634 / 0.442 | 0.388 / 0.430 | +0.246 / +0.012 [-0.038, +0.072] |
| negative precision / recall | 0.680 / 0.745 | 0.597 / 0.748 | +0.083 / -0.003 |
| positive precision / recall | 0.761 / 0.834 | 0.743 / 0.613 | +0.018 / +0.221 |

Neutral recall stays at 0.44.

**Replay share.** The grid mixed the 341 replay sentences with 341 summaries (50%); with 809 summaries the same
341 sentences are about 30% of the mix, and that is what the CV above used. The 50% mix was run too
(`REPLAY_N=809`, `sentiment_pair_cv_all_data_replay809`, 809 replay sentences from the 925 eligible): score 0.723
[0.699, 0.748] against 0.737 [0.716, 0.764] (the intervals overlap; not a paired comparison), neutral recall 0.401
against 0.442, positive recall 0.825 against 0.834. A bigger replay is not better, so the 341 stays.

**The model on all rows** (the one explanation of it; the amendment below and `Reproduce` point here).
`train_sentiment_pair_2026_11_09.py --final-all` trains on every row of both gold files read as a pair (809
summaries) and the 341 replay sentences (1,150 examples), grid setting, seed 4, no validation slice; epoch losses
0.820 and 0.394. It writes `models/sentiment_pair_all/final` (local, git-ignored, not published) and **records** the run
in MLflow experiment `sentiment_pair_final_all` (`final_all__e2_lr3e-05_s4`: settings, counts, path, losses). Its
estimate is the CV figures above; it has no held-out score. GPU training is not bit-reproducible: an earlier run of the
same command (before the mode logged) gave the same losses to three decimals but different weights, kept unrecorded as
`models/sentiment_pair_all_superseded/final_all_unrecorded`; the CV script's earlier model (losses 0.866 and 0.381,
different example order, run `final_all_data__s4` tagged `superseded`) is `.../final_cv_script`. Neither is the model of
this page and none of the three was compared with another.

## Error analysis of the neutral misses

`sentiment_neutral_errors_2026_11_10.py`, on the out-of-fold predictions of the cross-validation (raw counts, not
weighted; the model's reading is the mean of the 2 repeats). Of the 809 texts, the owner marked **202 neutral**; the
model recalled **101 (50%)**, and the 101 misses went to positive (62) and negative (39). Of the 150 texts it
called neutral, 101 are neutral for the owner, 31 positive and 18 negative. The owner left a note on 63 of the
202 neutral rows (22 of them recalled).

Neutral recall by where the company is named, by the strict matcher `sentiment_stage._text_mentions_subject` (four
groups that do not overlap and add up to the 809 texts):

| the company is named in | texts | neutral | share neutral | neutral recalled |
|---|---|---|---|---|
| the summary | 696 | 151 | 22% | 85 (56%) |
| not the summary, but the title | 56 | 25 | 45% | 12 (48%) |
| neither, but the body | 35 | 19 | 54% | 2 (11%) |
| none of the three | 22 | 7 | 32% | 2 (29%) |

Three quarters of the neutral texts (151 of 202) **do** name the company in the summary, so "the company is not
named" explains about a quarter of them. That is why the rule "not named, therefore neutral" did not work (the first row of "What did not work"). The matcher is strict and misses aliases, so the last two groups are upper bounds of "not named".

**What the misses look like is a reading, not a count.** The 30 missed neutral texts the script prints (not kept:
they are news text) were read by the developer: most are an analyst or broker action, the company as a comparator or
an incidental mention, or a tag that looks wrong. None of those categories was counted, and the relevance model
below is built on that reading ("is the news about the company"), which the page never measured apart from the
fact that the model does separate the two (the AUC of the relevance model).

## A relevance model ("does this news belong to this company")

Same data and the same labels: relevant = the owner marked a direction, not relevant = neutral. Hypothesis
`This text is mainly about <company>.` (premise = the summary).

1. **Zero-shot** (`company_relevance_zeroshot`, the category stage's NLI model
   `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, pinned revision): weighted AUC 0.743 alone (**the best of nine variants**: three hypotheses x three premises, summary / title + body / title + summary, 0.618 to 0.743, all nine in the export; the wording used later was chosen with this AUC in sight, on the same 809 texts) against 0.795 for
   the sentiment model's own `1 - p_neutral`; 0.84 when the two are rank-averaged.
2. **Fine-tuned** (`sentiment_pair_relevance_gate`, the same 5-fold x 2 splits, 3 epochs, lr 2e-5, batch 16,
   **float32**; the checkpoint loads as fp16 in transformers 5 and the optimizer then gives NaN): out-of-fold
   weighted AUC **0.902 / 0.890** (mean 0.896).
3. **Combined with the sentiment model**, evaluated nested (the rule fitted on 4/5 of the out-of-fold rows,
   applied to the other 1/5; 10 outer repeats x 2 CV repeats; weighted):
   *gate* = `argmax(p)`, neutral if the relevance score is under a threshold (median 0.525);
   *stack* = logistic regression on `[log p(3 classes), logit(relevance)]`.

| | sentiment only | gate | stack |
|---|---|---|---|
| score | 0.737 | 0.789 | **0.814** |
| macro precision / recall | 0.692 / 0.674 | 0.757 / 0.764 | 0.780 / 0.763 |
| macro ROC OVR (the gate keeps the sentiment probabilities: 0.845 by construction, not a comparison) | 0.845 | 0.845 | **0.898** |
| neutral precision / recall | 0.634 / 0.442 | 0.680 / **0.754** | **0.787** / 0.686 |
| negative precision / recall | 0.680 / 0.745 | 0.756 / 0.734 | 0.747 / 0.748 |
| positive precision / recall | 0.761 / 0.834 | **0.835** / 0.804 | 0.807 / **0.855** |

Paired gain over the sentiment model alone (95% bootstrap interval, 400 resamples, the same number as the
cross-validation above): gate score +0.052 [+0.038, +0.065], neutral recall +0.312 [+0.253, +0.378], **negative
recall -0.011 [-0.026, -0.002]** (the gate costs a little negative recall; with 200 resamples the interval read
[-0.023, +0.000] and hid it); stack score +0.077 [+0.057, +0.095], neutral precision +0.153 [+0.090, +0.209], macro
ROC OVR +0.053 [+0.038, +0.068], negative recall +0.003 [-0.018, +0.015] (unchanged). Negative precision and recall
are above the D32 bar (70%) for the stack, on the cross-validation.

## Amendment to D35 proposed for the owner

(`PLAN.md` D35 points here; the owner has not accepted it.) 405 more marks (`gold/sentiment_train_2.csv`; 809 of the 855 read as a pair); the recipe is judged by 5-fold cross-validation over them (D37 figures: `docs/sentiment_training.md`), the model for the pin trains on all rows (`--final-all`, replay stays 341; a 809 replay scored worse); a relevance model plus stack (CV 0.814 against 0.737, not held-out) is proposed (T-148c3c/d); nothing is pinned (T-148c4); the 270/135 split of the new file is the assistant's (T-148c3f).

**Which D35 answers this amends, and which stand.** It amends **D35 (6)**, which the owner accepted: "the final run trains
on the 360 train rows plus the replay, a fourth fixed seed". Proposed instead: the candidate is **one run** of the grid
setting (2 epochs, 3e-5, seed 4, the 341 replay sentences) on **every row read as a pair of both gold files** (809 rows),
with no selection among runs. A model trained on every row has no validation figure of its own; the 809-row
cross-validation estimates the *recipe* (each CV model saw 4/5 of the rows), not this model. **D35 (3), (4) and (7)
stand unchanged**: they already apply to the published revision scored on the 180-text gold (T-148c4), not to the
validation slice; the validation score of the earlier final run (67.8%) was only a preview that the rule would not
have pinned. So the cross-validated figures are the reason to publish a candidate, not a pin criterion, and if the gold
gives less than the cross-validation (to be expected for a recipe tuned on these tags) the rule decides as written. The
relevance model and the stack are in the same position: they are scored on the gold by the same rule before they are
wired into `sentiment_stage` (T-148c3d).

**What follows when the owner accepts it: T-148c3g.** Train the model on the whole dataset with `--final-all` (the 809 rows of
both gold files and the 341 replay sentences, the grid setting, seed 4, **one run, no selection among runs**), then publish
it as the candidate and score it once on the 180-text gold (T-148c4). One run of that command already exists (MLflow
`final_all__e2_lr3e-05_s4`, unpublished); because GPU training is not bit-reproducible, the owner says whether that run
is the candidate or the command is repeated after the acceptance. Until the amendment is accepted the task is blocked and
the grid's `--final` run (450 rows, train slice only) stays the recorded D35 final run.

## Limits of these figures (read before using them)

- **Not strictly nested.** The base models are out-of-fold (a text is never scored by a model that saw it),
  but the gate threshold and the stack were fitted on out-of-fold predictions of 4/5 of the texts and scored
  on the other 1/5 with a random outer split, not the CV folds. Those base models saw some of the scored
  texts in training (the usual stacking leak). The rule has one threshold or four coefficients. **Control (2026-11-10):** the same
  nested procedure with the relevance scores shuffled across the texts (weighted AUC 0.50). One shuffle
  (`control_shuffled_nested_*`): stack score gain +0.001 [-0.005, +0.008], gate -0.039. **20 shuffles** (2 outer x 2 CV
  repeats each, `control_shuffled_20_seeds`): stack score gain mean +0.002 (sd 0.001, range [+0.000, +0.005]), neutral
  precision +0.034, neutral recall -0.013; gate score gain mean -0.041 (sd 0.005): with no signal the gate's
  threshold falls to its lowest value (0.05) and it turns a share of the true positives and negatives into
  neutral. So the way the rule is fitted adds at most about +0.005 to the stack and none to the gate; the
  real gain (+0.077 stack, +0.052 gate) comes from the relevance information. The control does not test whether
  the *base models* are optimistic for texts they were trained near (duplicates, same company); the strictly
  nested retraining of T-148c3c does.
- **Selection optimism.** About a dozen variants were judged on the same 809 rows (company gate, flag,
  neutral threshold twice, two cue sets, three zero-shot hypotheses, the fine-tuned relevance model, gate and
  stack), and the recipe was tuned on the same tags. The intervals are per comparison, not corrected for that.
  The only unbiased figure is the 180-text gold (T-148c4), not scored.
- **Near-duplicates across folds** (one article for two companies) were not checked. They would inflate the
  absolute level of both models, not the gain.
- **The relevance model** has one setting (3 epochs, lr 2e-5, one hypothesis wording) and one seed per fold; no
  seed spread and no alternative wording were measured. Its target is "the owner marked a direction", so it is
  built from the sentiment labels: it learns which texts the owner found neutral, which is the point, but it
  is not independent evidence about the sentiment.
- **Class imbalance** of the tags was not addressed (no class weights, no resampling): T-148c3e.
- All figures are cross-validated on tags the recipe was tuned on; **none is a held-out score**.
- Not done, with their TASKS entries: the final relevance model on the 809 rows and the strictly nested check
  of the stack (T-148c3c); the stack in `sentiment_stage`, its cost per article, what is stored when the two
  models disagree, and the score on the 180-text gold (T-148c3d, T-148c4); class weights (T-148c3e). Nothing
  is pinned or published.

## Reproduce

Run from the repository root with the eval group. Most scripts take the repository root and the MLflow store
as arguments (`<root> <mlruns>`); `sentiment_company_gate_split` takes only the store. Each docstring says
what it reads and writes:

```bash
uv run --group eval python -I -c "import sys; sys.path[:0]=['src']; sys.argv=['x','.','mlruns']; exec(open('scripts/sentiment_cv_all_2026_11_10.py').read())"
```

The model on all rows: `--final-all`, described in "More data" above (the grid and `--final` still read only the first
file, so the PR's recorded grid stays reproducible).

`sentiment_cv_all` writes the out-of-fold predictions the later scripts read (`data/sentiment_pair_cv_all/predictions/`);
`sentiment_relevance_cv` needs them and `config/pipeline_models.json` (the NLI model); it saves its out-of-fold
relevance scores to `data/sentiment_pair_relevance_cv/oof_relevance.json`, and the environment variables of the
scripts: `REPLAY_N=<n>` (`sentiment_cv_all`: n replay sentences, outputs named `..._replay<n>`), `CONTROL=1`
(`sentiment_relevance_cv`: reuse the saved scores, shuffle them once, rerun the nested procedure: the control above)
`CONTROL_SEEDS=<n>` (the same with n shuffles; it implies `CONTROL=1` and stops with a message if the saved scores
are missing) and `REUSE_SCORES=1` (the saved scores as they are: the nested evaluation and the bootstrap again, nothing
logged). The retraining of the relevance model reproduced the recorded figures exactly. Run order:
`company_gate_split`, `company_flag`, `learning_curve`, `neutral_threshold`, `draw_sentiment_train_2` and
`migrate_sentiment_train_2_to_gold` (the new tags), `more_data`, `cv_all`, `cv_threshold`, `neutral_errors`,
`cv_cues`, `zeroshot_relevance`, `relevance_cv`.
