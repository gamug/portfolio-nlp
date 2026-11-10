# Sentiment training: the experiments after the pair trainer (PR #154)

**Status: written 2026-11-10.** What was tried to improve the sentiment "pair" model after PR #154's grid, in
the order it was tried, with the figures. Every run is in MLflow (`mlruns/`, git-ignored, local) under the
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

**The model on all rows.** `train_sentiment_pair_2026_11_09.py --final-all` (1,150 examples: 809 summaries and 341
replay, grid setting, seed 4; epoch losses 0.820 and 0.394), written to `models/sentiment_pair_all/final` (local, git-ignored, not published) and
**recorded** in MLflow experiment `sentiment_pair_final_all` (run `final_all__e2_lr3e-05_s4`: settings, 809 + 341 examples, the
path, the losses). An earlier run of the same command, made before the mode logged, gave the same losses to three
decimals but **different weights** (the training is not bit-reproducible on the GPU); it is kept, unrecorded, as
`models/sentiment_pair_all_superseded/final_all_unrecorded` and is not the model of this page.
The CV script made an earlier model the same way (losses 0.866 and 0.381) before `--final-all` existed (kept locally as
`models/sentiment_pair_all_superseded/final_cv_script`, not to be used; its MLflow run `final_all_data__s4` is tagged
`superseded`): the two are
**not identical** (the examples are in a different order, so the shuffles differ), and their predictions were not
compared. `--final-all` is the one implementation now; the CV script no longer trains a final model. The estimate of
either is the CV figures; neither has a held-out score.

## Error analysis of the neutral misses

`sentiment_neutral_errors_2026_11_10.py`. The neutral texts the model calls positive or negative are mostly
"the company is not the subject": an analyst or broker action, the company as a comparator or an incidental
mention, or a wrong tag. The sentiment of the text is real; its relevance to the company is not.

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

Paired gain over the sentiment model alone (95% bootstrap interval): gate score +0.052 [+0.038, +0.065],
neutral recall +0.312; stack score +0.077 [+0.058, +0.095], neutral precision +0.153, macro ROC OVR
+0.053 [+0.037, +0.068]; negative recall unchanged. Negative precision and recall are above the D32 bar
(70%) for the stack.

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

The model on all rows: `train_sentiment_pair_2026_11_09.py --final-all` (every file of `MARK_SETS`, no validation
slice, grid setting, seed 4; written to `models/sentiment_pair_all/final`, not published). The grid and `--final`
still read only the first file, so the PR's recorded grid stays reproducible.

`sentiment_cv_all` writes the out-of-fold predictions the later scripts read (`data/sentiment_pair_cv_all/predictions/`);
`sentiment_relevance_cv` needs them and `config/pipeline_models.json` (the NLI model); it saves its out-of-fold
relevance scores to `data/sentiment_pair_relevance_cv/oof_relevance.json`, and the environment variables of the
scripts: `REPLAY_N=<n>` (`sentiment_cv_all`: n replay sentences, outputs named `..._replay<n>`), `CONTROL=1`
(`sentiment_relevance_cv`: reuse the saved scores, shuffle them once, rerun the nested procedure: the control above)
and `CONTROL_SEEDS=<n>` (the same with n shuffles; it implies `CONTROL=1` and stops with a message if the saved scores
are missing). The retraining of the relevance model reproduced the recorded figures exactly. Run order:
`company_gate_split`, `company_flag`, `learning_curve`, `neutral_threshold`, `draw_sentiment_train_2` and
`migrate_sentiment_train_2_to_gold` (the new tags), `more_data`, `cv_all`, `cv_threshold`, `neutral_errors`,
`cv_cues`, `zeroshot_relevance`, `relevance_cv`.
