# Sentiment training: the experiments after the pair trainer (PR #154)

**Status: written 2026-11-10.** What was tried to improve the sentiment "pair" model after PR #154's grid, in
the order it was tried, with the figures. Every run is in MLflow (`mlruns/`, git-ignored, local) under the
experiment named in each section. The scripts that produced the figures are `scripts/*_2026_11_10.py`
(exploratory, run once; they carry a file-level `ruff: noqa`, and no test covers them beyond the gold file's guard test). Nothing here is pinned or published: the
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

| Experiment (MLflow) | What | Result |
|---|---|---|
| `sentiment_pair_company_gate` | Hard rule "company not named in the summary -> neutral", on the 83 validation texts | Neutral recall 0.27 -> 0.52, neutral precision 0.47 -> 0.38, score 0.678 -> 0.664 (final model). Not better. The matcher is strict (`_text_mentions_subject` misses aliases), so the groups are noisy |
| `sentiment_pair_company_yes_no` | Segment A says "company named in text / not named" so the model can learn the prior | Seeds 1-4 mean 0.673 against 0.671 without the flag. No effect |
| `sentiment_pair_neutral_threshold` | Weights on the neutral/negative probabilities (`argmax(p*w)`), fitted on the 83, out-of-fold | Mean 0.699 against 0.671, paired gain not decided in 3 of 4 seeds |
| `sentiment_pair_learning_curve` | 25/50/75/100% of the training rows | 0.649 / 0.657 / 0.663 / 0.668: more data helps a little (about +0.005 per 25%) |
| `sentiment_pair_more_data` | Add the new 270 rows (old 83 / new 135 validation) | Old validation 0.671 -> 0.695, new 0.680 -> 0.690 |
| `sentiment_pair_company_cues_cv` | A mention flag (V1 yes/no, V2 levels, alias-aware via `triage.CompanyResolver`) in the pair, cross-validated | Score 0.737 / 0.737 / 0.740 (none / V1 / V2), paired gain +0.000 [-0.015, +0.011]; neutral precision up 0.05, neutral recall down 0.06 |
| `sentiment_pair_cv_threshold` | The neutral prior on the cross-validated predictions, nested | Nested score 0.74 against 0.737; neutral recall 0.53 but negative recall 0.72 and precision 0.71. Not useful |

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

Neutral recall stays at 0.44. The final model on all 809 rows (+341 replay, seed 4) is
`models/sentiment_pair_all/final` (local, git-ignored, not published; its estimate is the CV figures).

## Error analysis of the neutral misses

`sentiment_neutral_errors_2026_11_10.py`. The neutral texts the model calls positive or negative are mostly
"the company is not the subject": an analyst or broker action, the company as a comparator or an incidental
mention, or a wrong tag. The sentiment of the text is real; its relevance to the company is not.

## A relevance model ("does this news belong to this company")

Same data and the same labels: relevant = the owner marked a direction, not relevant = neutral. Hypothesis
`This text is mainly about <company>.` (premise = the summary).

1. **Zero-shot** (`company_relevance_zeroshot`, the category stage's NLI model
   `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, pinned revision): weighted AUC 0.743 alone against 0.795 for
   the sentiment model's own `1 - p_neutral`; 0.84 when the two are rank-averaged.
2. **Fine-tuned** (`sentiment_pair_relevance_gate`, the same 5-fold x 2 splits, 3 epochs, lr 2e-5, batch 16,
   **float32**; the checkpoint loads as fp16 in transformers 5 and the optimizer then gives NaN): out-of-fold
   weighted AUC **0.902 / 0.890**.
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

## Limits of these figures (read before using them)

- **Not strictly nested.** The base models are out-of-fold (a text is never scored by a model that saw it),
  but the gate threshold and the stack were fitted on out-of-fold predictions of 4/5 of the texts and scored
  on the other 1/5 with a random outer split, not the CV folds. Those base models saw some of the scored
  texts in training (the usual stacking leak). The rule has one threshold or four coefficients. **Control (2026-11-10):** the same
  nested procedure, with the relevance scores shuffled across the texts (weighted AUC 0.50), gives a stack
  score gain of +0.001 [-0.005, +0.008] and a gate gain of -0.039 [-0.052, -0.027] over the sentiment model
  alone (MLflow `sentiment_pair_relevance_gate`, runs `control_shuffled_nested_*`). So the procedure does not
  manufacture a gain from the way the rule is fitted; the gain of the real scores (+0.077 stack, +0.052 gate)
  comes from the relevance information. The control does not test whether the *base models* are optimistic
  for texts they were trained near (duplicates, same company); the strictly nested retraining of T-148c3c does.
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
relevance scores to `data/sentiment_pair_relevance_cv/oof_relevance.json`, and `CONTROL=1` reruns the nested
procedure on those scores shuffled (the control above; the retraining reproduced the recorded figures exactly). Run order:
`company_gate_split`, `company_flag`, `learning_curve`, `neutral_threshold`, `draw_sentiment_train_2` and
`migrate_sentiment_train_2_to_gold` (the new tags), `more_data`, `cv_all`, `cv_threshold`, `neutral_errors`,
`cv_cues`, `zeroshot_relevance`, `relevance_cv`.
