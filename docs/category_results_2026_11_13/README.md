# Record of the category runs of 2026-11-13 (T-174, provisional)

The figures quoted in `docs/category_step.md` and in PR #182 come from the files of this folder, so they can be checked from the repository.
No news text is in them: article id, company, the owner's label, the prediction and, for the held-out run, the scores.

| File | What it is | Rows |
|---|---|---|
| `baseline_train.csv` | the untouched model, label picked on the 225 train rows (`scripts/category_baseline_2026_11_13.py`) | 225 |
| `baseline_heldout.csv` | the same model on the held-out single rows, scored once | 135 |
| `oof_flat.csv`, `oof_mixed.csv` | out-of-fold predictions of the 5-fold CV of the flat and the mixed recipe (`scripts/category_train_cv_2026_11_13.py`) | 225 each |
| `heldout_mixed_scores.csv` | the mixed fine-tune trained on the 225 train rows, scored once on the held-out single rows: predictions and the entailment score of each of the 14 labels and the 6 groups (`scripts/category_heldout_2026_11_13.py`) | 135 |

The CV runs did not save their scores (the trainer does now: `data/category_groups/oof_<mode>_scores.npz`), so ROC OVR is given for the held-out run only:
`uv run scripts/category_roc_ovr_2026_11_13.py` (macro 0.798 over the 13 labels with held-out rows, 0.783 over the groups).

## How they were made

- **Data.** Train: `gold/category_gold_v2.csv` and its texts, the single bodies and the structured multi bodies split by company (`scripts/category_groups_mock_2026_11_13.py`):
  225 rows with a company label; the narrative multi bodies are left out, `uncertain` is not a class (1 row). Held-out: `gold/category_gold_v2_test.csv`, 135 of
  the 200 bodies are scorable (116 single bodies); the 44 narrative multi bodies need extractor texts and are not scored, so the D24 figure is partial.
- **Base model.** `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` at `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8`; premise `<company>: <text, 2,000 characters>`, hypothesis
  `The news about <company> is about <definition>.`, entailment probability.
- **Fine-tune.** 3 epochs, AdamW, learning rate 2e-5, weight decay 0.01, 10% warm-up then linear decay, 4 negative pairs per positive redrawn each epoch, batch 4 with
  4 accumulation steps, maximum length 512, fp16 autocast, seed 0 (fold `k` uses seed `k`); folds grouped by article. The mixed recipe adds one positive and two
  negative group pairs for each row (the groups of option B, owner 2026-11-13).
- **Held-out discipline.** The held-out set was scored once, with the recipe chosen on the CV, and the primary reading (two-step) fixed before the run. The script now refuses a second scoring without `--again`.
- **Code.** The scripts at commit `fbc0dd7` (the held-out run) and their later formatting and one-source-of-definitions changes (no change of recipe).

## Figures (accuracy, 95% interval by bootstrap over articles)

| | Untouched model | Flat fine-tune (CV) | Mixed fine-tune (CV) | Mixed fine-tune (held-out, 135 rows) |
|---|---|---|---|---|
| label, flat argmax | 56.4% (train) / 43.0% (held-out) | 64.0% | 62.2% | 52.6% [44.6, 60.2] |
| label, two-step (primary, fixed before the run) | 53.8% / 41.5% | 61.8% | 64.4% | 48.9% [41.1, 56.7] |
| label, group x label | | 63.6% | 64.0% | 50.4% [42.5, 58.5] |
| group, own hypotheses | 59.1% / 51.9% | 66.2% | 69.8% | 62.2% [54.6, 70.1] |
| group, via the flat label | 62.7% / 55.6% | 68.9% | 68.0% | 65.9% [58.9, 73.3] |

The bar of D24 (80% agreement on the held-out 200) is not met by any of them. Mixed minus flat on the CV label: -2.1 points [-6.0, +1.7], not distinguishable.
The step of `--step category` uses the untouched model (the first column), not a fine-tune.
