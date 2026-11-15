# The held-out 200 scored once with the selected model (135 rows, refit on 1216 rows)

- **label agreement 54.1% [45.9%, 62.3%]**, group agreement 65.2% [57.4%, 72.7%]; bar of D24: 80% on the held-out single rows: **NOT passed**
- rows of kind `single`: 135, label agreement 54.1% [45.9%, 62.3%]
- macro F1 over the 12 labels that have held-out rows 0.448; over all 12 (a label with no row or never predicted counts 0) 0.448

## Per class, one versus rest (the top label of each row), 95% intervals by article

| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|---|
| earnings_performance | 27 | 34 | 0.74 [0.58, 0.91] | 0.93 [0.81, 1.00] | 0.96 |
| mergers_acquisitions | 5 | 11 | 0.36 [0.09, 0.67] | 0.80 [0.33, 1.00] | 0.90 |
| leadership_governance | 11 | 5 | 1.00 [1.00, 1.00] | 0.45 [0.14, 0.75] | 0.64 |
| legal_regulatory | 11 | 14 | 0.43 [0.22, 0.62] | 0.55 [0.25, 0.83] | 0.92 |
| product_innovation | 22 | 25 | 0.60 [0.42, 0.78] | 0.68 [0.48, 0.88] | 0.89 |
| capital_shareholder_returns | 7 | 4 | 0.50 [0.00, 1.00] | 0.29 [0.00, 0.67] | 0.73 |
| labor_human_capital | 3 | 6 | 0.33 [0.00, 1.00] | 0.67 [0.00, 1.00] | 0.98 |
| market_analyst_sentiment | 17 | 19 | 0.42 [0.20, 0.65] | 0.47 [0.23, 0.70] | 0.79 |
| partnerships_business_dev | 5 | 1 | 1.00 [1.00, 1.00] | 0.20 [0.00, 0.67] | 0.73 |
| reputational_harm | 7 | 4 | 0.50 [0.00, 1.00] | 0.29 [0.00, 0.67] | 0.93 |
| investor_activity | 16 | 3 | 0.67 [0.00, 1.00] | 0.12 [0.00, 0.31] | 0.77 |
| other | 4 | 9 | 0.11 [0.00, 0.30] | 0.25 [0.00, 0.76] | 0.71 |

## The same held-out rows, the earlier models (12 labels, `competitive_actions` and `peer_comparison` read as `other`)

| model | rows | label agreement |
|---|---|---|
| untouched zero-shot (the step today) | 135 | 43.7% |
| **selected recipe, refit on 1,216 rows** | 135 | **54.1%** |

## The largest confusions (gold -> predicted, rows)

- `investor_activity` -> `market_analyst_sentiment`: 8
- `market_analyst_sentiment` -> `earnings_performance`: 4
- `reputational_harm` -> `legal_regulatory`: 4
- `investor_activity` -> `other`: 3
- `capital_shareholder_returns` -> `mergers_acquisitions`: 2
- `investor_activity` -> `earnings_performance`: 2
- `leadership_governance` -> `mergers_acquisitions`: 2
- `legal_regulatory` -> `mergers_acquisitions`: 2

Held-out tags are of the earlier convention: 16 rows are tagged `investor_activity`, 8 of them predicted `market_analyst_sentiment` (the review of 2026-11-15 moved analyst ratings out of `investor_activity` in the training rows; the held-out tags were not reviewed).
