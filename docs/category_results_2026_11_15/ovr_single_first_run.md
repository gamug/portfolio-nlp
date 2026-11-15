# One versus rest per class, 12 labels, 1224 rows from 1049 articles (out of fold)

- **entailment, new tags**: 1224 rows, accuracy 85.0%, macro F1 0.701
- **untouched**: 1222 rows, accuracy 73.9%, macro F1 0.515
- **entailment, old tags**: 1222 rows, accuracy 73.6%, macro F1 0.629

## entailment, new tags: precision, recall and ROC AUC per class

| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|---|
| earnings_performance | 533 | 549 | 0.95 [0.91, 0.97] | 0.97 [0.96, 0.99] | 0.99 |
| mergers_acquisitions | 80 | 91 | 0.77 [0.68, 0.85] | 0.88 [0.79, 0.94] | 0.99 |
| leadership_governance | 39 | 39 | 0.90 [0.80, 0.98] | 0.90 [0.79, 0.98] | 0.99 |
| legal_regulatory | 46 | 44 | 0.70 [0.57, 0.84] | 0.67 [0.53, 0.80] | 0.99 |
| product_innovation | 89 | 96 | 0.79 [0.71, 0.87] | 0.85 [0.77, 0.92] | 0.98 |
| capital_shareholder_returns | 41 | 44 | 0.70 [0.56, 0.84] | 0.76 [0.62, 0.89] | 0.96 |
| labor_human_capital | 36 | 38 | 0.79 [0.65, 0.93] | 0.83 [0.70, 0.95] | 1.00 |
| market_analyst_sentiment | 221 | 221 | 0.84 [0.79, 0.89] | 0.84 [0.76, 0.91] | 0.96 |
| partnerships_business_dev | 41 | 36 | 0.64 [0.48, 0.79] | 0.56 [0.41, 0.73] | 0.93 |
| reputational_harm | 46 | 42 | 0.69 [0.54, 0.83] | 0.63 [0.50, 0.77] | 0.98 |
| investor_activity | 25 | 12 | 0.58 [0.25, 0.88] | 0.28 [0.11, 0.48] | 0.91 |
| other | 27 | 12 | 0.33 [0.08, 0.64] | 0.15 [0.03, 0.33] | 0.76 |

## The three models on the same rows and the same (new) tags: precision / recall / ROC AUC

| class | rows | entailment, new tags | untouched | entailment, old tags |
|---|---|---|---|---|
| earnings_performance | 533 | 0.95 / 0.97 / 0.99 | 0.92 / 0.94 / 0.99 | 0.95 / 0.97 / 0.99 |
| mergers_acquisitions | 80 | 0.77 / 0.88 / 0.99 | 0.69 / 0.80 / 0.98 | 0.79 / 0.86 / 0.99 |
| leadership_governance | 39 | 0.90 / 0.90 / 0.99 | 0.90 / 0.46 / 0.99 | 0.77 / 0.85 / 0.99 |
| legal_regulatory | 46 | 0.70 / 0.67 / 0.99 | 0.79 / 0.24 / 0.92 | 0.64 / 0.70 / 0.99 |
| product_innovation | 89 | 0.79 / 0.85 / 0.98 | 0.67 / 0.42 / 0.91 | 0.86 / 0.78 / 0.98 |
| capital_shareholder_returns | 41 | 0.70 / 0.76 / 0.96 | 0.60 / 0.07 / 0.82 | 0.70 / 0.76 / 0.97 |
| labor_human_capital | 36 | 0.79 / 0.83 / 1.00 | 0.81 / 0.58 / 0.99 | 0.73 / 0.83 / 0.99 |
| market_analyst_sentiment | 221 | 0.84 / 0.84 / 0.96 | 0.60 / 0.91 / 0.91 | 0.73 / 0.16 / 0.93 |
| partnerships_business_dev | 41 | 0.64 / 0.56 / 0.93 | 0.78 / 0.34 / 0.86 | 0.63 / 0.66 / 0.93 |
| reputational_harm | 46 | 0.69 / 0.63 / 0.98 | 0.47 / 0.30 / 0.86 | 0.66 / 0.72 / 0.98 |
| investor_activity | 25 | 0.58 / 0.28 / 0.91 | 0.62 / 0.32 / 0.85 | 0.11 / 0.80 / 0.90 |
| other | 27 | 0.33 / 0.15 / 0.76 | 0.15 / 0.41 / 0.45 | 0.43 / 0.11 / 0.71 |

