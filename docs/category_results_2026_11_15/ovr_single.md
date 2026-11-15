# One versus rest per class, 12 labels, 1216 rows from 1049 articles (out of fold)

- **entailment, new tags**: 1216 rows, accuracy 87.8%, macro F1 0.748
- **untouched**: 1214 rows, accuracy 75.5%, macro F1 0.525
- **entailment, old tags**: 1214 rows, accuracy 74.7%, macro F1 0.645

## entailment, new tags: precision, recall and ROC AUC per class

| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|---|
| earnings_performance | 547 | 552 | 0.97 [0.95, 0.98] | 0.97 [0.96, 0.99] | 1.00 |
| mergers_acquisitions | 81 | 90 | 0.80 [0.71, 0.88] | 0.89 [0.81, 0.95] | 0.99 |
| leadership_governance | 40 | 41 | 0.88 [0.77, 0.97] | 0.90 [0.80, 0.98] | 1.00 |
| legal_regulatory | 45 | 46 | 0.72 [0.58, 0.85] | 0.73 [0.60, 0.85] | 0.98 |
| product_innovation | 90 | 89 | 0.87 [0.79, 0.94] | 0.86 [0.78, 0.92] | 0.99 |
| capital_shareholder_returns | 40 | 36 | 0.78 [0.63, 0.92] | 0.70 [0.56, 0.84] | 0.96 |
| labor_human_capital | 36 | 37 | 0.84 [0.71, 0.94] | 0.86 [0.75, 0.97] | 1.00 |
| market_analyst_sentiment | 214 | 212 | 0.89 [0.85, 0.93] | 0.88 [0.84, 0.92] | 0.98 |
| partnerships_business_dev | 32 | 31 | 0.77 [0.62, 0.90] | 0.75 [0.60, 0.90] | 0.97 |
| reputational_harm | 39 | 41 | 0.71 [0.57, 0.85] | 0.74 [0.61, 0.88] | 0.98 |
| investor_activity | 20 | 20 | 0.50 [0.26, 0.73] | 0.50 [0.27, 0.73] | 0.93 |
| other | 32 | 21 | 0.29 [0.11, 0.50] | 0.19 [0.06, 0.36] | 0.80 |

## The three models on the same rows and the same (new) tags: precision / recall / ROC AUC

| class | rows | entailment, new tags | untouched | entailment, old tags |
|---|---|---|---|---|
| earnings_performance | 547 | 0.97 / 0.97 / 1.00 | 0.94 / 0.94 / 0.99 | 0.97 / 0.97 / 0.99 |
| mergers_acquisitions | 81 | 0.80 / 0.89 / 0.99 | 0.72 / 0.81 / 0.99 | 0.82 / 0.88 / 0.99 |
| leadership_governance | 40 | 0.88 / 0.90 / 1.00 | 0.95 / 0.45 / 0.99 | 0.81 / 0.85 / 0.99 |
| legal_regulatory | 45 | 0.72 / 0.73 / 0.98 | 0.79 / 0.24 / 0.93 | 0.68 / 0.76 / 0.99 |
| product_innovation | 90 | 0.87 / 0.86 / 0.99 | 0.72 / 0.42 / 0.91 | 0.91 / 0.78 / 0.98 |
| capital_shareholder_returns | 40 | 0.78 / 0.70 / 0.96 | 0.60 / 0.07 / 0.81 | 0.70 / 0.78 / 0.97 |
| labor_human_capital | 36 | 0.84 / 0.86 / 1.00 | 0.85 / 0.61 / 1.00 | 0.75 / 0.83 / 1.00 |
| market_analyst_sentiment | 214 | 0.89 / 0.88 / 0.98 | 0.61 / 0.95 / 0.92 | 0.69 / 0.16 / 0.95 |
| partnerships_business_dev | 32 | 0.77 / 0.75 / 0.97 | 0.71 / 0.38 / 0.93 | 0.63 / 0.81 / 0.98 |
| reputational_harm | 39 | 0.71 / 0.74 / 0.98 | 0.41 / 0.31 / 0.87 | 0.63 / 0.79 / 0.98 |
| investor_activity | 20 | 0.50 / 0.50 / 0.93 | 0.54 / 0.35 / 0.86 | 0.08 / 0.70 / 0.87 |
| other | 32 | 0.29 / 0.19 / 0.80 | 0.17 / 0.38 / 0.38 | 0.57 / 0.12 / 0.68 |
