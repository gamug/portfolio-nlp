# The category model on real data: 300 sub-news pairs under 1,000 characters (the text the step reads)

Drawn from the first corpus run (`category-65f33b55b26b`, cut `units5-d3e2cbfa830e`), seeded, none from a tagged set. The tags are the assistant's (blind to both models), not the owner's. Tag counts: {'market_analyst_sentiment': np.int64(99), 'earnings_performance': np.int64(56), 'other': np.int64(39), 'mergers_acquisitions': np.int64(30), 'partnerships_business_dev': np.int64(26), 'product_innovation': np.int64(18), 'legal_regulatory': np.int64(13), 'investor_activity': np.int64(7), 'leadership_governance': np.int64(5), 'labor_human_capital': np.int64(4), 'reputational_harm': np.int64(2), 'capital_shareholder_returns': np.int64(1)}. Rows the tagger flagged unclear: 107.

## Selected recipe (refit on 1,216 rows)

- **label agreement 71.3% [66.0%, 76.3%]**, group agreement 73.0% [68.0%, 78.0%], macro F1 over the 12 labels 0.637

| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|---|
| earnings_performance | 56 | 61 | 0.82 [0.72, 0.91] | 0.89 [0.80, 0.96] | 0.98 |
| mergers_acquisitions | 30 | 30 | 0.80 [0.65, 0.94] | 0.80 [0.64, 0.94] | 0.98 |
| leadership_governance | 5 | 7 | 0.29 [0.00, 0.67] | 0.40 [0.00, 1.00] | 0.98 |
| legal_regulatory | 13 | 13 | 0.77 [0.50, 1.00] | 0.77 [0.54, 1.00] | 0.99 |
| product_innovation | 18 | 28 | 0.46 [0.29, 0.65] | 0.72 [0.50, 0.92] | 0.93 |
| capital_shareholder_returns | 1 | 4 | 0.25 [0.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 |
| labor_human_capital | 4 | 4 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 |
| market_analyst_sentiment | 99 | 98 | 0.81 [0.73, 0.88] | 0.80 [0.72, 0.87] | 0.95 |
| partnerships_business_dev | 26 | 31 | 0.65 [0.48, 0.81] | 0.77 [0.58, 0.93] | 0.95 |
| reputational_harm | 2 | 6 | 0.33 [0.00, 0.83] | 1.00 [1.00, 1.00] | 0.99 |
| investor_activity | 7 | 8 | 0.75 [0.40, 1.00] | 0.86 [0.50, 1.00] | 0.92 |
| other | 39 | 10 | 0.30 [0.00, 0.64] | 0.08 [0.00, 0.18] | 0.68 |

## Untouched zero-shot model (what the step ran), 14 labels read as 12

- **label agreement 65.7% [60.3%, 71.0%]**, group agreement 67.7% [62.3%, 73.0%], macro F1 over the 12 labels 0.483

| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|---|
| earnings_performance | 56 | 62 | 0.79 [0.68, 0.89] | 0.88 [0.78, 0.95] | 0.97 |
| mergers_acquisitions | 30 | 37 | 0.59 [0.44, 0.75] | 0.73 [0.58, 0.88] | 0.96 |
| leadership_governance | 5 | 3 | 0.33 [0.00, 1.00] | 0.20 [0.00, 0.67] | 0.95 |
| legal_regulatory | 13 | 5 | 1.00 [1.00, 1.00] | 0.38 [0.12, 0.67] | 0.97 |
| product_innovation | 18 | 11 | 0.45 [0.17, 0.78] | 0.28 [0.08, 0.50] | 0.76 |
| capital_shareholder_returns | 1 | 0 | - | 0.00 [0.00, 0.00] | 0.94 |
| labor_human_capital | 4 | 4 | 0.75 [0.00, 1.00] | 0.75 [0.00, 1.00] | 0.99 |
| market_analyst_sentiment | 99 | 133 | 0.67 [0.59, 0.74] | 0.90 [0.84, 0.95] | 0.87 |
| partnerships_business_dev | 26 | 18 | 0.78 [0.56, 0.95] | 0.54 [0.32, 0.75] | 0.91 |
| reputational_harm | 2 | 5 | 0.20 [0.00, 0.75] | 0.50 [0.00, 1.00] | 0.92 |
| investor_activity | 7 | 4 | 0.75 [0.00, 1.00] | 0.43 [0.00, 0.83] | 0.84 |
| other | 39 | 18 | 0.28 [0.08, 0.50] | 0.13 [0.03, 0.24] | 0.46 |

## Label agreement by part of the sample

| rows | n | selected | untouched |
|---|---|---|---|
| first batch (idx < 150) | 150 | 69.3% | 62.7% |
| second batch (idx >= 150) | 150 | 73.3% | 68.7% |
| all | 300 | 71.3% | 65.7% |
| tagger sure | 193 | 83.9% | 79.3% |
| tagger unsure (flagged) | 107 | 48.6% | 41.1% |

Paired: the selected model is right where the untouched one is wrong on 38 rows and the reverse on 21 (two-sided sign test p = 0.036).
