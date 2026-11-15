# First run -> second run: 1216 rows both have (1224 and 1216 rows in all)


## Each run on its own tags (n = 1216)

accuracy 85.4% -> 87.8%, macro F1 0.706 -> 0.748

| class | rows (second tags) | precision | recall | ROC AUC |
|---|---|---|---|---|
| earnings_performance | 547 | 0.95 -> 0.97 | 0.97 -> 0.97 | 0.99 -> 1.00 |
| mergers_acquisitions | 81 | 0.78 -> 0.80 | 0.89 -> 0.89 | 0.99 -> 0.99 |
| leadership_governance | 40 | 0.89 -> 0.88 | 0.92 -> 0.90 | 0.99 -> 1.00 |
| legal_regulatory | 45 | 0.70 -> 0.72 | 0.70 -> 0.73 | 0.99 -> 0.98 |
| product_innovation | 90 | 0.82 -> 0.87 | 0.85 -> 0.86 | 0.98 -> 0.99 |
| capital_shareholder_returns | 40 | 0.70 -> 0.78 | 0.76 -> 0.70 | 0.96 -> 0.96 |
| labor_human_capital | 36 | 0.81 -> 0.84 | 0.86 -> 0.86 | 1.00 -> 1.00 |
| market_analyst_sentiment | 214 | 0.84 -> 0.89 | 0.85 -> 0.88 | 0.96 -> 0.98 |
| partnerships_business_dev | 32 | 0.63 -> 0.77 | 0.55 -> 0.75 | 0.93 -> 0.97 |
| reputational_harm | 39 | 0.71 -> 0.71 | 0.63 -> 0.74 | 0.98 -> 0.98 |
| investor_activity | 20 | 0.58 -> 0.50 | 0.28 -> 0.50 | 0.91 -> 0.93 |
| other | 32 | 0.33 -> 0.29 | 0.15 -> 0.19 | 0.76 -> 0.80 |

## Both runs on the second tags (n = 1216)

accuracy 87.9% -> 87.8%, macro F1 0.738 -> 0.748

| class | rows (second tags) | precision | recall | ROC AUC |
|---|---|---|---|---|
| earnings_performance | 547 | 0.97 -> 0.97 | 0.97 -> 0.97 | 1.00 -> 1.00 |
| mergers_acquisitions | 81 | 0.81 -> 0.80 | 0.90 -> 0.89 | 0.99 -> 0.99 |
| leadership_governance | 40 | 0.95 -> 0.88 | 0.90 -> 0.90 | 1.00 -> 1.00 |
| legal_regulatory | 45 | 0.75 -> 0.72 | 0.73 -> 0.73 | 0.99 -> 0.98 |
| product_innovation | 90 | 0.83 -> 0.87 | 0.86 -> 0.86 | 0.99 -> 0.99 |
| capital_shareholder_returns | 40 | 0.73 -> 0.78 | 0.80 -> 0.70 | 0.96 -> 0.96 |
| labor_human_capital | 36 | 0.84 -> 0.84 | 0.86 -> 0.86 | 1.00 -> 1.00 |
| market_analyst_sentiment | 214 | 0.87 -> 0.89 | 0.90 -> 0.88 | 0.98 -> 0.98 |
| partnerships_business_dev | 32 | 0.66 -> 0.77 | 0.72 -> 0.75 | 0.98 -> 0.97 |
| reputational_harm | 39 | 0.71 -> 0.71 | 0.74 -> 0.74 | 0.98 -> 0.98 |
| investor_activity | 20 | 0.58 -> 0.50 | 0.35 -> 0.50 | 0.92 -> 0.93 |
| other | 32 | 0.33 -> 0.29 | 0.12 -> 0.19 | 0.77 -> 0.80 |

## The older rows only, each run on its own tags (n = 217)

accuracy 69.6% -> 81.6%, macro F1 0.558 -> 0.714

| class | rows (second tags) | precision | recall | ROC AUC |
|---|---|---|---|---|
| earnings_performance | 78 | 0.80 -> 0.95 | 0.98 -> 0.95 | 0.95 -> 1.00 |
| mergers_acquisitions | 16 | 0.65 -> 0.83 | 0.93 -> 0.94 | 0.98 -> 0.99 |
| leadership_governance | 10 | 0.78 -> 0.90 | 1.00 -> 0.90 | 1.00 -> 1.00 |
| legal_regulatory | 5 | 0.33 -> 0.50 | 0.50 -> 0.80 | 0.89 -> 0.98 |
| product_innovation | 27 | 0.79 -> 0.83 | 0.85 -> 0.93 | 0.96 -> 0.98 |
| capital_shareholder_returns | 5 | 0.57 -> 0.50 | 0.67 -> 0.60 | 0.90 -> 0.99 |
| labor_human_capital | 19 | 0.73 -> 0.78 | 0.89 -> 0.95 | 0.99 -> 0.99 |
| market_analyst_sentiment | 14 | 0.35 -> 0.67 | 0.30 -> 0.86 | 0.69 -> 0.95 |
| partnerships_business_dev | 8 | 0.46 -> 0.86 | 0.38 -> 0.75 | 0.77 -> 0.98 |
| reputational_harm | 16 | 0.85 -> 0.89 | 0.48 -> 0.50 | 0.95 -> 0.95 |
| investor_activity | 3 | 0.50 -> 0.67 | 0.12 -> 0.67 | 0.88 -> 0.94 |
| other | 16 | 0.00 -> 0.14 | 0.00 -> 0.06 | 0.44 -> 0.55 |

## The older rows only, both on the second tags (n = 217)

accuracy 83.4% -> 81.6%, macro F1 0.719 -> 0.714

| class | rows (second tags) | precision | recall | ROC AUC |
|---|---|---|---|---|
| earnings_performance | 78 | 0.96 -> 0.95 | 0.97 -> 0.95 | 1.00 -> 1.00 |
| mergers_acquisitions | 16 | 0.80 -> 0.83 | 1.00 -> 0.94 | 0.99 -> 0.99 |
| leadership_governance | 10 | 1.00 -> 0.90 | 0.90 -> 0.90 | 1.00 -> 1.00 |
| legal_regulatory | 5 | 0.67 -> 0.50 | 0.80 -> 0.80 | 0.94 -> 0.98 |
| product_innovation | 27 | 0.82 -> 0.83 | 0.85 -> 0.93 | 0.98 -> 0.98 |
| capital_shareholder_returns | 5 | 0.71 -> 0.50 | 1.00 -> 0.60 | 0.99 -> 0.99 |
| labor_human_capital | 19 | 0.77 -> 0.78 | 0.89 -> 0.95 | 0.99 -> 0.99 |
| market_analyst_sentiment | 14 | 0.71 -> 0.67 | 0.86 -> 0.86 | 0.97 -> 0.95 |
| partnerships_business_dev | 8 | 0.54 -> 0.86 | 0.88 -> 0.75 | 0.99 -> 0.98 |
| reputational_harm | 16 | 0.85 -> 0.89 | 0.69 -> 0.50 | 0.96 -> 0.95 |
| investor_activity | 3 | 0.50 -> 0.67 | 0.33 -> 0.67 | 0.95 -> 0.94 |
| other | 16 | 0.00 -> 0.14 | 0.00 -> 0.06 | 0.50 -> 0.55 |

## The sheet's rows only (same tags in both runs) (n = 999)

accuracy 88.9% -> 89.2%, macro F1 0.748 -> 0.759

| class | rows (second tags) | precision | recall | ROC AUC |
|---|---|---|---|---|
| earnings_performance | 469 | 0.97 -> 0.97 | 0.97 -> 0.98 | 1.00 -> 1.00 |
| mergers_acquisitions | 65 | 0.81 -> 0.79 | 0.88 -> 0.88 | 0.99 -> 0.99 |
| leadership_governance | 30 | 0.93 -> 0.87 | 0.90 -> 0.90 | 1.00 -> 1.00 |
| legal_regulatory | 40 | 0.76 -> 0.76 | 0.72 -> 0.72 | 0.99 -> 0.98 |
| product_innovation | 63 | 0.83 -> 0.88 | 0.86 -> 0.83 | 0.99 -> 0.99 |
| capital_shareholder_returns | 35 | 0.73 -> 0.83 | 0.77 -> 0.71 | 0.96 -> 0.95 |
| labor_human_capital | 17 | 0.93 -> 0.93 | 0.82 -> 0.76 | 1.00 -> 1.00 |
| market_analyst_sentiment | 200 | 0.88 -> 0.91 | 0.90 -> 0.89 | 0.98 -> 0.99 |
| partnerships_business_dev | 24 | 0.73 -> 0.75 | 0.67 -> 0.75 | 0.99 -> 0.97 |
| reputational_harm | 23 | 0.64 -> 0.66 | 0.78 -> 0.91 | 0.98 -> 0.98 |
| investor_activity | 17 | 0.60 -> 0.47 | 0.35 -> 0.47 | 0.92 -> 0.94 |
| other | 16 | 0.36 -> 0.36 | 0.25 -> 0.31 | 0.82 -> 0.83 |

Older rows whose tag changed between the runs: 48.

