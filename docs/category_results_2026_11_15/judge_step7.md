# The LLM judge on step 7: 80 real sub-news, blind (T-209)

- 80 judged, 0 parse failures (excluded). The judge is a model: agreement with a judge, not truth.
- Texts: a seeded draw of the 300 real sub-news under 1,000 characters (the text the step reads); the judge never saw a model's label.

| labels | agree with the judge (strict) [95%] | with the judge's second label too |
|---|---|---|
| untouched zero-shot (the step until now) | 72.5% [62.5%, 82.5%] | 88.8% |
| **fine-tune pinned in `config/category_step.json`** | 73.8% [63.7%, 83.8%] | 87.5% |
| the assistant's tags (calibration of the judge) | 81.2% [72.5%, 88.8%] | 90.0% |

Paired: the fine-tune agrees with the judge where the untouched model does not on 8 texts, the reverse on 7; exact sign test p = 1.000.

`other`: the judge says `other` on 4 texts, the pinned model on 3.

Largest disagreements (judge -> pinned model):
- `earnings_performance` -> `market_analyst_sentiment`: 2
- `market_analyst_sentiment` -> `earnings_performance`: 2
- `market_analyst_sentiment` -> `legal_regulatory`: 2
- `mergers_acquisitions` -> `other`: 2
- `other` -> `partnerships_business_dev`: 2
- `labor_human_capital` -> `partnerships_business_dev`: 1
