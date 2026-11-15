# The LLM judge on step 7: 80 real sub-news, blind (T-209)

- Judge model: `deepseek-chat` (the `LLM_MODEL` of `.env`; it wrote none of the labels compared). Date 2026-11-15. Sample: a seeded draw (seed 20261115) of 80 of the 300 real sub-news under 1,000 characters (the text the step reads).
- Judge prompt `src/news_nlp/eval/prompts/category_step.md`, SHA-256 `4acd175ff9ee5e98…`; temperature 0.
- 80 judged, 0 parse failures (excluded). The judge is a model: agreement with a judge, not truth.
- The judge never saw a model's label. The assistant's tags and the judge are both LLM output: their agreement is not an independent check.

| labels | agree with the judge (strict) [95%] | with the judge's second label too |
|---|---|---|
| untouched zero-shot (the step until now) | 72.5% [62.5%, 82.5%] | 88.8% |
| **fine-tune pinned in `config/category_step.json`** | 73.8% [63.7%, 83.8%] | 87.5% |
| the assistant's tags (a second LLM labeller) | 81.2% [72.5%, 88.8%] | 90.0% |

Paired: the fine-tune agrees with the judge where the untouched model does not on 8 texts, the reverse on 7; exact sign test p = 1.000.

`other`: the judge says `other` on 4 texts, the pinned model on 3.

## Per class of the judge (the texts the judge puts in the class)

| class | texts | fine-tune right | untouched right |
|---|---|---|---|
| earnings_performance | 15 | 13 | 12 |
| mergers_acquisitions | 11 | 8 | 8 |
| leadership_governance | 1 | 0 | 0 |
| legal_regulatory | 3 | 3 | 2 |
| product_innovation | 4 | 2 | 2 |
| labor_human_capital | 3 | 1 | 1 |
| market_analyst_sentiment | 30 | 23 | 27 |
| partnerships_business_dev | 4 | 4 | 3 |
| investor_activity | 5 | 5 | 2 |
| other | 4 | 0 | 1 |

Largest disagreements (judge -> pinned model):
- `market_analyst_sentiment` -> `legal_regulatory`: 2
- `mergers_acquisitions` -> `other`: 2
- `other` -> `partnerships_business_dev`: 2
- `market_analyst_sentiment` -> `earnings_performance`: 2
- `earnings_performance` -> `market_analyst_sentiment`: 2
- `product_innovation` -> `other`: 1
