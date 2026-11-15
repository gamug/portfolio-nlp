# The LLM judge on step 7: 300 real sub-news, blind (T-209)

- Judge model: `deepseek-chat` (the `LLM_MODEL` of `.env`; it wrote none of the labels compared). Date 2026-11-15. Sample: a seeded draw (seed 20261115) of 300 of the 300 real sub-news under 1,000 characters (the text the step reads).
- Judge prompt `src/news_nlp/eval/prompts/category_step.md`, SHA-256 `4acd175ff9ee5e98…`; temperature 0.
- 300 judged, 0 parse failures (excluded). The judge is a model: agreement with a judge, not truth.
- The judge never saw a model's label. The assistant's tags and the judge are both LLM output: their agreement is not an independent check.

| labels | agree with the judge (strict) [95%] | with the judge's second label too |
|---|---|---|
| untouched zero-shot (the step until now) | 71.0% [66.0%, 76.0%] | 81.0% |
| **fine-tune pinned in `config/category_step.json`** | 77.0% [72.0%, 81.7%] | 88.0% |
| the assistant's tags (a second LLM labeller) | 83.0% [78.7%, 87.0%] | 91.0% |

Paired: the fine-tune agrees with the judge where the untouched model does not on 41 texts, the reverse on 23; exact sign test p = 0.033.

`other`: the judge says `other` on 16 texts, the pinned model on 10.

## Per class of the judge (the texts the judge puts in the class)

| class | texts | fine-tune right | untouched right |
|---|---|---|---|
| earnings_performance | 54 | 51 | 47 |
| mergers_acquisitions | 31 | 26 | 23 |
| leadership_governance | 6 | 3 | 2 |
| legal_regulatory | 10 | 9 | 5 |
| product_innovation | 18 | 13 | 7 |
| capital_shareholder_returns | 3 | 2 | 0 |
| labor_human_capital | 8 | 4 | 3 |
| market_analyst_sentiment | 115 | 89 | 104 |
| partnerships_business_dev | 26 | 20 | 14 |
| reputational_harm | 5 | 3 | 2 |
| investor_activity | 8 | 7 | 3 |
| other | 16 | 4 | 3 |

Largest disagreements (judge -> pinned model):
- `market_analyst_sentiment` -> `earnings_performance`: 7
- `other` -> `partnerships_business_dev`: 5
- `market_analyst_sentiment` -> `product_innovation`: 5
- `partnerships_business_dev` -> `product_innovation`: 4
- `market_analyst_sentiment` -> `legal_regulatory`: 3
- `other` -> `market_analyst_sentiment`: 3
