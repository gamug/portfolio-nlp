# The LLM judge on step 7: 300 real sub-news, blind (T-209)

- Judge model: `deepseek-chat`; it wrote none of the labels compared. Date 2026-11-15. Sample: all 300 real sub-news under 1,000 characters (the text the step reads). Rules: the tagging rules of the project, new convention (`docs/category-taxonomy.md`, "One tag per row"), not the older one of the held-out.
- Judge prompt `src/news_nlp/eval/prompts/category_step.md`, SHA-256 `4acd175ff9ee5e98…`; temperature 0, **not repeatable** (see "Two runs, one sample" when both runs exist).
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

## Two runs, one sample
- The 80 texts of the first run are inside these 300; on them the judge's label **changed on 2 of 80** between the two runs (temperature 0 does not make this judge repeatable).
- The 220 texts the first run did not see: fine-tune 78.2%, untouched 70.9%; paired 32 against 16, exact sign test p = 0.029. The p of all 300 is a second look at a sample that began with the first run: read it as nominal.

## Official reading, and the four weak labels as `other` (owner, 2026-11-17)

**Official reading:** the model agrees with the judge when its label is the judge's first **or second** label (the "with the judge's second label too" column above). Pinned model **88.0% [83.8, 91.2]** (264 of 300), untouched zero-shot 81.0% [76.2, 85.0]. Strict (first label only): 77.0% and 71.0% (Wilson intervals here; the table above uses another interval method).
**Through the planned definition** (`product_innovation`, `leadership_governance`, `capital_shareholder_returns`, `reputational_harm` read as `other` on both sides; nothing retrained; `scripts/category_planned_8_2026_11_16.py`, `planned_8_labels.json`, key `judge_step7`):

| pinned model | 12 labels | 8 labels (`other` scored) | 7 measured labels (rows the judge puts in `other` left out) |
|---|---|---|---|
| official (first or second) | 88.0% | 92.0% [88.4, 94.6] | 94.4% [90.9, 96.7] (252 rows) |
| strict (first only) | 77.0% | 80.0% [75.1, 84.1] | 81.7% [76.5, 86.0] |

The 8- and 7-label columns are the 12-label model read through the fold; the 8-label model is measured when it is built (T-214).

