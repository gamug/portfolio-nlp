# Category taxonomy (D15)

The labels step 7 of the stepped pipeline gives each text **per main company** (`PLAN.md` D15, D29, D39 (7)): a
summary or a sub-news is read with its main company, and the label says which dimension of that company's news the
text is about. The owner settled the taxonomy on 2026-10-29 and the rules for close cases on 2026-11-05 and
2026-11-11; the tagging guide with every owner call and its example articles is `docs/labelling-spec-v2.md`, and this
page is the list of labels it applies.

**Where it is in code.** `news_nlp.retag.D15_LABELS` is the machine-readable list (the nine slugs of
`news_nlp.taxonomy.CATEGORY_SLUGS`, the four added in D15, then `other`, `uncertain` and `unrelated`);
`retag.COMPANY_LABELS` is the same without `unrelated`, which is a scope, not a label of a company.
`retag.NEW_LABEL_GUIDE` carries the one-line definitions of the added labels for the owner's guide CSV.
`tests/news_nlp/test_labelling_spec.py` checks that `docs/labelling-spec-v2.md` names every one of them.

**The model.** The step (`--step category`, `docs/category_step.md`) runs the entailment fine-tune `category-entailment-single-12` (`docs/category_training.md`), pinned by the owner on 2026-11-15 as the default although it does not meet the bar of D24 (>= 80% agreement on the held-out `gold/category_gold_v2_test.csv`, 200 texts): 54.1% on its 135 single rows, 43.7% for the untouched zero-shot model it replaced. It reads 12 labels and one tag per row (below). The v1 category stage, a zero-shot
classifier over the nine slugs and `other` per (URL, ticker) pair, is deprecated (owner, 2026-11-12); its taxonomy
page, sources and calibration are in `docs/legacy/category-taxonomy-v1.md`.

**The groups (option B, owner 2026-11-13).** Each label belongs to one group of six; `news_category.group_label` carries it. `results_capital`: `earnings_performance`, `capital_shareholder_returns`. `market_view`: `market_analyst_sentiment`, `investor_activity`. `deals_alliances`: `mergers_acquisitions`, `partnerships_business_dev`. `product_people`: `product_innovation`, `labor_human_capital`. `governance_legal_reputation`: `leadership_governance`, `legal_regulatory`, `reputational_harm`. `other`: `other`. (The step reads 12 labels: `competitive_actions` and `peer_comparison`, which the tagger may still meet below, are read as `other`, owner 2026-11-14.) The single source in code is `GROUPS` of `src/news_category_stage.py`.

**`uncertain` is a human answer.** The provisional step chooses among the 14 labels above and never writes `uncertain` (a model that abstains would be another model and revision); a row the owner tagged `uncertain` is not in the train or the test rows of the model, unless the review of 2026-11-15 gave it a tag (two rows of the 1,000-row sheet did; `docs/category_training.md` 3.2).

**The model reads shorter definitions.** The "The text is about" column above is the rule for the human tagger; the provisional step reads the short forms of `docs/category_step.md` (the ones measured in the baseline of 2026-11-13; a test keeps that table equal to the code), and does not read the "Not this label when" column. Reading the full rules is another revision, to be measured first (T-187).

## One tag per row (owner, 2026-11-14 and 2026-11-15)

The model of T-187 (D43) is trained on **twelve labels and one tag per (article, company)**: `competitive_actions` and `peer_comparison` are tagged `other` (owner, 2026-11-14), and when several labels fit, the tag is the best fit. These are the rules the sheet of 1,000 rows and the 225 older rows were read with (2026-11-15); they add to the owner's calls of the table below and to `docs/labelling-spec-v2.md`, and a new row is tagged by them. The record of how they were applied and what they changed is `docs/category_training.md` 3.

1. An analyst rating or price target is the story: `market_analyst_sentiment`. A fund or an investor buying or selling a stake: `investor_activity`.
2. Results with guidance, a delay, a charge or a share move on the results: `earnings_performance`, even when a dividend or a buyback is also named.
3. A regulator acting on the company (an investigation, a fine, a ban, a ruling, a hearing): `legal_regulatory`.
4. A drug, vaccine or product approval, a trial result, a delay, a feature or a launch: `product_innovation`.
5. A launch built on a partner is `partnerships_business_dev` when the partner is the story and `product_innovation` when the product is.
6. A deal, a bid, a break-up or the sale of a unit: `mergers_acquisitions`. An IPO: `capital_shareholder_returns`.
7. A change of index membership, a rate or price change, a discount or a macro data release is `other`, even when the share price moves.
8. A proxy fight or pressure on the board: `leadership_governance`, not `investor_activity`.
9. An (article, company) pair gets one row and one tag.

## The labels

A text gets one label per main company; a company may carry two labels in one article when the article reports two
events about it (owner call 12). Take the dimension the **headline event** belongs to; an analyst's reaction to that
event does not change it (the reaction is `market_analyst_sentiment` only when the reaction is the story).

| Label | The text is about | Not this label when |
|---|---|---|
| `earnings_performance` | results, guidance, revenue or profit figures | a rating on the results is the story (`market_analyst_sentiment`) |
| `mergers_acquisitions` | deals, takeovers, divestitures | a joint effort with no change of ownership (`partnerships_business_dev`) |
| `leadership_governance` | executive appointments and departures, board actions, an executive's insider trade | a fund or famous investor moves a stake (`investor_activity`) |
| `legal_regulatory` | litigation, regulatory action, sanctions, bankruptcy | the public standing is the story with no legal event (`reputational_harm`) |
| `product_innovation` | product launches, R&D, technology; also news about a product the public consumes when a regulator or a health body is the actor (together with `legal_regulatory`, owner call 15) | |
| `capital_shareholder_returns` | dividends, buybacks, debt and credit actions of the company | an investor buys or sells the stock (`investor_activity`) |
| `labor_human_capital` | layoffs, hiring, workforce and labour relations | |
| `market_analyst_sentiment` | analyst ratings and price targets, stock price moves, an investor's bullish or bearish view | the article compares companies (`peer_comparison`); a stake moves (`investor_activity`) |
| `partnerships_business_dev` | strategic alliances, joint ventures, distribution deals | the company acts against a rival (`competitive_actions`) |
| `reputational_harm` | an event that damages or repairs the company's public standing with no legal or earnings event as the story: apologies, service failures, controversies, boycotts, pledges after criticism | a lawsuit or fine (`legal_regulatory`); layoffs (`labor_human_capital`) |
| `competitive_actions` | the company's own move or statement against or about a competitor: share fights, rival facilities, pricing pressure | a joint effort (`partnerships_business_dev`); a rating (`market_analyst_sentiment`); a writer weighing companies (`peer_comparison`) |
| `investor_activity` | what investors do with the company's stock when a stake moves: a fund or famous investor buys or sells, an activist position, 13F filings, large block trades (owner call 3) | the company's own buyback or dividend (`capital_shareholder_returns`); an executive's insider trade (`leadership_governance`); a view alone (`market_analyst_sentiment`) |
| `peer_comparison` | the point of the article is a comparison between companies (which is the better stock, who is ahead); every compared company gets it, and a single company weighed against its sector or peers too (owner call 6) | a rating or price target on one company (`market_analyst_sentiment`) |
| `other` | the event is clear and about the company but fits none of the 13 labels above (an anniversary, an office move) | a CEO's view on the economy (that body is `market_economy`, with no company label) |
| `uncertain` | two or more labels fit equally and the headline event does not decide, or the event is not in the first 6,000 characters; a label-level answer only, the scope never abstains (D20) | the event is clear but fits no label (`other`) |

`unrelated` is a **scope**, not a label: a body not about business, finance or the economy, flagged and skipped by
every later step (D16). The scopes (`single`, `multi`, `sector`, `market_economy`, `investing_general`, `unrelated`)
are decided before any label, by step 1 (`docs/scope_training.md`); `market_economy`, `investing_general` and
`unrelated` stop after the scope and get no category (D39).

## What the nine original labels drew on

The nine slugs kept from the v1 taxonomy were triangulated from RavenPack News Analytics, the SASB Materiality Map, IPTC
Media Topics and Refinitiv/TRNA topic codes; the source of each is in `docs/legacy/category-taxonomy-v1.md`. The four
added labels come from the owner's notes on the 720 articles tagged in October 2026 (`docs/labelling-spec-v2.md`).
