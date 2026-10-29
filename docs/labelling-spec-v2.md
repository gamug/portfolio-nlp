# Labelling specification v2 (Work item 18, T-146a)

**Status: draft for the owner's review.** It is the guide for the re-tag of the 720 articles
(gold v2, T-146b), for the revised LLM labelling prompt (T-146c) and for the held-out gold
(T-146e). Where a rule below is marked **[owner call]**, the PLAN did not fix it and the owner
decides. Decisions it builds on: D11–D24 (`TASKS.md` T-143). The older 9-label definitions with
their close calls are in `docs/category-taxonomy.md`.

## What is tagged

One row per **unique article body** (D11). The title and the first 6,000 characters are what is
read (what the judge and the first tagging saw). Columns of the re-tag file
(`data/category_gold_v2_review.csv`):

| column | content |
|---|---|
| `scope` | one of the six scopes below (always filled) |
| `companies` | the **analysed** companies, tickers separated by `;` (single: exactly one; multi: two or more; the other scopes: none) |
| `company_labels` | one label per analysed company, `TICKER:label; TICKER:label` |
| `mentions` | companies named but not analysed (D23), tickers separated by `;` |
| `sub_industry` | a GICS sub-industry of the source (sector bodies only; spelling as in the source) |
| `notes` | anything that helps (never required) |

`v1_label`, `v1_notes` and `v1_scope_derived` show the first tagging as guidance.
`v1_scope_derived` is a regex reading of the v1 notes, not the owner's decision: overrule it freely.

## The six scopes

Decide in this order; the first scope whose test is met wins.

1. **`unrelated`**: not about business, finance or the economy (health news without a company
   angle, politics, lifestyle, career tips). *Examples:* 414317 (Israeli Covid study),
   286933 (House Intel Committee), 91934 (a job-interview question). A company named only as the
   setting does not make an article related. Flagged and skipped by every later stage (D16).
2. **`market_economy`**: about the market as a whole or the economy: indices, S&P 500, rates,
   inflation, tariffs, strategist calls on the whole market. No company analysis. *Examples:*
   130093 (economic recovery), 480248 (a double-digit rally), 296504 (S&P 500 at records). It is
   summarised in the `MARKET` / `ECONOMY` rows of `sector_summary` (D14).
3. **`sector`**: about one industry or sub-industry rather than a company: sector trends, an
   industry-wide event, a sector's outlook; companies appear as examples. Needs a `sub_industry`.
   *Examples:* 115964 (semiconductors), 323392 (spirits), 137874 (bank sector). Test: if the
   named company were replaced by another of the same industry, would the article still hold?
4. **`investing_general`**: investing or market-advice content about no company and no sector:
   investing philosophy, strategy pieces, personal-finance and "how to invest" pieces. *Example:*
   263795 (a bank's call for a new investing approach). No per-company analysis; summarised in
   the `MARKET` row (D22). **[owner call]** Articles where a company is the setting of a
   non-financial human-interest story (e.g. 61309, squatters in an Airbnb condo) were tagged
   `other` in v1 and derive to this scope; decide whether they are `investing_general`,
   `unrelated`, or `single` with `reputational_harm`.
5. **`multi`**: two or more companies are each discussed with their own events or opinions:
   round-ups ("Midday Movers: GILD, BSX, AES & More", 2214), comparisons ("LH or ALGN: which is
   the better value stock", 303163), stories about several firms. Every analysed company gets a
   label.
6. **`single`**: one company is the subject. If other companies are named for context only they go
   in `mentions` (D23: a company holding more than half of the subject mentions makes the body
   *single + mentions*).

**Analysed or mention?** A company is *analysed* when the article reports an event, result or
opinion about it of at least a sentence; it is a *mention* when it is named as context, a
competitor, a source ("Goldman analysts say"), or an index member. **[owner call]** For a
round-up of many companies, whether every company of a list of ten is analysed (the rule
agreed on 2026-10-29: every analysed company is labelled) or only those with a sentence of
their own; the second reading keeps the labelling effort bounded.

## The labels (per analysed company)

Nine existing labels (full tag-when / do-not-tag-when in `docs/category-taxonomy.md`):
`earnings_performance`, `mergers_acquisitions`, `leadership_governance`, `legal_regulatory`,
`product_innovation`, `capital_shareholder_returns`, `labor_human_capital`,
`market_analyst_sentiment`, `partnerships_business_dev`.

Four new labels, from the owner's v1 notes:

- **`reputational_harm`**: an event that damages (or repairs) the company's public standing,
  with no legal or earnings event as the story: apologies, service failures that anger customers,
  controversies, boycotts, pledges after criticism. *Examples:* 436001 (Southwest apologises for
  cancellations), 171583 (Coinbase customers demand refunds after a stablecoin glitch), 296753
  (JPMorgan's pledge to anti-hate groups). *Not:* a lawsuit or fine (`legal_regulatory`), layoffs
  (`labor_human_capital`).
- **`competitive_actions`**: moves and statements against or about a competitor, market-share
  fights, rival facilities, pricing pressure. *Examples:* 117163 (Apple rejects an app that
  criticises its 30% cut), 411216 (a CEO says his company is "crushing" a rival), 319695 (a new
  facility in a rival's backyard). *Not:* a joint effort (`partnerships_business_dev`), a rating
  (`market_analyst_sentiment`).
- **`investor_activity`**: what investors do with the company's stock: a fund or famous investor
  buying or selling a stake, activist campaigns, 13F filings, large block trades. *Not:* the
  company's own buyback or dividend (`capital_shareholder_returns`), an executive's insider trade
  (`leadership_governance`), an analyst rating (`market_analyst_sentiment`). **[owner call]** the
  boundary with `market_analyst_sentiment` for "investor X is bullish on the company".
- **`peer_comparison`**: the article's point is a comparison between companies (which is the
  better stock, who is ahead). In a `multi` comparison, every compared company is labelled
  `peer_comparison`. *Examples:* 96481 (Google vs Amazon), 303163 (LH or ALGN).

Three more:

- **`other`**: the body concerns the company but fits none of the 13 labels above.
- **`uncertain`**: a label cannot be chosen with confidence (category-level only; the scope is
  never uncertain, D20).
- (`unrelated` is a scope, not a company label.)

**Which label when two apply?** Take the dimension the headline event belongs to; an analyst's
reaction to that event does not change it (the reaction is `market_analyst_sentiment` only when
the reaction is the story).

## Sector bodies

Pick the GICS **sub-industry** the article is about, as spelled in the source `articles` table
(`gics_sub_industry`). The check rejects names the source does not have. **[owner call]** For a
broad sector story that spans several sub-industries, whether to choose the most central one or
use a sector-level value (the source has a sector column; the plan tests sub-industry only, D24).

## Open owner calls (summary)

1. The non-financial human-interest articles (`investing_general`, `unrelated` or `single`).
2. Round-ups: every listed company analysed, or only those with a sentence of their own.
3. `investor_activity` vs `market_analyst_sentiment` for an investor's bullish call.
4. Broad sector stories: central sub-industry or a sector-level value.

The checker (`news_nlp/retag.py`) enforces the shape above; once these four are answered the
rules are aligned to them and PR #117 merges.
