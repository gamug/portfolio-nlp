# Labelling specification v2 (Work item 18, T-146a)

**Status: settled with the owner on 2026-10-29.** Example ids are `articles.id` values of the
SOURCE store (each checked to exist there on 2026-10-29).

It is the guide for the re-tag of the 720 articles (gold v2, T-146b), for the revised LLM
labelling prompt (T-146c) and for the held-out gold (T-146e). Rules marked **[owner call]** were
decided by the owner. It builds on D11–D24 (`TASKS.md` T-143). The older 9-label definitions with
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

Decide the **headline event** first (what the article is about), then take the first scope in
this order whose test the headline event meets. A market or industry framing does not pull an
article away from `single` unless the market or the industry is the point (see scopes 2, 3 and
6).

1. **`unrelated`**: not about business, finance or the economy (health news without a company
   angle, politics, lifestyle, career tips). *Examples:* 414317 (Israeli Covid study),
   286933 (House Intel Committee), 91934 (a job-interview question). A company named only as the
   setting does not make an article related (**[owner call 1]**: 61309, squatters in an Airbnb
   condo, is `unrelated`). Flagged and skipped by every later stage (D16).
2. **`market_economy`**: about the market as a whole or the economy: indices, S&P 500, rates,
   inflation, tariffs, strategist calls on the whole market. No company analysis. *Examples:*
   130093 (economic recovery), 480248 (a strategist's call for a double-digit rally), 296504
   (stocks rally; a strategist says to buy the dips). A company event that also moves the market
   follows the headline event: 164613, "Caterpillar downgrade pressures the overall stock
   market", is `single` with `market_analyst_sentiment` when the downgrade is the story and
   `market_economy` when the market move is. Summarised in the `MARKET` / `ECONOMY` rows of
   `sector_summary` (D14).
3. **`sector`**: about one industry or sub-industry rather than a company: sector trends, an
   industry-wide event, a sector's outlook; companies appear as examples. Needs a `sub_industry`.
   Test: if the named company were replaced by another of the same industry, would the article
   still hold? Only then is it `sector`. *Example:* 262959 (what the biggest tech IPO in years
   signals for investment banks). A story whose headline event is one company's own (115964,
   TSMC's revenue surge "bucking" the chip slump; 137874, a bankruptcy that might rescue Bank of
   America; 323392, an upgrade of one spirits stock) is **`single`** even when framed by the
   industry; the other companies go in `mentions`.
4. **`investing_general`**: investing or market-advice content about no company and no sector:
   investing philosophy, strategy pieces, personal-finance and "how to invest" pieces. *Example:*
   263795 (a bank's call for a new investing approach). No per-company analysis; summarised in
   the `MARKET` row (D22).
5. **`multi`**: two or more companies are each discussed with their own events or opinions
   (**[owner call 5, 2026-10-29]** also when they all belong to one industry: "three chip stocks
   to watch" is `multi`; it is `sector` only when the industry itself is the point and the
   companies are examples): round-ups ("Midday Movers: GILD, BSX, AES & More", 2214),
   comparisons ("LH or ALGN: which is the better value stock", 303163), stories about several
   firms. Every analysed company gets a label.
6. **`single`**: one company is the subject. If other companies are named for context only they go
   in `mentions`. (The automatic rules approximate this with D23's "more than half of the subject
   mentions"; that count is not the gold definition, see below.)

**Analysed or mention?** (The content rule governs the gold, **[owner call, 2026-10-29]**; D23's
"more than half of the subject mentions" is only a feature of the automatic rules, not the gold
definition.) A company is *analysed* when the article reports an event, result or
opinion about it of at least a sentence; it is a *mention* when it is named as context, a
competitor, a source ("Goldman analysts say"), or an index member. **[owner call, 2026-10-29]** In a
round-up of many companies, only a company with a sentence of its own is analysed (and so
labelled); names in a bare list are mentions. Every analysed company is labelled.

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
- **`competitive_actions`**: a company's own move or statement against or about a competitor,
  market-share fights, rival facilities, pricing pressure. *Examples:* 117163 (Apple rejects an
  app that criticises its 30% cut), 411216 (a CEO says his company is "crushing" a rival), 319695
  (a new facility in a rival's backyard). *Not:* a joint effort (`partnerships_business_dev`), a rating
  (`market_analyst_sentiment`).
- **`investor_activity`**: what investors do with the company's stock: a fund or famous investor
  buying or selling a stake, activist campaigns, 13F filings, large block trades. *Not:* the
  company's own buyback or dividend (`capital_shareholder_returns`), an executive's insider trade
  (`leadership_governance`), an analyst rating (`market_analyst_sentiment`). **[owner call 3]**
  `investor_activity` only when a stake moves (buy, sell, hold, activist position); an
  investor's bullish or bearish view alone is `market_analyst_sentiment`.
- **`peer_comparison`**: the article's point is a comparison between companies (which is the
  better stock, who is ahead). In a `multi` comparison, every compared company is labelled
  `peer_comparison`, not `market_analyst_sentiment` (v1 used the latter for comparisons; a
  rating or price target on one company stays `market_analyst_sentiment`). **[owner call 6]** A
  single-company body whose point is a comparison with its sector or a peer group (illustrative:
  "is Nvidia cheap against chips?") is `single` with `peer_comparison`; the peers go in
  `mentions`. A peer that is named and has a sentence of its own is analysed under the content
  rule (not a separate owner call), so the body is then `multi` and every such company is
  labelled `peer_comparison`. Versus `competitive_actions`: `peer_comparison` is a writer or
  analyst weighing companies against each other; `competitive_actions` is something a company
  does or says (411216, a CEO saying his firm is "crushing" a rival). *Examples:* 96481
  (Google vs Amazon), 303163 (LH or ALGN).

Three more:

- **`other`**: the body concerns the company but fits none of the 13 labels above.
- **`uncertain`**: two or more labels fit equally and the headline event does not decide between
  them, or the event is not stated in the first 6,000 characters (category-level only; the scope
  is never uncertain, D20). Example: a profile of a company that mixes strategy, products and
  results with no lead event. Contrast `other`: the event is clear but belongs to none of the 13
  labels (e.g. a CEO's opinion piece on the economy).
- (`unrelated` is a scope, not a company label.)

**Which label when two apply?** Take the dimension the headline event belongs to; an analyst's
reaction to that event does not change it (the reaction is `market_analyst_sentiment` only when
the reaction is the story).

## Hints from tags v1

The v1 tag `other` carries no scope: the 78 articles whose `v1_scope_derived` is
`investing_general` are a mix (263795 is `investing_general`; 95691, Cramer on how Salesforce can
rival Amazon, is `single` with `competitive_actions`; 88609, a spinoff raising money, is
`single`), so re-scope every one on its content.

## Sector bodies

Pick the GICS **sub-industry** the article is about, as spelled in the source `articles` table
(`gics_sub_industry`). The check rejects names the source does not have; the list of valid names,
with their sector, is `data/gics_sub_industries.csv`, written by `scripts/retag_gold_2026_10_29.py
export` (PR #117) together with `data/category_gold_v2_guide.csv`, a CSV of the scopes and labels
of this document. **[owner call, 2026-10-29]** For a
broad sector story that spans several sub-industries, enter the most central sub-industry (the plan
tests sub-industry, D24); there is no sector-level value.

## Owner calls (answered 2026-10-29)

1. Human-interest stories that only use a company as setting: `unrelated`.
2. Round-ups: only companies with a sentence of their own are analysed; the rest are mentions.
3. `investor_activity` only when a stake moves; an investor's view alone is `market_analyst_sentiment`.
4. Broad sector stories: the most central sub-industry.
5. A round-up of firms from one industry is `multi`; `sector` only when the industry is the point.
6. A one-company comparison against its sector or peers is `single` with `peer_comparison`.

Also settled by the owner and stated above: the headline event decides the scope (a company framed
by its industry or moving the market stays `single` unless the industry or the market is the
point), and the content rule, not D23's count, decides analysed vs mention in the gold.

The checker (`news_nlp/retag.py`, PR #117) enforces the shape above.
