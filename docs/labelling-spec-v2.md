# Labelling specification v2 (Work item 18, T-146a)

**Status: settled with the owner on 2026-10-29.** Example ids are `articles.id` values of the
SOURCE store (each checked to exist there on 2026-10-29).

It is the guide for the re-tag of the 720 articles (gold v2, T-146b), for the revised LLM
labelling prompt (T-146c) and for the held-out gold (T-146e). Rules marked **[owner call N]** were
decided by the owner (the numbered list at the end). It builds on D11–D24 (`TASKS.md` T-143). The older 9-label definitions with
their close calls are in `docs/category-taxonomy.md`.

## What is tagged

One row per **unique article body** (D11). The title and the first 6,000 characters are what is
read (what the judge and the first tagging saw). Columns of the re-tag file
(`data/category_gold_v2_review.csv`):

| column | content |
|---|---|
| `scope` | one of the six scopes below (always filled) |
| `company_labels` | the **analysed** companies, each with its D15 label, separated by commas: `AAPL:product_innovation, MSFT:other` (single: exactly one entry; multi: two or more; the other scopes: none). There is no separate companies column: the analysed companies are the labelled ones. Tickers, or names and nicknames as the article writes them: `scripts/retag_gold_2026_10_29.py normalise` converts them to tickers before `check` (a comma inside a name such as `Booking Holdings, Inc.` is kept). A company outside the 509 of the universe (Macy's, Twitter, Toyota) is written by name as the article writes it: it is kept as typed, counts as analysed (and for the scope) like any other, and `check` lists these names so that a typo is noticed (**[owner call 9]**); an upper-case text of up to five characters is read as a ticker and must exist |
| `mentions` | companies named but not analysed (D23), separated by commas (names or tickers); in every scope, including `sector`, `market_economy` and `investing_general` (**[owner call 7]**) |
| `sub_industry` | a GICS sub-industry, as its 8-digit code or its name (`data/gics_sub_industries.csv` lists both): required for a sector body, **optional on every other scope** (**[owner call 8]**) |
| `notes` | anything that helps (never required) |

`spanish_title` and `spanish_text` follow `text`: a Spanish reading aid (finance jargon translated, the English term in parentheses the first time), made by the LLM from the same 6,000 characters (the Spanish text therefore stops where the English one is cut). It is never an input of a model and never in the key file. **Every file the owner has to tag carries them.** The tagging rule is always the English article.

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
   (**[owner call 5]** also when they all belong to one industry: "three chip stocks
   to watch" is `multi`; it is `sector` only when the industry itself is the point and the
   companies are examples): round-ups ("Midday Movers: GILD, BSX, AES & More", 2214),
   comparisons ("LH or ALGN: which is the better value stock", 303163), stories about several
   firms. Every analysed company gets a label.
6. **`single`**: one company is the subject. If other companies are named for context only they go
   in `mentions`. (The automatic rules approximate this with D23's "more than half of the subject
   mentions"; that count is not the gold definition, see below.)

**Analysed or mention?** (The content rule governs the gold, as the owner settled (closing paragraph); D23's
"more than half of the subject mentions" is only a feature of the automatic rules, not the gold
definition.) A company is *analysed* when the article reports an event, result or
opinion about it of at least a sentence; it is a *mention* when it is named as context, a
competitor, a source ("Goldman analysts say"), or an index member. **[owner call 2]** In a
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
  (a new facility in a rival's backyard). *Not:* a joint effort (`partnerships_business_dev`), a
  rating (`market_analyst_sentiment`).
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
  labels (e.g. a company's anniversary celebration or an office relocation). A CEO's opinion
  on the economy is not `other`: it is `market_economy`, with no company label.
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

Pick the GICS **sub-industry** the article is about, as its 8-digit GICS code (the table of
<https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard>, kept in
`news_nlp/gics_codes.py`) or as its name in the source `articles` table (`gics_sub_industry`). The
check rejects a code it does not know and a code or name no company of the universe has; the list of valid names,
with their sector, is `data/gics_sub_industries.csv`, written by `scripts/retag_gold_2026_10_29.py
export` together with `data/category_gold_v2_guide.csv`, a CSV of the scopes and labels
of this document. **[owner call 4]** For a broad sector story that spans several sub-industries,
enter the most central sub-industry (the plan tests sub-industry, D24); there is no sector-level
value. The companies a sector body names go in `mentions`, not in `company_labels` (**[owner call 7]**).

## Returning the file

1. Close `data/category_gold_v2_review.csv` in Excel before running anything (an open file cannot be
   rewritten, and a file saved in another format is refused). Save it as **CSV UTF-8, comma
   delimited**.
2. `uv run scripts/retag_gold_2026_10_29.py normalise` writes
   `data/category_gold_v2_review_tickers.csv` with tickers, lists the texts it could not resolve
   (fix them in the review file, not in the copy) and the matches it inferred (a first word, a
   brand, a ticker that is also a word): confirm those.
3. `uv run scripts/retag_gold_2026_10_29.py check --file category_gold_v2_review_tickers.csv`
   validates the copy.
4. At the end of each tagging session copy `data/category_gold_v2_review.csv` into `data/backup/`
   with the date in its name: no checker can detect a mis-save such as a sort that detaches the
   rows from their ids.
5. `guide` rewrites the guide CSVs at any time; `export` refuses to touch a tagged review file.
6. A file saved under another name needs `--file NAME` (`check`, `normalise`, `translate`). A file
   still in the first format (a `companies` column, `;` separators) is refused with the way out:
   `migrate --file NAME` writes a copy without `companies` and with commas, under a new name with
   the date and time (never over an earlier copy), keeps every company of the old column in
   `company_labels` (without a label, listed for you) and never touches the original. Work in
   one file only: tagging done in the old file after the copy is not in the copy.

## Owner calls (answered 2026-10-29)

1. Human-interest stories that only use a company as setting: `unrelated`.
2. Round-ups: only companies with a sentence of their own are analysed; the rest are mentions.
3. `investor_activity` only when a stake moves; an investor's view alone is `market_analyst_sentiment`.
4. Broad sector stories: the most central sub-industry.
5. A round-up of firms from one industry is `multi`; `sector` only when the industry is the point.
6. A one-company comparison against its sector or peers is `single` with `peer_comparison`.
7. Companies named in `sector`, `market_economy` and `investing_general` bodies are listed in
   `mentions`; `company_labels` stays empty there.
8. `sub_industry` (a code or a name) may be given on any scope, and is required only for a sector
   body (the owner's first tagged row, a `multi` body, carries one).
9. A company outside the 509 of the universe is recorded by name as typed (in `company_labels` and
   in `mentions`), counts for the scope and the label, and is never a ticker-level target.

Also settled by the owner and stated above: the headline event decides the scope (a company framed
by its industry or moving the market stays `single` unless the industry or the market is the
point), and the content rule, not D23's count, decides analysed vs mention in the gold.

The checker (`news_nlp/retag.py`) enforces the shape above.

## The fixed 240 of the prompt gate (T-146c)

The prompt gate runs on a fixed 240 of the 720 bodies; the other 480 wait until the 240 passes
(they are needed only for the confirmation run and the rules re-measure). Everything below is done
by `scripts/draw_gate240_2026_11_04.py`; the logic is `retag.draw_gate` / `gate_rows` /
`merge_tags`.

- **The set.** Quotas by the scope the v1 notes imply (`v1_scope_derived`, because gold v2 scopes
  do not exist before the owner tags): sector 9, market_economy 9, unrelated 31, investing_general
  40, multi 60, single 91 (`retag.GATE_QUOTAS`). The owner's already tagged rows come first, so the
  240 is stratified but not a random sample; the rest of each stratum is a seeded draw. The draw
  stops, writing nothing, if it is not 240 or a stratum holds more tagged rows than its quota.
- **Where it is recorded.** `data/gate240_ids.csv` is local and git-ignored; it carries the six
  strata, which no tracked file does. The only gold files git tracks are two, both added by #123
  (added by #123): the train set
  `gold/category_gold_v2.csv` (the 240 plus the 44 sector bodies, one row per article, `session`
  = `gate240` or `sector_extra`, tags only, no news text) and the held-out test set
  `gold/category_gold_v2_test.csv` (the 200 of T-146e, still to be tagged). Nothing else lives in
  `gold/`, and every later feature reads those two. The draw depends on which rows were tagged when
  it ran, so a redraw with the same seed on a later file would give another set: the train file, not
  the seed, is the record of its members. The gate is scored on the overall figure only (see
  T-146c), so the strata do not need a tracked home; per-scope recall is judged on the held-out test.
- **Tagging.** `data/gate240_to_tag.csv` is a tagging file of its own: the 240 rows in the review
  file's columns (the Spanish aid after `text`) with the tags already made. It is never
  overwritten once it holds rows.
- **Bringing the tags back.** The train gold is the source of truth (it is built from the gate file
  directly). `merge` makes a convenience copy of the same tags in the 720-row review file, which only
  the confirmation run and the rules re-measure read; run it with `--gate-wins` so the copy agrees
  with the gold. It copies the tags into the review file as a new
  `<review>_merged_<timestamp>.csv`; neither input is touched. **The newest `_merged_` file is the
  current review file**: `draw` and `merge` both start from it, so tags chain. A row counts as
  tagged when any tag column has content.
- **Conflicts.** A review row that already holds different tags keeps them and is listed with the
  differing columns; `merge` then exits 1 ("merged file NOT complete") and writes
  `<review>_conflicts_<timestamp>.csv` (review value, gate value, resolution). Because the owner
  tags in the gate file only, its tags are the later ones: after looking at that file, run
  `merge --gate-wins` and the gate file's tags replace the different ones (the same audit file
  records `took_gate`). Nothing is replaced without that flag.
- **Not covered.** Spreadsheet formula prefixes in titles are not neutralised (research tooling over
  public headlines, not a production service).
