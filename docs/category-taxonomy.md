# News article category taxonomy

## Purpose

`news_nlp`'s category stage classifies every article into one of 10 fixed
categories representing a *dimension of company performance* (earnings,
M&A, leadership, etc.), so that sentiment produced by the sentiment stage
can be attributed to the right dimension rather than treated as one
undifferentiated "sentiment about the company" number. This is intended as
a reusable controlled vocabulary for the not-yet-built knowledge-graph
module downstream of `news_nlp` (see `README.md`'s pipeline diagram), not
just an internal detail of the category stage — hence living at the top
level of `docs/`, not nested under `docs/modules/`.

Zero-shot NLI classification (the technique used here — see
`src/pipeline.py`'s `run_category_stage`) needs a fixed
`candidate_labels` list; it doesn't invent categories, it only picks the
best-fitting one(s) from a list supplied at inference time. So this
taxonomy had to be designed up front, from real published sources, rather
than left for the model to discover.

## Sources

The 9 substantive categories below were triangulated across four
independent, real-world financial-news/company-event taxonomies, so no
single source's idiosyncrasies dominate the result:

1. **RavenPack News Analytics taxonomy** —
   https://www.ravenpack.com/technology/classification — a financial-news
   analytics vendor's proprietary event taxonomy. Categories drawn from:
   acquisitions & mergers, analyst ratings, credit, credit ratings,
   dividends, earnings, equity actions, insider trading, labor issues,
   legal, marketing, partnerships, price targets, products/services,
   regulatory, revenues, sanctions, stock picks, stock price.
2. **SASB Materiality Map** — https://sasb.ifrs.org — the Sustainability
   Accounting Standards Board's 5 primary company-performance dimensions
   (Environment, Social Capital, Human Capital, Business Model &
   Innovation, Leadership & Governance) and their 26 underlying General
   Issue Categories. Used for the categories with no direct RavenPack
   analogue: Leadership & Governance, Business Model & Innovation, Human
   Capital.
3. **IPTC Media Topics** —
   https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html —
   the news industry's standard story-classification taxonomy, maintained
   by the International Press Telecommunications Council. Its "Economy,
   Business and Finance" branch covers corporate earnings, stock buyback,
   corporate dividends, bankruptcy, business restructuring,
   layoffs/downsizing, mergers/acquisitions — used to corroborate category
   boundaries against how the news industry itself already classifies this
   content.
4. **Refinitiv/Thomson Reuters News Analytics (TRNA) topic codes** — a
   financial-data vendor's real-time news topic codes, e.g. `MRG` (Mergers
   & Acquisitions), `RES`/`RESF` (Results/Results Forecast), `DIV`
   (Dividends), `RCH` (Research/analyst) — used as a cross-check that the
   category boundaries match how a major financial-news feed already tags
   stories.

## The taxonomy

`news_nlp.taxonomy`'s `CATEGORY_LABELS` is the canonical machine-readable
source for this table — keep this doc in sync with it if the labels ever
change. `CATEGORY_GROUPS` (see "Hierarchical classification" below) groups
these same 9 slugs into 3 classification groups; it doesn't add or remove
any category here.

| slug | display name | scope | sources |
|---|---|---|---|
| `earnings_performance` | Earnings & Financial Performance | Quarterly/annual results, guidance, revenue/profit figures | RavenPack (earnings, revenues), Refinitiv (RES/RESF), IPTC (corporate earnings) |
| `mergers_acquisitions` | Mergers & Acquisitions | Deals, takeovers, divestitures | RavenPack (acquisitions & mergers), Refinitiv (MRG), IPTC (mergers/acquisitions) |
| `leadership_governance` | Corporate Leadership & Governance | Executive appointments/departures, board actions, insider trading | SASB (Leadership & Governance), RavenPack (insider trading) |
| `legal_regulatory` | Legal & Regulatory | Litigation, regulatory action, sanctions, bankruptcy | RavenPack (legal, regulatory, sanctions), IPTC (bankruptcy, business restructuring) |
| `product_innovation` | Product & Innovation | Product launches, R&D, technology | SASB (Business Model & Innovation), RavenPack (products/services) |
| `capital_shareholder_returns` | Capital Actions & Shareholder Returns | Dividends, buybacks, debt/credit actions | RavenPack (dividends, equity actions, credit/credit ratings), IPTC (stock buyback, corporate dividends) |
| `labor_human_capital` | Labor & Human Capital | Layoffs, hiring, workforce/labor relations | SASB (Human Capital), RavenPack (labor issues), IPTC (layoffs and downsizing) |
| `market_analyst_sentiment` | Market & Analyst Sentiment | Analyst ratings/price targets, stock price moves | RavenPack (analyst ratings, price targets, stock picks, stock price), Refinitiv (RCH) |
| `partnerships_business_dev` | Partnerships & Business Development | Strategic alliances, joint ventures, distribution deals | RavenPack (partnerships, marketing) |
| `other` | Other | Catch-all — anything below the confidence threshold against every category above | n/a — the below-threshold fallback |

## Hierarchical classification (2026-09-09)

The classifier does NOT run all 9 substantive labels against each other in
one softmax. It used to (a flat 9-way softmax), but that empirically starved
real signal for several labels: with 9 mutually-exclusive candidates, the
no-signal uniform baseline is only ~0.11, and `product_innovation`
(recall 0.209 at n=359 true examples in an eval run), `partnerships_business_dev`
(0.224, n=170), and `leadership_governance` (0.282, n=85) were predicted
`other` on 48-74% of their true instances — on those misses the model's own
raw score for the *correct* slug averaged only 0.14-0.16, barely above the
0.111 baseline (i.e. genuinely no signal, not a near-miss on the threshold —
under 5% of misses were even ≥0.3). Sampled judge rationales confirmed these
were unambiguous articles (a product launch, clinical trial data, a
cloud-strategy piece), not genuine edge cases — the model just couldn't
produce a confident signal while competing against 8 other candidates at
once.

Fix: classify in two smaller softmax passes instead of one big one. `news_nlp.taxonomy.CATEGORY_GROUPS`
groups the 9 slugs into 3 groups of 3, each mixing at least one strong and
one weak performer from the numbers above:

| group slug | children |
|---|---|
| `corporate_actions` | `earnings_performance`, `mergers_acquisitions`, `capital_shareholder_returns` |
| `governance_legal_workforce` | `leadership_governance`, `legal_regulatory`, `labor_human_capital` |
| `market_product_partnerships` | `product_innovation`, `market_analyst_sentiment`, `partnerships_business_dev` |

1. **Level 1**: a 3-way softmax over the 3 groups' own hypotheses
   (uniform baseline ~0.33, ~3x higher signal-to-noise than the old flat
   design). If the winning group's score is below `CATEGORY_GROUP_FLOOR`
   (**0.40**, ~1.2x that baseline), the distribution is treated as flat/
   uninformative and the article is labeled `other` directly — level 2 is
   skipped entirely for it, and every leaf score column is `0.0`.
2. **Level 2** (only for articles that didn't short-circuit): the **top-2**
   groups from level 1 — not just the winner — combine their 6 children into
   one candidate set for a second softmax (baseline ~0.17). The final
   `label` is that softmax's winner if it clears
   `pipeline.CATEGORY_CONFIDENCE_THRESHOLD` (**0.6** as of the 2026-09-09
   calibration below — ~3.6x its own 6-way baseline, back in line with the
   original flat 9-way design's own ~3.6x ratio over *its* 0.111 baseline;
   launched at 0.4/~2.4x, see why that changed below), else `other`.
   Using the *top-2* groups, not just the top-1, means a narrow level-1 miss
   can still be recovered at level 2 as long as the true group was 2nd
   place — a strict single-path cascade could never recover from that.

**What gets persisted**: `article_category` gains `group_label`/`group_score`
(the winning level-1 group and its probability, set even when the final
`label` is `other`), alongside the same 9 leaf-slug score columns as before.
The 3rd-place group's 3 children — the group that didn't make an article's
top-2 — never get a level-2 forward pass at all, so their columns are `0.0`.
**This `0.0` means "not evaluated," not "confidently rejected"** — don't
read it as the model having ruled that category out; it just never got
asked. The same is true, for all 9 leaf columns, on the flat-level-1
short-circuit path.

### Threshold calibration (2026-09-09)

The launch value of `CATEGORY_CONFIDENCE_THRESHOLD` (0.4, ~2.4x the level-2
baseline) was a starting estimate, explicitly flagged as needing real data.
The first post-hierarchy eval run (2800 rows, eval_run 19 / mlflow
`8c470ea1`) confirmed the target categories improved dramatically —
`product_innovation` recall 0.209→0.613, `partnerships_business_dev`
0.224→0.510, `leadership_governance` 0.282→0.526 — but `acc_other` collapsed
from ~0.80-0.83 (previously the *best*-performing class) to **0.215** (the
*worst*): 701 of 1,322 judge-confirmed true-`other` articles got assigned a
specific wrong label instead.

Checked before changing anything: those 701 were not near-threshold misses
(mean/median winning score 0.661/0.641, only 22.7% even close to 0.4) —
reducing per-decision competition to let real signal surface for weak
categories also let spurious signal surface for genuinely generic/ambiguous
articles that the old, stricter 9-way contest used to correctly route to
`other`. Also checked level 1: even *correctly*-resolved true-`other`
articles clear `CATEGORY_GROUP_FLOOR` comfortably (mean group_score 0.50),
so the problem lived at level 2's threshold, not level 1's floor (left
unchanged at 0.40).

Raised `CATEGORY_CONFIDENCE_THRESHOLD` to **0.6**: at that level, 301/701
(43%) of the false-`other` losses resolve correctly, at a cost of only
8-18% of the newly-won true-positive recall on the three target categories
(their correctly-labeled scores cluster far higher, mean ~0.80) — a good
trade, not a coin flip, but still a reasoned calibration point rather than a
fully validated one. Retune again using `article_category`'s stored
per-label score distribution once more post-calibration data exists.

This changes what gets written for *future* pipeline runs only — it does not
retroactively reclassify already-scored articles (`run_category_stage` only
processes rows absent from `article_category`). A bulk re-classification of
existing rows, if ever wanted, is a separate, not-yet-built follow-up (no
reusable bulk-backfill script currently exists in `scripts/`).

### Per-slug precision/recall, and why precision is the metric that matters (2026-09-09)

Full methodology, the per-slug table, and the `other`-bucket caveat live in
`docs/evaluation.md`'s "corrected post-calibration numbers" / "Why
precision, not recall, for category" / "The `other` bucket" sections
(eval_run 21, mlflow `0a18577e`) — summarized here because it bears directly
on how to read this taxonomy's categories in practice:

- **Precision matters more than recall for this stage.** Unlike sentiment
  (where a missed negative is a blind spot with no fallback), every article
  that isn't confidently a specific category already has a safe one:
  `other`. A model too cautious about a slug just under-fills that slug —
  recoverable. A model too eager actively mislabels an article with a
  specific, actionable-sounding wrong category — not recoverable by a
  downstream consumer reading `article_category.label`. So "when the model
  commits to a label, is it right" (precision) is the number that matters
  most per slug, not "did it catch every instance" (recall).
  `capital_shareholder_returns` (precision 0.074) and `mergers_acquisitions`
  (precision 0.926, recall 0.481 — conservative, not wrong) are the two
  slugs furthest apart on this axis right now.
- **`other` is not yet a trustworthy "no category" signal.** Its own
  precision is 0.474 — over half the time the model says `other`, the judge
  says there was a real category. The failure *direction* still matches the
  business preference (no false specific claim reaches a consumer), but
  `other` today reads more like "not confident enough to commit" than
  "verified no relevant category" — anything downstream filtering out
  `other` rows as irrelevant is discarding a lot of real hits along with it.

## Classification input

The classifier runs on the article's title plus the lead chunk of its body
(not the full article, and not the opt-in `article_summary` — see
`docs/modules/news-nlp.md` for why). News articles are inverted-pyramid, so
the opening sentences almost always establish the dominant topic. Unaffected
by the hierarchical redesign above — both levels classify the same premise,
just against different (smaller) hypothesis sets.
