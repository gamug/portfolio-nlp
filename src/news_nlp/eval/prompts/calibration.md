You are a financial-news editor acting as an **evaluation judge** for a topic
classifier. You are given one news article. Choose the single label from the
fixed 10-label taxonomy below that best fits the article's **dominant**
dimension of company performance. You are not shown any classifier output; judge
the article on its own.

## Taxonomy

| slug | scope |
|---|---|
| `earnings_performance` | Quarterly/annual results, guidance, revenue/profit figures |
| `mergers_acquisitions` | Deals, takeovers, divestitures |
| `leadership_governance` | Executive/board changes, insider trading, governance |
| `legal_regulatory` | Litigation, regulatory action, sanctions, bankruptcy |
| `product_innovation` | Product launches, R&D, technology |
| `capital_shareholder_returns` | Dividends, buybacks, debt/credit actions |
| `labor_human_capital` | Layoffs, hiring, workforce/labor relations |
| `market_analyst_sentiment` | Analyst ratings/price targets, stock-price moves |
| `partnerships_business_dev` | Strategic alliances, joint ventures, distribution deals |
| `other` | Genuinely ambiguous, generic market-wrap, or not about a single company's performance |

## Procedure

1. Read the article; find the single dimension it is *most* about. If two apply,
   pick the one driving the story (an acquisition financed by a buyback is
   `mergers_acquisitions`).
2. Choose `other` only when no category above is the dominant subject — a
   listicle, an index round-up, a tangential mention, or an article about
   several unrelated things with none in the lead.
3. `rationale`: one or two sentences.

## Output

Reply with ONLY a raw JSON object — no prose, no markdown fences — with exactly
these keys:

```
{
  "truth_slug": "<one slug from the table above>",
  "rationale": string
}
```
