# Text sent to a summariser by population (2026-11-15)

Only the texts over 2,000 characters (the size rule of step 2). `uv run scripts/summary_chars_to_summarise_2026_11_15.py ...` (read-only).

| population | items | characters | mean | longest |
|---|---|---|---|---|
| single | 59,432 | 235,078,518 | 3,955 | 84,177 |
| sub_news | 5,048 | 15,263,579 | 3,023 | 19,773 |
| sector | 14,803 | 60,503,969 | 4,087 | 41,759 |
| **all** | **79,283** | **310,846,066** | 3,920 | 84,177 |

Tokens: 4.31 characters per input token (measured on the 300 articles of the experiment), 85 tokens of prompt and 87 of output per call: **78.9M input and 6.9M output**.

| rates (per million tokens: input / output) | cost |
|---|---|
| low: $0.15 / $0.60 | **$16.0** |
| high: $0.30 / $1.20 | **$31.9** |

The rates are those of the vendor's pricing page of 2026-11-15, off-peak (low) and peak (high), cache-miss input; confirm them on the page before a run.
