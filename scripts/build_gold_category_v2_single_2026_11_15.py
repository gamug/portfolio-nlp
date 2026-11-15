# ruff: noqa: E402  -- one-shot research script
"""Writes ``gold/category_v2_single.csv``: the 225 older single-label rows read again with the rules of the 1,000-row sheet (T-187, owner 2026-11-15).

    uv run scripts/build_gold_category_v2_single_2026_11_15.py [--force]

One line per row of ``category_single_cv_2026_11_15.rows()`` with source ``v2`` (in that order): ``row`` (position), ``article_id``,
``company``, ``label_old`` (the tag before the review, the 14 labels read through the merge), ``label`` (the tag after; empty = the row
is dropped, because an (article, company) pair that carried two tags keeps one) and ``why``. The rules are the sheet's: an analyst rating
or price target is ``market_analyst_sentiment``; results with guidance, a delay or a charge are ``earnings_performance``; a regulator
acting on the company (investigation, ruling, fine) is ``legal_regulatory``; an index change or a blurb about a rate, a discount or a
stock move stays ``other``; a deal, a bid or a break-up is ``mergers_acquisitions``; a launch is ``partnerships_business_dev`` when the
partner is the story and ``product_innovation`` when the product is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import category_single_cv_2026_11_15 as cv

OUT = ROOT / "gold" / "category_v2_single.csv"
MAS, EP, LR, MA, PI = (
    "market_analyst_sentiment",
    "earnings_performance",
    "legal_regulatory",
    "mergers_acquisitions",
    "product_innovation",
)
PB, LH, LG, RH, CS, IA = (
    "partnerships_business_dev",
    "labor_human_capital",
    "leadership_governance",
    "reputational_harm",
    "capital_shareholder_returns",
    "investor_activity",
)
# row -> (new tag, why); a row not listed keeps its tag. None = dropped (the same article and company carried a second tag).
CHANGES: dict[int, tuple[str | None, str]] = {
    2: ("other", "a spin-out of Alphabet raises money; not a harm to Alphabet"),
    12: (None, "same article and company as row 11 (product_innovation kept)"),
    17: ("other", "profile of a start-up led by a former Amazon executive"),
    23: (MAS, "upgrade to buy with a price target"),
    26: (None, "same article and company as row 25 (reputational_harm kept)"),
    32: (PI, "iPhone production cut at the Zhengzhou plant; the analyst only comments"),
    33: (None, "same article and company as row 32"),
    50: ("other", "the ADP payroll report is a macro data release, not news about the company"),
    54: (MAS, "an analyst's research note is the story"),
    57: (MA, "Berkshire plans to buy the rest of Wesco"),
    65: (MAS, "upgrade to buy"),
    80: (PI, "787 delivery date"),
    87: (MA, "shareholders approve the sale of Fox assets; China approval is pending"),
    88: (MAS, "downgrade"),
    92: (EP, "results with a share move"),
    93: (EP, "results with a share move"),
    94: (EP, "results with a share move"),
    95: (EP, "results with a share move"),
    96: (EP, "results with a share move"),
    97: (EP, "results with a share move"),
    98: (EP, "results with a share move"),
    99: (EP, "results with a share move"),
    100: (EP, "loss per share and job cuts: results"),
    101: (LR, "federal regulators open an investigation"),
    106: (MA, "rejects a takeover bid"),
    107: (LH, "accusation about employee work hours"),
    109: ("other", "closing a country business; not a deal"),
    110: (IA, "the Treasury sells the warrants it holds"),
    111: (EP, "cyberattack weighed on sales and profit"),
    113: (LR, "SEC enforcement rumour drives the drop"),
    115: (LG, "new president and COO"),
    116: (EP, "profits earned in Venezuela cannot be converted"),
    119: (EP, "comparable sales guidance below forecasts"),
    124: (MAS, "upgrade to buy"),
    125: (MAS, "upgrade to buy"),
    126: (PI, "Amazon tests its own delivery service"),
    128: (LG, "proxy fight (same rule as row 49)"),
    130: (PB, "nearing a deal to develop an urban area"),
    131: ("other", "a coverage decision on one drug"),
    132: ("other", "index membership change"),
    133: ("other", "a company profile, not results"),
    134: (MA, "break-up of the business"),
    150: (CS, "IPO (same rule as the HCA IPO, row 154)"),
    153: (None, "same article and company as row 152"),
    157: (None, "same article and company as row 156"),
    160: (EP, "quarterly profit fell on legal costs"),
    162: (EP, "first-quarter results will be hurt"),
    164: (LH, "two employees dismissed"),
    174: (MAS, "downgrade"),
    176: (None, "same article and company as row 175"),
    185: ("other", "a CEO joins a state visit"),
    189: (None, "same article and company as row 188"),
    193: ("other", "a customer policy"),
    204: (EP, "outlook for chip demand"),
    221: (None, "same article and company as row 222"),
    222: (LG, "CEO steps down from the Fed advisory council"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and not a.force:
        sys.exit(f"{OUT} exists; --force to overwrite")
    df = cv.rows_old()
    out = []
    for i, r in df.reset_index(drop=True).iterrows():
        new, why = CHANGES.get(int(str(i)), (r.gold[0], "unchanged"))
        out.append(
            {
                "row": i,
                "article_id": r.article_id,
                "company": r.company,
                "label_old": r.gold[0],
                "label": new or "",
                "why": why,
            }
        )
    pd.DataFrame(out).to_csv(OUT, index=False)
    t = pd.DataFrame(out)
    print(
        f"{len(t)} rows; changed {(t.label != t.label_old).sum()} (dropped {(t.label == '').sum()})"
    )
    print(t[t.label != ""].label.value_counts().to_string())


if __name__ == "__main__":
    main()
