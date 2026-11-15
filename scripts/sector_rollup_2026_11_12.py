"""Roll the owner's sector tags up to one level (2026-11-12).

Reads the `sector` bodies of the four tagged files in gold/, and writes data/sector_rollup.csv for the
owner's review: the original code, the sector (2 digits), the industry group (4 digits) when the tag
has at least that depth, and a status (ok, cross_sector, invalid). Codes are read as text.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from news_nlp.gics_codes import SUB_INDUSTRY_CODES  # noqa: E402

FILES = [
    "category_gold_v2.csv",
    "category_gold_v2_test.csv",
    "scope_enrich.csv",
    "scope_hook.csv",
]
GROUP_DIGITS = 4
SECTORS = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}


def valid(code: str) -> bool:
    return any(k.startswith(code) for k in SUB_INDUSTRY_CODES)


def main() -> None:
    rows = []
    for name in FILES:
        with open(ROOT / "gold" / name, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["scope"] != "sector":
                    continue
                raw = (r["sub_industry"] or "").strip()
                codes = [c.strip() for c in raw.split(",") if c.strip()]
                ok = bool(codes) and all(re.fullmatch(r"\d+", c) and valid(c) for c in codes)
                secs = sorted({c[:2] for c in codes}) if ok else []
                groups = sorted({c[:4] for c in codes if len(c) >= GROUP_DIGITS}) if ok else []
                status = "invalid" if not ok else ("cross_sector" if len(secs) > 1 else "ok")
                rows.append(
                    {
                        "file": name,
                        "article_id": r["article_id"],
                        "original": raw,
                        "sectors": ",".join(secs),
                        "sector_names": " | ".join(SECTORS.get(s, "?") for s in secs),
                        "groups": ",".join(groups),
                        "status": status,
                        "notes": r["notes"],
                    }
                )
    out = ROOT / "data" / "sector_rollup.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "sector bodies ->", out)
    print(Counter(r["status"] for r in rows))
    print(Counter(r["sectors"] for r in rows if r["status"] == "ok").most_common())


if __name__ == "__main__":
    main()
