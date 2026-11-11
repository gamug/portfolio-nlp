"""Step 1.2, the cut: the ``units`` extractor turns a ``multi`` body into the texts of its companies (PLAN D34 (e), D40; T-163).

The extractor is the one the owner selected on 2026-11-08 (``reading="units"`` of ``news_nlp.attribution_model``): the body
without its footer is cut at lines, bullets and clauses; a piece that names a company goes to it, and the pieces that name
none go to the company the trained model chooses, or to nobody. Each company's text is one sub-news. The sub-news does not
store the ticker (the schema of ``sub_news`` has no such column): the company is found again downstream, by NER and the
matcher (D39 (iii)).

The revision of a cut is ``units-`` plus 12 hex digits of the SHA-256 of: the model directory's identity, the company names
the model is asked about, the model's threshold and the revision of the code that cuts. Any of them changing starts new rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from news_nlp import attribution_model as am
from news_nlp import retag
from news_nlp.db import NewsNlpDatabase
from news_nlp.triage import Company, CompanyResolver

READING: am.Reading = "units"


def companies_of(conn: NewsNlpDatabase) -> list[Company]:
    """The company universe, as the SOURCE store holds it (the same query as ``retag_gold`` uses for the gold)."""
    rows = conn.execute(
        f"SELECT DISTINCT ticker, company, gics_sector, gics_sub_industry FROM {conn.articles_rel}.articles "  # noqa: S608 -- main or source
        "WHERE fetch_status = 'ok' ORDER BY ticker, company"
    ).fetchall()
    return [Company(r[0], r[1] or "", r[2] or "", r[3] or "") for r in rows]


def revision_of(model_identity: str, names: dict[str, str]) -> str:
    """The revision of a cut: the model, the names it is asked about, its threshold and the code that cuts."""
    code = retag.code_revision(*am.UNITS_CODE_MODULES)
    names_hash = hashlib.sha256(json.dumps(sorted(names.items())).encode()).hexdigest()[:8]
    key = f"{model_identity}@names:{names_hash}@t{am.THRESHOLD}@{code}"
    return "units-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def distinct(texts: Sequence[str]) -> list[str]:
    """The texts in order without repeats: two companies that only share the same sentences get one sub-news."""
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


class UnitsExtractor:
    """The selected extractor for a given company universe and scorer; ``cut`` returns the texts of one body."""

    def __init__(
        self, scorer: am.Scorer, companies: Sequence[Company], model_identity: str
    ) -> None:
        self.scorer = scorer
        self.resolver = CompanyResolver(companies)
        self.names = {c.ticker: c.name for c in companies}
        self.revision = revision_of(model_identity, self.names)

    @classmethod
    def load(cls, model_dir: Path, companies: Sequence[Company]) -> UnitsExtractor:
        """Load the trained attribution model of ``model_dir`` (a local folder, identified by the hash of its files)."""
        identity = f"local:{am.directory_identity(model_dir)}"
        return cls(am.TransformerScorer(str(model_dir)), companies, identity)

    def cut(self, body: str) -> list[str]:
        """The text of each company the body names, in order of first appearance, without repeats."""
        texts = am.model_company_texts(
            body, self.resolver, self.names, self.scorer, reading=READING
        )
        return distinct(list(texts.values()))
