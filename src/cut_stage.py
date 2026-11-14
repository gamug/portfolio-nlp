"""Step 1.2, the cut: the ``units`` extractor turns a ``multi`` body into the texts of its companies (PLAN D34 (e), D40; T-163).

The extractor is the one the owner selected on 2026-11-08 (``reading="units"`` of ``news_nlp.attribution_model``), in the form
``units5`` of T-177 (owner, 2026-11-12), which builds on ``units4`` of T-176 (a line is not given to another company, a header is not cut off its text): the body
without its footer is cut at lines, bullets and clauses; a piece that names a company goes to it, and the pieces that name
none go to the company the trained model chooses, or to nobody. Each company's text is one sub-news. The sub-news does not
store the ticker (the schema of ``sub_news`` has no such column): the company is found again downstream, by NER and the
matcher (D39 (iii)).

``units5`` (T-177) is the default reading since 2026-11-12; ``units4`` and ``units`` stay available with ``--reading``. ``units5`` also reads the headers ``units4`` misses (names in
capitals, group labels, names with a period) and gives a sentence after a header of nobody to the companies it names.

The revision of a cut is the reading (``units5`` by default, ``units4`` or ``units`` with ``--reading``) plus ``-`` and 12 hex digits of
the SHA-256 of: the model directory's identity, the company names the model is asked about, the model's threshold, the revision of
the code that cuts and, when it is not the default, the reading. Any of them changing starts new rows. Rows written before
T-176 are ``units-75642e3f4f72``-style: a different set, not rewritten, and a reader must name the revision it wants.
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

DEFAULT_READING: am.Reading = "units5"  # owner, 2026-11-12 (T-177); `units4` (T-176) was the default of 2026-11-11, `units` the selection of 2026-11-08 (T-148b10)
CUT_READINGS: tuple[am.Reading, ...] = (
    "units",
    "units4",
    "units5",
)  # `units` and `units4` are kept to cut as before


def companies_of(conn: NewsNlpDatabase) -> list[Company]:
    """The company universe, as the SOURCE store holds it (the same query as ``retag_gold`` uses for the gold)."""
    rows = conn.execute(
        f"SELECT DISTINCT ticker, company, gics_sector, gics_sub_industry FROM {conn.articles_rel}.articles "  # noqa: S608 -- main or source
        "WHERE fetch_status = 'ok' ORDER BY ticker, company"
    ).fetchall()
    return [Company(r[0], r[1] or "", r[2] or "", r[3] or "") for r in rows]


def revision_of(
    model_identity: str, names: dict[str, str], reading: am.Reading = DEFAULT_READING
) -> str:
    """The revision of a cut: the model, the names it is asked about, its threshold, the code that cuts and, when it is
    not the default reading, the reading."""
    code = retag.code_revision(*am.UNITS_CODE_MODULES)
    names_hash = hashlib.sha256(json.dumps(sorted(names.items())).encode()).hexdigest()[:8]
    key = f"{model_identity}@names:{names_hash}@t{am.THRESHOLD}@{code}"
    if reading != DEFAULT_READING:
        key += f"@{reading}"
    return f"{reading}-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def distinct(texts: Sequence[str]) -> list[str]:
    """The texts in order without repeats: two companies that only share the same sentences get one sub-news."""
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


PIN_PATH = Path(__file__).resolve().parent.parent / "config" / "attribution_model.json"


def folder_identity(model_dir: Path) -> str:
    """The identity of the folder (a hash of its files): computed once per run, the weights are large."""
    return am.directory_identity(model_dir)


def is_pinned(
    model_dir: Path, pin_path: Path | None = None, identity: str | None = None
) -> bool | None:
    """Whether the folder holds the pinned attribution model (same identity); ``None`` when no pin is recorded.

    *identity* is the folder's, when the caller has it already (``folder_identity``).
    """
    path = pin_path or PIN_PATH
    if not path.exists():
        return None
    return bool(
        json.loads(path.read_text(encoding="utf-8")).get("identity")
        == (identity or folder_identity(model_dir))
    )


def pin_status(model_dir: Path, pin_path: Path | None = None, identity: str | None = None) -> str:
    """One line saying whether *model_dir* is the pinned attribution model (``config/attribution_model.json``, T-194)."""
    identity = identity or folder_identity(model_dir)
    pinned = is_pinned(model_dir, pin_path, identity)
    if pinned is None:
        return f"attribution model {identity}: no pin recorded"
    pin = json.loads((pin_path or PIN_PATH).read_text(encoding="utf-8"))
    if pinned:
        return f"attribution model {identity}: pinned ({pin['model']}@{pin['revision'][:8]})"
    return (
        f"attribution model {identity}: NOT the pinned one ({pin.get('identity')}); "
        "the cut is not traceable to the Hub copy"
    )


def fetch_pinned(model_dir: Path, pin_path: Path | None = None) -> str:
    """Download the pinned attribution model from the Hub into *model_dir* (empty or new) and check its identity.

    The repository is private: the token comes from ``HF_TOKEN`` (never printed). Only the files the pin lists are downloaded,
    into a temporary folder next to *model_dir*, and moved in only when they hash to the pinned identity; a wrong or altered
    copy is deleted. A folder that already holds files is refused (move it away first).
    """
    import shutil  # noqa: PLC0415 -- only the download needs these

    from huggingface_hub import get_token, snapshot_download  # noqa: PLC0415

    pin = json.loads((pin_path or PIN_PATH).read_text(encoding="utf-8"))
    if (
        not get_token()
    ):  # the Hub answers "not found" to a private repository without a token: say what is missing
        raise PermissionError(
            f"{pin['model']} is private and no Hub token is set: put HF_TOKEN in the environment or in .env"
        )
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(
            f"{model_dir} is not empty: move it away before fetching the pinned model"
        )
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_dir.parent / f".{model_dir.name}.fetching"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        snapshot_download(
            pin["model"], revision=pin["revision"], local_dir=str(tmp), allow_patterns=pin["files"]
        )
        shutil.rmtree(tmp / ".cache", ignore_errors=True)
        got = am.directory_identity(tmp)
        if got != pin["identity"]:
            raise RuntimeError(
                f"{pin['model']}@{pin['revision'][:8]} has the identity {got}, not the pinned "
                f"{pin['identity']}; the download was deleted"
            )
        if model_dir.exists():
            model_dir.rmdir()
        tmp.rename(model_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return got


class UnitsExtractor:
    """The selected extractor for a given company universe and scorer; ``cut`` returns the texts of one body."""

    def __init__(
        self,
        scorer: am.Scorer,
        companies: Sequence[Company],
        model_identity: str,
        reading: am.Reading = DEFAULT_READING,
    ) -> None:
        if reading not in CUT_READINGS:
            raise ValueError(f"reading {reading!r}: one of {CUT_READINGS}")
        self.reading = reading
        self.scorer = scorer
        self.resolver = CompanyResolver(companies)
        self.names = {c.ticker: c.name for c in companies}
        self.revision = revision_of(model_identity, self.names, reading)

    @classmethod
    def load(
        cls, model_dir: Path, companies: Sequence[Company], reading: am.Reading = DEFAULT_READING
    ) -> UnitsExtractor:
        """Load the trained attribution model of ``model_dir`` (a local folder, identified by the hash of its files)."""
        identity = f"local:{am.directory_identity(model_dir)}"
        return cls(am.TransformerScorer(str(model_dir)), companies, identity, reading)

    def cut(self, body: str) -> list[str]:
        """The text of each company the body names, in order of first appearance, without repeats."""
        texts = am.model_company_texts(
            body, self.resolver, self.names, self.scorer, reading=self.reading
        )
        return distinct(list(texts.values()))
