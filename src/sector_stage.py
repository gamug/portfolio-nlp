"""The sector step: the GICS sector of a ``sector`` body: a small embedding model, the companies the body names, and a gate (PLAN D39 (4); T-167).

A body the scope step calls ``sector`` gets a GICS sector (the 2-digit code, 11 classes) or is sent to ``out_of_scope``:

1. **Score.** The body is cut in chunks of 120 words (the title goes in front of the first one), each chunk is embedded by the
   pinned ``BAAI/bge-small-en-v1.5`` (``config/sector_step.json``) and compared with the descriptions of the 11 sectors and of
   four *not a sector* classes (market_economy, investing_general, investment_activity, unrelated: the scope errors the owner
   found in the first review). A class scores the mean of its three best chunks; a direction that is common to all analyst
   notes ("upgrades the stock, raises the target") is taken out of the chunks first. The margin is the best score minus the second.
2. **Votes.** ``triage.CompanyResolver`` finds the universe companies the body names; each votes for its own GICS sector
   (at most 3 votes per company).
3. **Gate.** The sector is assigned when the best class is a sector and either the margin is over the threshold or the companies'
   top sector is the same one. Anything else is ``out_of_scope``: the owner's rule (2026-11-13), because a wrong sector costs
   more than a missing one.

How the choices were made, as it was and not as a protocol would have it: the sector descriptions were written on the tune half of the labelled
bodies; the *not a sector* classes were written from the owner's notes on the first yes/no review; the margin threshold is the 40th percentile of
the tune half (tagged bodies and reviewed rows); the gate (``margin or agree``) was picked after both halves had been printed in one run. The test
halves were read many times. The only clean figure is the owner's review of 2026-11-13 (``gold/sector_validation.csv``, ``fresh``): see
``docs/sector_step.md``.

The revision stored with every result is ``sector-`` plus 12 hex digits of the hash of the model and its pin, the parameters, the descriptions,
the universe and the code of this module *as parsed* (``news_summary_stage.normalized_code``: comments, formatting and docstrings do not count, as
in the summary step): a change of any of the rest starts new rows and leaves the old ones, a typo fix in a comment does not.

It runs on CUDA when there is one and on the CPU otherwise (constitution, Pipeline models 2); the model is loaded on the first
text, so ``--status`` and the tests need none.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from news_nlp.triage import Company, CompanyResolver

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "sector_step.json"
SECTOR_CODES = {
    "Energy": "10",
    "Materials": "15",
    "Industrials": "20",
    "Consumer Discretionary": "25",
    "Consumer Staples": "30",
    "Health Care": "35",
    "Financials": "40",
    "Information Technology": "45",
    "Communication Services": "50",
    "Utilities": "55",
    "Real Estate": "60",
}
SECTOR_DESCRIPTIONS: dict[str, list[str]] = {
    "10": [
        "Energy stocks: oil and gas producers, crude oil prices, Brent and WTI, OPEC output cuts, shale drilling",
        "Midstream pipeline operators, LNG export terminals, natural gas prices and storage, refiners and refining margins",
        "Oilfield services and drillers, rig counts, offshore drilling, exploration and production spending",
        "Oil majors' earnings, production growth, capital discipline and buybacks in the oil patch, coal producers",
        "Analysts weigh in on the energy sector as oil prices swing, upgrades and downgrades of energy stocks",
    ],
    "15": [
        "Chemicals makers, fertilizers, industrial gases and specialty chemicals, pricing and demand",
        "Mining companies, copper, gold, silver, lithium, iron ore and aluminum prices, metal output",
        "Steel producers, tariffs on steel and aluminum, mill capacity and steel demand",
        "Packaging, paper and forest products, construction materials such as cement, aggregates and lumber",
        "Gold and copper miners' production guidance, commodity prices and mine costs",
    ],
    "20": [
        "Aerospace and defense contractors, aircraft production and orders, Pentagon contracts, shipbuilding",
        "Airlines, fares, capacity and fuel costs, travel demand, government aid to carriers, pilots and unions",
        "Railroads, trucking, freight and logistics, parcel delivery, shipping rates and container volumes",
        "Industrial machinery, construction and engineering, electrical equipment, building products, heavy equipment orders",
        "Waste services, staffing and business services, conglomerates, industrial stocks and factory orders",
    ],
    "25": [
        "Automakers, car sales and auto profits, electric vehicles, tariffs on cars, auto parts and dealers",
        "Retailers, department stores, e-commerce platforms, apparel and footwear brands, same-store sales, holiday shopping",
        "Restaurant chains and fast food, same-store sales, menu prices, burgers, coffee and pizza chains",
        "Hotels, casinos, cruise lines, airlines' leisure travel, theme parks, homebuilders and household durables",
        "Chinese internet platforms and online marketplaces, food delivery and consumer e-commerce stocks",
    ],
    "30": [
        "Tobacco and cigarette makers, cigarette sales decline, smoking, vaping and nicotine products",
        "Beer, wine, spirits and liquor makers, soft drinks, bottlers, energy drinks and beverage volumes",
        "Packaged food makers, snacks, cereal and candy, food prices, commodity costs and volumes",
        "Grocery chains, supermarkets, warehouse clubs, drugstore chains, discount retailers and food distributors",
        "Household and personal care products, cosmetics, diapers, detergent, consumer staples pricing power",
    ],
    "35": [
        "Drugmakers and biotechnology, clinical trial results, FDA approvals, drug patents and generics",
        "Medical device makers, diagnostics, life sciences tools, surgical robots and lab equipment",
        "Hospitals, health care providers, nursing homes, physician groups and health services",
        "Health insurers, managed care, Medicare and Medicaid, health care costs and premiums",
        "Drug pricing policy, vaccines, obesity drugs, pharmaceutical sales and pipelines",
    ],
    "40": [
        "Banks: net interest margin, loans, deposits, loan losses, credit quality, stress tests and capital rules",
        "Insurers and reinsurers, premiums, claims, catastrophe losses and insurance underwriting",
        "Asset managers, exchanges, brokerages, private equity firms, fees and assets under management",
        "Payment networks, credit cards, consumer finance, fintech lenders and mortgage lenders",
        "Bank earnings season, regional bank deposits, financial regulation and bank mergers",
    ],
    "45": [
        "Software and cloud computing stocks, enterprise software spending, SaaS subscriptions, cybersecurity",
        "Semiconductor and chip stocks, chipmakers, memory chips, chip equipment, data center demand",
        "Artificial intelligence spending, AI chips and servers, hyperscalers' capital expenditure",
        "Computer hardware, smartphones, networking gear, storage, electronic components and tech distributors",
        "IT services and consulting, technology stocks rally or selloff, tech valuations and earnings",
    ],
    "50": [
        "Telecom carriers, wireless subscribers, broadband, 5G, cable and satellite operators",
        "Media and entertainment companies, film studios, streaming services, subscribers and content spending",
        "Social media and internet platforms, search engines, digital advertising revenue",
        "Video game publishers, music, publishing, newspapers and broadcasters",
        "Advertising agencies, television networks, ratings and media mergers",
    ],
    "55": [
        "Electric utilities, power generation, rate cases, state regulators and electricity prices",
        "Natural gas utilities and water utilities, rate increases, customers and regulated returns",
        "Nuclear plants, renewable power producers, wind and solar developers, independent power producers",
        "Power grids, transmission lines, wildfire liability, utility earnings and dividends",
        "Power demand growth from data centers, utility capital plans and electricity load",
    ],
    "60": [
        "Real estate investment trusts, office, retail and industrial properties, occupancy and rents",
        "Shopping malls and mall owners, apartment and housing REITs, healthcare properties",
        "Data center, cell tower, warehouse and self-storage REITs, funds from operations",
        "Housing market recovery, home prices, home sales, mortgage rates and housing starts",
        "Commercial real estate, property values, office vacancy, real estate services and brokers",
    ],
}
#: What is not a sector body: the scope errors the owner named in the first yes/no review (2026-11-13).
NOT_A_SECTOR: dict[str, list[str]] = {
    "market_economy": [
        "Jobs report, private payrolls, unemployment, GDP, inflation, consumer prices and economic data below expectations",
        "Federal Reserve, interest rates, tariffs and trade policy, the economy, bitcoin and commodity prices",
    ],
    "investing_general": [
        "CEO interview about leadership and business advice, a billionaire investor on how to build a business",
        "Stock picks and portfolio strategy, market outlook, how to invest, strategists' targets and predictions",
    ],
    "investment_activity": [
        "Fund managers and hedge funds buying, ETF flows, small cap rally, investors positioning and funds bets",
    ],
    "unrelated": [
        "Politics, lawsuits, social issues, inauguration, harassment, charity, scams and news not about any industry",
    ],
}
NOISE = "An investment bank analyst upgrades the stock, raises the price target and says investors should buy the shares"
CLASSES: dict[str, list[str]] = {**SECTOR_DESCRIPTIONS, **NOT_A_SECTOR}
STATUS_ASSIGNED, STATUS_OUT = "assigned", "out_of_scope"

#: Anything that turns text into unit vectors, one row per text (the real one is :class:`BgeEmbedder`).
Embed = Callable[[list[str]], np.ndarray]


def pinned_config() -> dict[str, Any]:
    """``config/sector_step.json``: the embedder pin, the chunking and the gate's numbers."""
    config: dict[str, Any] = json.loads(_CONFIG.read_text(encoding="utf-8"))
    return config


@dataclass(frozen=True)
class SectorResult:
    status: str  # assigned / out_of_scope
    sector: str  # the 2-digit GICS code, '' when out_of_scope
    candidate: str  # the best class of the embedding score (a sector code or a not-a-sector class)
    margin: float
    vote_sector: str  # the top sector of the companies the body names, '' when it names none
    scores: dict[str, float]
    votes: dict[str, int]
    truncated_chunks: int = 0  # chunks of this body that filled the embedder's window and were cut there (0 for an embedder that does not say)


def chunks_of(text: str, words: int) -> list[str]:
    parts = text.split()
    return [" ".join(parts[i : i + words]) for i in range(0, max(len(parts), 1), words)] or [""]


def _unit(x: np.ndarray) -> np.ndarray:
    return np.asarray(x / np.linalg.norm(x, axis=-1, keepdims=True))


def sector_votes(
    resolver: CompanyResolver, sector_of: dict[str, str], text: str, max_votes: int
) -> dict[str, int]:
    """Votes per sector code: each universe company the text names adds its mentions (at most ``max_votes``) to its sector."""
    votes: Counter[str] = Counter()
    for ticker, mentions in resolver.resolve(text).items():
        if ticker in sector_of:
            votes[sector_of[ticker]] += min(mentions, max_votes)
    return dict(votes)


def top_vote(votes: dict[str, int]) -> str:
    """The sector with most votes; '' when nobody voted **or the top is shared** (a tie is not agreement: the second review of #179 found the
    lowest code winning ties, so a split vote "agreed" with the embedding whenever its sector had the lower code)."""
    if not votes:
        return ""
    best = max(votes.values())
    top = [code for code, n in votes.items() if n == best]
    return top[0] if len(top) == 1 else ""


def decide(
    scores: dict[str, float], votes: dict[str, int], threshold: float
) -> tuple[str, str, str, float]:
    """``(status, sector, candidate, margin)``: the gate.

    Assigned when the best class is a sector and (the margin is at least ``threshold`` or the companies vote for that same sector).
    """
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    candidate, best = ranked[0]
    margin = best - ranked[1][1]
    voted = top_vote(votes)
    if candidate in SECTOR_DESCRIPTIONS and (margin >= threshold or voted == candidate):
        return STATUS_ASSIGNED, candidate, candidate, margin
    return STATUS_OUT, "", candidate, margin


def universe_signature(companies: Sequence[Company]) -> str:
    rows = sorted((c.ticker, c.name, c.gics_sector) for c in companies)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:12]


def _module_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def code_hash(source: str) -> str:
    """The hash of a module's code as parsed: the same ``normalized_code`` the summary step hashes (a comment or a docstring does not count)."""
    # the one definition of "the code, as parsed" (loads torch and transformers: only a revision needs it)
    from news_summary_stage import normalized_code  # noqa: PLC0415

    return hashlib.sha256(normalized_code(source).encode()).hexdigest()[:12]


def revision_of(config: dict[str, Any], companies: Sequence[Company]) -> str:
    """``sector-<12 hex>``: the model and its pin, the parameters, the descriptions, the universe and this module's code as parsed."""
    key = json.dumps(
        {
            "config": config,
            "classes": CLASSES,
            "noise": NOISE,
            "universe": universe_signature(companies),
            "code": code_hash(_module_source()),
        },
        sort_keys=True,
    )
    return "sector-" + hashlib.sha256(key.encode()).hexdigest()[:12]


class BgeEmbedder:
    """The pinned embedding model: CLS pooling, normalised; loaded on the first call."""

    def __init__(self, model: str, revision: str, max_length: int, batch_size: int) -> None:
        self.model_name, self.model_revision = model, revision
        self.max_length, self.batch_size = max_length, batch_size
        self._tok: Any = None
        self._model: Any = None
        self._device: Any = None
        #: for the texts of the last call, in order: did the text fill the whole window (``max_length`` tokens) and get cut there? The model cannot
        #: see the rest, so it is recorded with each body (constitution, Pipeline models 3); a chunk of 120 words is far under the window
        self.last_truncated = np.zeros(0, dtype=bool)

    def _load(self) -> None:
        if self._model is None:
            import torch  # noqa: PLC0415 -- only a real run needs it
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.model_revision)
            self._model = (
                AutoModel.from_pretrained(self.model_name, revision=self.model_revision)
                .to(self._device)
                .eval()
            )

    def __call__(self, texts: list[str]) -> np.ndarray:
        import torch  # noqa: PLC0415

        self._load()
        out = []
        cut: list[bool] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self._tok(
                    texts[i : i + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                cut.extend((batch["attention_mask"].sum(dim=1) >= self.max_length).tolist())
                cls = self._model(**batch).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        self.last_truncated = np.array(cut, dtype=bool)
        return np.vstack(out) if out else np.zeros((0, 1))


class SectorClassifier:
    """The sector step: ``classify`` returns one :class:`SectorResult` per ``(title, text)``, in the order given."""

    def __init__(
        self,
        companies: Sequence[Company],
        config: dict[str, Any] | None = None,
        embed: Embed | None = None,
    ) -> None:
        self.config = config or pinned_config()
        emb = self.config["embedder"]
        self.embed: Embed = embed or BgeEmbedder(
            emb["model"], emb["revision"], emb["max_length"], emb["batch_size"]
        )
        self.revision = revision_of(self.config, companies)
        self.resolver = CompanyResolver(list(companies))
        self.sector_of = {
            c.ticker: SECTOR_CODES[c.gics_sector]
            for c in companies
            if c.gics_sector in SECTOR_CODES
        }
        self._classes = list(CLASSES)
        self._owner: np.ndarray | None = None
        self._desc: np.ndarray | None = None
        self._noise: np.ndarray | None = None

    def prepare(self) -> None:
        """Load the model and embed the descriptions now, so a progress bar does not count it in the first batch (idempotent)."""
        self._prepare()

    def _prepare(self) -> None:
        if self._desc is None:
            flat = [d for name in self._classes for d in CLASSES[name]]
            self._desc = self.embed(flat)
            self._owner = np.repeat(
                np.arange(len(self._classes)), [len(CLASSES[n]) for n in self._classes]
            )
            self._noise = self.embed([NOISE])[0]

    def scores_of(self, chunk_vecs: np.ndarray) -> dict[str, float]:
        """One body's score per class: the mean of its ``top_chunks`` best chunks, each chunk taking its best description."""
        assert self._desc is not None and self._owner is not None and self._noise is not None
        alpha = self.config["noise_alpha"]
        vecs = _unit(
            chunk_vecs - alpha * (chunk_vecs @ self._noise)[:, None] * self._noise[None, :]
        )
        sims = vecs @ self._desc.T  # chunks x descriptions
        per_class = np.stack(
            [sims[:, self._owner == j].max(axis=1) for j in range(len(self._classes))], axis=1
        )
        top = np.sort(per_class, axis=0)[-self.config["top_chunks"] :].mean(axis=0)
        return {name: float(top[j]) for j, name in enumerate(self._classes)}

    def classify(self, items: Sequence[tuple[str, str]]) -> list[SectorResult]:
        self._prepare()
        words = self.config["chunk_words"]
        pieces: list[list[str]] = []
        for title, text in items:
            chunks = chunks_of(text, words)
            pieces.append([f"{title}. {chunks[0]}", *chunks[1:]])
        flat = self.embed([p for piece in pieces for p in piece])
        cut = np.asarray(getattr(self.embed, "last_truncated", np.zeros(len(flat), dtype=bool)))
        results, start = [], 0
        for (title, text), piece in zip(items, pieces, strict=True):
            scores = self.scores_of(flat[start : start + len(piece)])
            start += len(piece)
            votes = sector_votes(
                self.resolver,
                self.sector_of,
                f"{title}. {text}",
                self.config["max_votes_per_company"],
            )
            status, sector, candidate, margin = decide(
                scores, votes, self.config["margin_threshold"]
            )
            results.append(
                SectorResult(
                    status,
                    sector,
                    candidate,
                    margin,
                    top_vote(votes),
                    scores,
                    votes,
                    int(cut[start - len(piece) : start].sum()) if len(cut) == len(flat) else 0,
                )
            )
        return results
