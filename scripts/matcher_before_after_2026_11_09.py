"""Bodies that name each company with the matcher of a git revision and with the working tree (PR #155, D38).

    uv run scripts/matcher_before_after_2026_11_09.py [--bodies 4000] [--modulo 60] [--base 786766f]
        [--tickers FRT STT ...] [--self-check]

Read-only, CPU only, no model; it needs the same environment as the count script (``SOURCE_DATABASE_URL``,
and ``DATABASE_URL`` for the pipeline connection). It reads a sample of the properly fetched SOURCE
bodies (``id % modulo = residue``, 7 by default; another residue is a held-out sample; unique by
``triage.body_hash``), resolves each with ``CompanyResolver.resolve`` of the
``triage.py`` at ``--base`` (read from git, never imported over the working one) and of the working
tree, and prints, for the given tickers, the bodies naming the company before and after, the share
dropped, and the words the dropped mentions were. It is the check behind the numbers of D38: compare a
full run of ``company_news_counts_2026_11_09.py`` with the 2026-11-09 CSV the same way.

``--base`` is the revision *before* the change (786766f, the last commit of master before PR #155):
``origin/master`` would be the branch itself once it is merged. ``--self-check`` reads no body: it
resolves each company of the universe from its own name and its short form and lists those the
working matcher does not find (a stop list must not erase a company).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import company_news_counts_2026_11_09 as counts

from news_nlp import db, env, triage

TICKERS = [
    "FRT",
    "STT",
    "UNP",
    "TTD",
    "WST",
    "HD",
    "TTWO",
    "BE",
    "XYZ",
    "UHS",
    "JKHY",
    "MLM",
    "TRV",
    "ROST",
    "WMB",
]


def triage_at(revision: str, repo: Path) -> ModuleType:
    """``news_nlp/triage.py`` as of git *revision*, loaded as a separate module."""
    source = subprocess.run(  # noqa: S603 -- fixed arguments, the revision is the owner's
        ["git", "-C", str(repo), "show", f"{revision}:src/news_nlp/triage.py"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    spec = importlib.util.spec_from_loader("triage_before", loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["triage_before"] = module  # the dataclasses of the module look itself up
    exec(compile(source, f"{revision}:triage.py", "exec"), module.__dict__)  # noqa: S102
    return module


def self_check(resolver: triage.CompanyResolver, universe: list[triage.Company]) -> list[str]:
    """The companies *resolver* does not find by their own name (raw, with its curly apostrophes and
    no-break spaces too); a stop word or a gap in the name pattern shows here."""
    missing = []
    for company in universe:
        canonical = resolver.by_ticker[company.ticker.upper()]
        for name in (company.name, company.name.replace("'", "\u2019").replace(" ", "\u00a0")):
            if not resolver.resolve(f"{name} rose.", skip_attribution=False)[canonical]:
                missing.append(f"{company.ticker}: {name!r}")
                break
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--bodies", type=int, default=4000)
    parser.add_argument("--modulo", type=int, default=60)
    parser.add_argument("--residue", type=int, default=7)
    parser.add_argument("--base", default="786766f")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    args = parser.parse_args()
    env.load_env_file()
    conn = db.connect_pipeline()
    universe = counts.load_universe(conn)
    before_module = triage_at(args.base, Path(__file__).resolve().parents[1])
    if args.self_check:
        now = self_check(triage.CompanyResolver(universe), universe)
        was = self_check(
            before_module.CompanyResolver([before_module.Company(*c) for c in universe]), universe
        )
        print(f"{len(universe) - len(now)} of {len(universe)} companies found by their own name")
        print(
            f"{len(was)} were not found at {args.base} (it read only a plain space, no apostrophe)"
        )
        print("not found now:\n" + "\n".join(now))
        lost = [m for m in now if m not in was]
        print(f"LOST since {args.base} ({len(lost)}):\n" + "\n".join(lost))
        return
    before = before_module.CompanyResolver([before_module.Company(*c) for c in universe])
    after = triage.CompanyResolver(universe)
    old: Counter[str] = Counter()
    new: Counter[str] = Counter()
    words: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    for (body,) in conn.execute(
        "SELECT body_text FROM source.articles WHERE fetch_status IN ('ok', 'quarantined') "
        "AND body_text IS NOT NULL AND trim(body_text, char(32, 9, 10, 13)) != '' AND id % ? = ? "
        "ORDER BY id",
        (args.modulo, args.residue),
    ):
        digest = triage.body_hash(body)
        if digest in seen:
            continue
        seen.add(digest)
        if len(seen) > args.bodies:
            break
        was, is_ = before.resolve(body), after.resolve(body)
        for ticker in set(was) | set(is_):
            old[ticker] += bool(was[ticker])
            new[ticker] += bool(is_[ticker])
            if was[ticker] and not is_[ticker]:
                for start, end, t in before.mention_spans(body):
                    if t == ticker:
                        words[ticker][body[start:end]] += 1
                        break
    print(
        f"{len(seen) - 1 if len(seen) > args.bodies else len(seen)} unique bodies; base {args.base}"
    )
    print("ticker  before  after  dropped  words of the dropped mentions")
    for t in args.tickers:
        drop = 100 * (old[t] - new[t]) / old[t] if old[t] else 0
        print(f"{t:6} {old[t]:7} {new[t]:6} {drop:6.0f}%  {dict(words[t].most_common(4))}")
    print(f"company-body pairs: {sum(old.values()):,} before, {sum(new.values()):,} after")


if __name__ == "__main__":
    main()
