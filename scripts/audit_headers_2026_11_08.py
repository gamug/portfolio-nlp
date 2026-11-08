"""Audit the "NAMES - text" header rule of ``news_nlp.attribution_units`` on a random sample of the
corpus (Work item 18, T-148b5, T-148b6).

    uv run scripts/audit_headers_2026_11_08.py [--n 4000] [--seed 20261108] [--sample 20]

Draws ``--n`` bodies of ``source.articles`` (a seeded draw over the ids) and reads every line of the
story (footer stripped) as the ``units2`` readings do. It counts and shows:

* lines the rule takes for a header (``header_owners`` is not ``None``);
* of those, the lines **given to nobody** (the names before the dash name no company the guard
  accepts) that nevertheless name a universe company after the dash, by reason: the part before the
  dash names a company only by a short form (the guard rejects it: "Kellogg - ..." with no
  "Kellogg Company" in the body), or names none (a company outside the universe, or a label);
* lines that would be headers if the first part were allowed to hold a full stop, and name a
  company there ("Berkshire Hathaway Inc. - ...", "U.S. Steel - ..."): headers the rule misses.

It reads SOURCE only and writes nothing.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import attribution_units as au
from news_nlp.triage import CompanyResolver

#: The header rule with full stops allowed in the first part (abbreviations: Inc., Corp., U.S.).
_DOTTED = re.compile(r'^(?P<head>[^!?:;"]{2,80}?)\s+(?:\u2014|\u2013|-)\s+(?P<rest>\S.*)$')


def audit(
    bodies: dict[int, str], resolver: CompanyResolver, seed: int = 1, sample: int = 20
) -> dict[str, object]:
    """Counts and a seeded *sample* of the lines described in the module docstring."""
    headers = 0
    lower = 0
    lower_lines: list[tuple[int, str]] = []
    nobody: list[tuple[int, str]] = []
    short_form = 0
    dotted: list[tuple[int, str]] = []
    for article, body in sorted(bodies.items()):
        story = au.strip_footer(body)
        guarded = au.GuardedResolver(resolver, story)
        for raw in story.split("\n"):
            line = au._BULLET.sub("", raw.strip())
            owners = au.header_owners(line, guarded)
            if owners is not None:
                headers += 1
                match = au._HEADER.match(line)
                if match and not au.starts_a_sentence(match["rest"]):
                    lower += 1
                    lower_lines.append((article, line))
                if owners == () and match and guarded.mention_spans(match["rest"]):
                    nobody.append((article, line))
                    short_form += bool(resolver.mention_spans(match["head"]))
                continue
            match = _DOTTED.match(line)
            if match and "." in match["head"] and guarded.mention_spans(match["head"]):
                words = [w for w in re.split(r"[\s,]+", match["head"]) if w]
                if len(words) <= au.MAX_HEAD_WORDS:
                    dotted.append((article, line))
    rng = random.Random(seed)  # noqa: S311 -- a reproducible sample
    return {
        "bodies": len(bodies),
        "header lines": headers,
        "header lines whose text after the dash does not start a sentence": lower,
        "headers of nobody that name a company after the dash": len(nobody),
        "of those, the part before the dash names a company by a short form only": short_form,
        "headers missed for a full stop in the first part": len(dotted),
        "nobody": rng.sample(nobody, min(sample, len(nobody))),
        "dotted": rng.sample(dotted, min(sample, len(dotted))),
        "lower": rng.sample(lower_lines, min(sample, len(lower_lines))),
    }


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()
    conn = db.connect_pipeline()
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM source.articles WHERE body_text IS NOT NULL AND body_text != ''"
        )
    ]
    picked = random.Random(args.seed).sample(ids, min(args.n, len(ids)))  # noqa: S311
    bodies = {
        a: conn.execute("SELECT body_text FROM source.articles WHERE id = ?", (a,)).fetchone()[0]
        for a in picked
    }
    result = audit(bodies, CompanyResolver(retag_cli.companies()), args.seed, args.sample)
    for key in list(result)[:6]:
        print(f"{key}: {result[key]}")
    for label in ("nobody", "dotted", "lower"):
        print(f"--- a sample of the lines, {label}:")
        for article, line in result[label]:  # type: ignore[attr-defined]
            print(f"[{article}] {line[:300]!r}")


if __name__ == "__main__":
    main()
