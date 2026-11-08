"""Draw the narrative multi bodies of the extractor's faithfulness check and write the file the
owner marks (Work item 18, T-148b2, D27, D32).

    uv run scripts/draw_faithfulness_2026_11_06.py [--out data] [--n 30] [--seed 20261106]
        [--extractors baseline[,model]] [--attribution-model DIR_OR_ID] [--yes]
    uv run scripts/draw_faithfulness_2026_11_06.py --add-extractor model [--out data] [--yes]
        --attribution-model data/models/attribution_2026_11_08
        (or a Hub id with --attribution-revision)

The bodies are the ones the owner tagged ``multi`` (``gold/category_gold_v2.csv``) that the
one-segment-per-company parser cannot split (``company_texts.split_company_texts`` is ``None``),
minus the held-out set; ``--n`` of them are drawn once (``<out>/faithfulness_bodies.csv``, never
drawn again, so the baseline and the model are judged on the same bodies).

Each extractor (``attribution_baseline.baseline_company_texts`` or the trained model of T-148b3,
``news_nlp.attribution_model``, named by ``--attribution-model``) writes the text each company
named in the body gets. The owner reads each text with its article and marks ``faithful`` yes or
no (``retag.FAITHFULNESS_COLUMNS``, the Spanish pair right after ``text``, and the Spanish version
of the article). The file is **blind to the extractor**: the rows are shuffled, and a text two
extractors wrote identically is one row. Which extractor wrote which row (by ``review_order``) is
in ``<out>/faithfulness_ids.csv``, to compute the figures with ``report_wi18``.

``--extractors baseline`` alone writes the file for the baseline only (the decision on the
baseline, D34); nothing is written for ``model`` without ``--attribution-model``, which is loaded
only when ``model`` is among the extractors used. The Spanish aid is the approved paid
translation of the same engine; nothing is bought without ``--yes``. Its cache is
``<out>/faithfulness/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import (
    attribution_baseline,
    attribution_model,
    attribution_units,
    company_texts,
    gold_metrics,
    retag,
    triage,
)
from news_nlp.triage import CompanyResolver

BODIES_NAME = "faithfulness_bodies.csv"
IDS_NAME = "faithfulness_ids.csv"
TO_TAG_NAME = "faithfulness_to_tag.csv"
CACHE_DIR = "faithfulness"
GOLD = Path("gold/category_gold_v2.csv")
BODY_FIELDS = ("article_id", "segments")
COVERAGE_NAME = gold_metrics.COVERAGE_NAME
EXEMPT_NAME = gold_metrics.EXEMPT_NAME
EXEMPT_READ_NAME = gold_metrics.EXEMPT_READ_NAME
COVERAGE_FIELDS = gold_metrics.COVERAGE_COLUMNS
IDS_FIELDS = (
    "review_order",
    "key",
    "article_id",
    "ticker",
    "text_hash",
    "extractors",
    "revisions",
    "pub_date",
)
UNITS_READINGS = ("units", "units2", "units3")  # the extractors that read the body by units
Extractor = Callable[[str, CompanyResolver], dict[str, str]]
EXTRACTORS: dict[str, Extractor] = {"baseline": attribution_baseline.baseline_company_texts}
MODEL_IDENTITY: list[str] = []  # the trained model's identity, set by register_model


def extractor_revision(name: str) -> str:
    """The revision of extractor *name*: the baseline's code, or the trained model's identity (the
    Hub id and revision, or a local directory's files hash) with the code of the extractor:
    ``model`` the model's modules, the ``units`` readings those and the footer and unit cutting."""
    if name not in ("model", *UNITS_READINGS):
        return retag.extractor_revision(name)
    if not MODEL_IDENTITY:
        raise ValueError("the model extractor is not registered: pass --attribution-model")
    modules = (
        attribution_model.UNITS_CODE_MODULES
        if name in UNITS_READINGS
        else attribution_model.CODE_MODULES
    )
    code = retag.code_revision(*modules)
    return f"extractor:{name}@{MODEL_IDENTITY[0]}@{code}"


def model_extractor(
    scorer: attribution_model.Scorer, names: dict[str, str], *, reading: str = "sentences"
) -> Extractor:
    """The trained model as an extractor: the baseline's signature, the model behind *scorer*. With
    *reading* ``units`` the body is read without its footer and cut at lines, bullets and clauses
    (``news_nlp.attribution_units``, T-148b4); ``units2`` also with the guarded resolver and the
    lines headed by names (T-148b5); ``units3`` also gives such a line to the companies its text
    names (T-148b6)."""

    def extract(body: str, resolver: CompanyResolver) -> dict[str, str]:
        return attribution_model.model_company_texts(body, resolver, names, scorer, reading=reading)

    return extract


def local_identity(directory: Path) -> str:
    """A hash of every top-level file of the model directory (weights, config, tokenizer)."""
    try:
        return f"local:{attribution_model.directory_identity(directory)}"
    except ValueError as error:
        sys.exit(str(error))


def register_model(model: str, revision: str | None, names: dict[str, str]) -> None:
    """Make ``model`` an extractor. A Hub id needs a *revision* (every model is pinned); a local
    directory is identified by the hash of its weights and tokenizer. The identity also carries the
    company names the model is asked about and the threshold, which change the texts too."""
    local = Path(model).is_dir()
    if not local and not revision:
        sys.exit(f"{model} is not a local directory: it needs a revision (every model is pinned)")
    base = local_identity(Path(model)) if local else f"{model}@{revision}"
    names_hash = hashlib.sha256(json.dumps(sorted(names.items())).encode()).hexdigest()[:8]
    MODEL_IDENTITY[:] = [f"{base}@names:{names_hash}@t{attribution_model.THRESHOLD}"]
    scorer = attribution_model.TransformerScorer(model, revision)
    EXTRACTORS["model"] = model_extractor(scorer, names)
    EXTRACTORS["units"] = model_extractor(scorer, names, reading="units")
    EXTRACTORS["units2"] = model_extractor(scorer, names, reading="units2")
    EXTRACTORS["units3"] = model_extractor(scorer, names, reading="units3")


def draw_bodies(candidates: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """*n* of the candidate bodies (``article_id`` and ``segments``), seeded, in article order."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    pool = sorted(candidates, key=lambda r: int(r["article_id"]))
    rng.shuffle(pool)
    return sorted(pool[:n], key=lambda r: int(r["article_id"]))


def merge_texts(
    per_extractor: dict[str, dict[str, str]],
) -> list[tuple[str, tuple[str, ...]]]:
    """``(text, extractors)`` for one company, one entry per distinct text, extractors sorted."""
    by_text: dict[str, list[str]] = {}
    for name in sorted(per_extractor):
        text = per_extractor[name].get("text", "").strip()
        if text:
            by_text.setdefault(text, []).append(name)
    return [(t, tuple(v)) for t, v in by_text.items()]


def candidate_bodies(
    out: Path, resolver: CompanyResolver, before: Path | None = None
) -> list[dict[str, str]]:
    """The narrative multi bodies the one-segment parser cannot split, minus the held-out set and,
    for a confirmation draw, the bodies (by hash: twins share a body) of the earlier draw in
    *before*, whose held-out list is used when *out* has none."""
    conn = db.connect_pipeline()
    heldout = out / "heldout_ids.csv"
    if before is not None and not heldout.exists():
        heldout = before / "heldout_ids.csv"
    ids = {int(r["article_id"]) for r in extra._read(heldout)}
    if before is not None:
        ids |= {int(r["article_id"]) for r in extra._read(before / BODIES_NAME)}
    held = extra.excluded_hashes(conn, ids)
    found: list[dict[str, str]] = []
    for row in extra._read(GOLD):
        if row["scope"] != "multi":
            continue
        src = conn.execute(
            "SELECT body_text FROM source.articles WHERE id = ?", (int(row["article_id"]),)
        ).fetchone()
        body = src["body_text"] or ""
        if triage.body_hash(body) in held:  # the body, not the article id: twins share a body
            continue
        if company_texts.split_company_texts(body, resolver) is None:
            found.append(
                {
                    "article_id": row["article_id"],
                    "segments": str(len(company_texts.segments(body))),
                }
            )
    return found


def stale_report(
    ids: list[dict[str, str]],
    items: list[dict[str, str]],
    extractors: list[str],
    revisions: dict[str, str],
) -> dict[str, list[str]]:
    """What the code writes now against the rows the owner marked (*ids*), for the *extractors*
    that were run (*items* are their texts, :func:`collect_items`). ``stale``: a marked row whose
    extractor now writes another text (or none) for that company, so its mark is on a text the code
    no longer produces; ``new``: texts the code writes now that no row holds, which would be new
    rows to read; ``revision``: rows whose extractor revision differs from the current one (the
    text may still be the same: the mark stands, the record is old)."""
    now: dict[tuple[str, str, str], str] = {}
    for item in items:
        for name in item["extractors"].split("|"):
            now[(item["article_id"], item["ticker"], name)] = item["text_hash"]
    held = {(r["article_id"], r["ticker"], r["text_hash"]) for r in ids}
    out: dict[str, list[str]] = {"stale": [], "new": [], "revision": []}
    for row in ids:
        have = dict(zip(row["extractors"].split("|"), row["revisions"].split("|"), strict=True))
        for name in sorted(set(have) & set(extractors)):
            label = f"{row['key']} ({name})"
            if now.get((row["article_id"], row["ticker"], name)) != row["text_hash"]:
                out["stale"].append(label)
            if have[name] != revisions[name]:
                out["revision"].append(label)
    out["new"] = [
        f"{i['key']} ({i['extractors']})"
        for i in items
        if (i["article_id"], i["ticker"], i["text_hash"]) not in held
    ]
    return out


def exempt_rules() -> str:
    """The revision of the code that decides which companies are exempt (footer and guard)."""
    return retag.code_revision(*attribution_model.UNITS_CODE_MODULES)


def _read_reading(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """The reading file as the owner's spreadsheet left it: its columns in its order, and its rows."""
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        if not gold_metrics.exempt_read_header_ok(header):
            sys.exit(f"{path.name} must have the columns {gold_metrics.EXEMPT_READ_COLUMNS}")
        return header, list(reader)


def write_exempt(out: Path, *, yes: bool = False, max_new: int = retag_cli.DEFAULT_MAX_NEW) -> None:
    """``--exempt``: write ``faithfulness_exempt.json`` (the companies each drawn body names that
    the ``units`` extractors are not asked about, with the revision of the rules that found them)
    and ``faithfulness_exempt_to_read.csv``: one row per company with the line that names it, for
    the owner to mark ``real_company`` yes or no. **The owner's file is merged, never replaced:**
    its columns stay in the order the spreadsheet left them, its rows stay, ``real_company`` and
    every other value is untouched; only a company not in the file is appended, and only empty
    Spanish cells are filled. The Spanish is the approved paid translation of the title and the
    line (no body), bought only with ``--yes``; one already in the file is not bought again."""
    rows, reading = exempt_rows(
        extra._read(out / BODIES_NAME), CompanyResolver(retag_cli.companies())
    )
    document = {"rules": exempt_rules(), "companies": rows}
    read_path = out / EXEMPT_READ_NAME
    header, have = _read_reading(read_path)
    known = {(r["article_id"], r["ticker"]) for r in have}
    table = have + [r for r in reading if (r["article_id"], r["ticker"]) not in known]
    todo = [r for r in table if not r["spanish_text"]]
    if yes and todo:
        cache = out / CACHE_DIR
        cache.mkdir(exist_ok=True)
        asked = [
            {
                "article_id": f"{r['article_id']}#{r['ticker']}",
                "title": r["title"],
                "text": r["text"],
            }
            for r in todo
        ]
        done = {
            r["article_id"]: r
            for r in retag_cli.add_spanish(
                cache, asked, retag_cli._translator(), yes=True, max_new=max_new
            )
        }
        for row in todo:
            got = done[f"{row['article_id']}#{row['ticker']}"]
            row["spanish_title"], row["spanish_text"] = got["spanish_title"], got["spanish_text"]
    elif todo:
        print(f"{len(todo)} rows without Spanish: --yes pays for the translation")
    safe = retag.spreadsheet_safe(table, tag_columns=("real_company",))
    # the spreadsheet saves with CRLF: the file keeps the line endings it was found with
    crlf = read_path.exists() and b"\r\n" in read_path.read_bytes()
    gold_metrics.write_pair_atomically(
        {
            out / EXEMPT_NAME: json.dumps(document, indent=1) + "\n",
            # with a byte order mark, like the other files the owner opens in a spreadsheet: without
            # it Excel reads the UTF-8 as Windows-1252 and "percepción" shows as "percepciÃ³n"
            read_path: "\ufeff"
            + gold_metrics.render_csv(
                tuple(header or gold_metrics.EXEMPT_READ_COLUMNS),
                safe,
                lineterminator="\r\n" if crlf else "\n",
            ),
        }
    )
    pairs = {(r["article_id"], r["ticker"]) for r in rows}
    print(
        f"{len(pairs)} companies exempt for at least one extractor written to {out / EXEMPT_NAME}"
    )


def label_extractors(label: str) -> set[str]:
    """The extractors named in a ``<key> (<a>|<b>)`` label of :func:`stale_report`."""
    return set(label.rsplit("(", 1)[1].rstrip(")").split("|"))


def run_stale(args: argparse.Namespace) -> int:
    """``--stale``: free and read-only (it writes ``faithfulness_stale.json``, the result the report
    quotes); run the registered extractors (``--only`` limits them) over the drawn bodies and say
    which marked rows the code no longer reproduces. The file keeps the last check of every
    extractor: a run of some does not erase the others. Returns the exit code, a sum: 1 some marked
    row is stale, 2 some extractor of the ids was neither run nor excluded by ``--only`` (the model
    needs ``--attribution-model``); 0 all that was asked for was checked and is clean."""
    out: Path = args.out
    names = sorted(EXTRACTORS)
    if args.only:
        names = sorted({n.strip() for n in args.only.split(",") if n.strip()})
        unknown = [n for n in names if n not in EXTRACTORS]
        if unknown:
            sys.exit(f"--only {unknown}: not registered here ({sorted(EXTRACTORS)})")
    resolver = CompanyResolver(retag_cli.companies())
    items, _ = collect_items(extra._read(out / BODIES_NAME), names, resolver)
    revisions = {n: extractor_revision(n) for n in names}
    ids = extra._read(out / IDS_NAME)
    result = stale_report(ids, items, names, revisions)
    left = sorted({n for row in ids for n in row["extractors"].split("|")} - set(names))
    unchecked = [] if args.only else left
    print("extractors run:", ", ".join(names))
    if left and args.only:
        print(f"left out by --only: {', '.join(left)}")
    elif left:
        print(
            f"not checked: {', '.join(left)} (model and units need --attribution-model; an "
            "extractor this branch does not define cannot be checked here)"
        )
    for label, rows in result.items():
        print(f"{label}: {len(rows)}")
        if label == "revision":  # the text is the same: only the record is old, so a count each
            by_extractor = Counter(row.rsplit("(", 1)[1].rstrip(")") for row in rows)
            print("  ", dict(sorted(by_extractor.items())))
            continue
        for row in rows[:40]:
            print("  ", row)
    path = out / gold_metrics.STALE_NAME
    ran = {
        n: gold_metrics.StaleEntry(
            date=date.today().isoformat(),
            revision=revisions[n],
            stale=[r for r in result["stale"] if r.endswith(f"({n})")],
            new=[r for r in result["new"] if n in label_extractors(r)],
        )
        for n in names
    }
    document = gold_metrics.merge_stale(gold_metrics.read_stale(path), ran)
    gold_metrics.write_pair_atomically({path: json.dumps(document, indent=1) + "\n"})
    return (1 if result["stale"] else 0) + (2 if unchecked else 0)


def exempt_rows(
    bodies: list[dict[str, str]], resolver: CompanyResolver
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The companies each body names that the ``units`` extractors are not asked about: the ones
    named only in the footer (``units``, ``units2``, ``units3``) and the ones named in the story
    only by the first word of their name, which the guard does not accept (``units2``,
    ``units3``). Returns the rows of ``EXEMPT_NAME`` (``COVERAGE_FIELDS``, one per extractor) and
    the rows of ``EXEMPT_READ_NAME`` (one per company, with the line that names it, to read)."""
    conn = db.connect_pipeline()
    rows: list[dict[str, str]] = []
    reading: list[dict[str, str]] = []
    names = {c.ticker: c.name for c in retag_cli.companies()}
    for b in bodies:
        src = conn.execute(
            "SELECT title, body_text, pub_date FROM source.articles WHERE id = ?",
            (int(b["article_id"]),),
        ).fetchone()
        body = src["body_text"] or ""
        story = attribution_units.strip_footer(body)
        in_story = set(resolver.resolve(story, skip_attribution=False))
        accepted = set(attribution_units.GuardedResolver(resolver, story).resolve(story))
        why = {t: "footer" for t in set(resolver.resolve(body, skip_attribution=False)) - in_story}
        why |= {t: "guard" for t in in_story - accepted}
        for ticker, reason in sorted(why.items()):
            extractors = UNITS_READINGS if reason == "footer" else UNITS_READINGS[1:]
            rows += [
                {
                    "article_id": b["article_id"],
                    "ticker": ticker,
                    "extractor": name,
                    "pub_date": src["pub_date"] or "",
                }
                for name in extractors
            ]
            reading.append(
                {
                    "article_id": b["article_id"],
                    "ticker": ticker,
                    "company": names.get(ticker, ticker),
                    "why": reason,
                    "extractors": "|".join(extractors),
                    "title": src["title"] or "",
                    "text": _line_naming(body, resolver, ticker),
                    "spanish_title": "",
                    "spanish_text": "",
                    "real_company": "",
                }
            )
    return rows, reading


def _line_naming(body: str, resolver: CompanyResolver, ticker: str) -> str:
    """The line of *body* that holds the first mention of *ticker* (at most 400 characters)."""
    for start, _end, found in resolver.mention_spans(body):
        if found == ticker:
            before = body.rfind("\n", 0, start) + 1
            after = body.find("\n", start)
            return body[before : after if after >= 0 else len(body)].strip()[:400]
    return ""


def collect_items(
    bodies: list[dict[str, str]], names: list[str], resolver: CompanyResolver
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """One item per distinct text an extractor wrote for a company named in a body, with the
    extractors that wrote it (``extractors``) and the article it was taken from; and one row per
    named company an extractor left without a text (``COVERAGE_FIELDS``): the report counts each as
    unfaithful, so an extractor cannot gain by leaving the hard companies out."""
    conn = db.connect_pipeline()
    items: list[dict[str, str]] = []
    omissions: list[dict[str, str]] = []
    for b in bodies:
        src = conn.execute(
            "SELECT title, body_text, pub_date, source_domain FROM source.articles WHERE id = ?",
            (int(b["article_id"]),),
        ).fetchone()
        body = src["body_text"] or ""
        found = {name: EXTRACTORS[name](body, resolver) for name in names}
        # a company named only in the footer is not a company of the story: the units extractor,
        # which reads the body without its footer, is not asked for a text of it
        story = attribution_units.strip_footer(body)
        in_story = resolver.resolve(story, skip_attribution=False)
        guarded = (
            attribution_units.GuardedResolver(resolver, story).resolve(story)
            if {"units2", "units3"} & set(names)
            else {}
        )
        for ticker in sorted(resolver.resolve(body, skip_attribution=False)):
            per = {n: {"text": found[n].get(ticker, "")} for n in names}
            omissions += [
                {
                    "article_id": b["article_id"],
                    "ticker": ticker,
                    "extractor": n,
                    "pub_date": src["pub_date"] or "",
                }
                for n in names
                if not per[n]["text"].strip()
                and (n != "units" or ticker in in_story)
                and (n not in ("units2", "units3") or ticker in guarded)
            ]
            for number, (text, extractors) in enumerate(merge_texts(per), 1):
                items.append(
                    {
                        "key": f"{b['article_id']}#{ticker}#{number}",
                        "article_id": b["article_id"],
                        "ticker": ticker,
                        "text_hash": retag.text_hash(text[: extra.TEXT_CAP]),
                        "extractors": "|".join(extractors),
                        "revisions": "|".join(extractor_revision(e) for e in extractors),
                        "title": src["title"] or "",
                        "text": text[: extra.TEXT_CAP],
                        "pub_date": src["pub_date"] or "",
                        "source_domain": src["source_domain"] or "",
                        "body": body[: extra.TEXT_CAP],
                    }
                )
    return items, omissions


def to_translate(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """The texts (key ``<id>#<ticker>#<n>``) and each article once (key ``<id>#body``)."""
    keyed = [{"article_id": i["key"], "title": i["title"], "text": i["text"]} for i in items]
    articles = {i["article_id"]: i for i in items}
    keyed += [
        {"article_id": f"{a}#body", "title": i["title"], "text": i["body"]}
        for a, i in articles.items()
    ]
    return keyed


def write_coverage(path: Path, omissions: list[dict[str, str]]) -> None:
    """The companies left without a text, also when there are none (the report needs the file)."""
    gold_metrics.write_pair_atomically({path: gold_metrics.render_csv(COVERAGE_FIELDS, omissions)})


def write_first_draw(
    csv_files: dict[Path, list[dict[str, str]]],
    coverage_path: Path,
    omissions: list[dict[str, str]],
) -> None:
    """The ids, the owner's file and the coverage of a first draw. If one cannot be written (open in
    Excel, disk full) the ones already written are put back as they were, so a half-drawn set never
    sits next to an old tagging file; ``OSError`` is then raised with what could not be undone."""
    paths = [*csv_files, coverage_path]
    before = {p: p.read_bytes() if p.exists() else None for p in paths}
    try:
        for path, rows in csv_files.items():
            extra._write(path, rows)
        write_coverage(coverage_path, omissions)
    except OSError as exc:
        stuck = []
        for path, old in before.items():
            try:
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(old)
            except OSError:
                stuck.append(path.name)
        if stuck:
            raise OSError(f"{exc}; and could not put back {', '.join(stuck)}") from None
        raise


def tag_rows(
    items: list[dict[str, str]], spanish: dict[str, dict[str, str]], names_of: dict[str, str]
) -> list[dict[str, str]]:
    """The owner's rows; ``review_order`` is the item's own when it has one (rows added later for
    another extractor), else its position."""
    return [
        {
            "review_order": i.get("review_order", str(order)),
            "article_id": i["article_id"],
            "ticker": i["ticker"],
            "company": names_of.get(i["ticker"], i["ticker"]),
            "pub_date": i["pub_date"],
            "source_domain": i["source_domain"],
            "title": i["title"],
            "text": i["text"],
            "spanish_title": spanish[i["key"]]["spanish_title"],
            "spanish_text": spanish[i["key"]]["spanish_text"],
            "body": i["body"],
            "spanish_body": spanish[f"{i['article_id']}#body"]["spanish_text"],
            "faithful": "",
            "notes": "",
        }
        for order, i in enumerate(items, 1)
    ]


def merge_new_extractor(
    ids: list[dict[str, str]], items: list[dict[str, str]], seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Add the texts of one more extractor to a file the owner has begun to mark, touching no mark.

    A text that an earlier row already holds (same article, company and ``text_hash``) only gets the
    extractor's name and revision added to that row of the ids. Any other text is a new row, with the
    next ``review_order`` and a key past the ones in use, in a shuffled order. Returns the updated
    ids and the new items (``review_order`` and ``key`` set); the earlier rows are not reordered."""
    by_text = {(r["article_id"], r["ticker"], r["text_hash"]): r for r in ids}
    new: list[dict[str, str]] = []
    for item in items:
        row = by_text.get((item["article_id"], item["ticker"], item["text_hash"]))
        if row is None:
            new.append(item)
            continue
        have = dict(zip(row["extractors"].split("|"), row["revisions"].split("|"), strict=True))
        have.update(zip(item["extractors"].split("|"), item["revisions"].split("|"), strict=True))
        row["extractors"] = "|".join(sorted(have))
        row["revisions"] = "|".join(have[n] for n in sorted(have))
    used: dict[tuple[str, str], int] = {}
    for r in ids:
        article, ticker, number = r["key"].rsplit("#", 2)
        used[(article, ticker)] = max(used.get((article, ticker), 0), int(number))
    random.Random(seed).shuffle(new)  # noqa: S311 -- blind within the new rows
    order = max((int(r["review_order"]) for r in ids), default=0)
    out: list[dict[str, str]] = []
    for item in new:
        order += 1
        used[(item["article_id"], item["ticker"])] = number = (
            used.get((item["article_id"], item["ticker"]), 0) + 1
        )
        out.append(
            {
                **item,
                "review_order": str(order),
                "key": f"{item['article_id']}#{item['ticker']}#{number}",
            }
        )
    return ids + [{k: i[k] for k in IDS_FIELDS} for i in out], out


def add_extractor(args: argparse.Namespace) -> None:
    """``--add-extractor NAME``: bring one more extractor's texts into the file being marked."""
    out: Path = args.out
    name: str = args.add_extractor
    if name not in EXTRACTORS:
        sys.exit(
            f"no extractor {name!r}: only {sorted(EXTRACTORS)} exist now "
            "(the model needs --attribution-model)."
        )
    target, ids_path, bodies_path = out / TO_TAG_NAME, out / IDS_NAME, out / BODIES_NAME
    if not (extra.has_rows(target) and ids_path.exists() and bodies_path.exists()):
        sys.exit(
            f"--add-extractor needs {target.name} (with rows), {IDS_NAME} and {BODIES_NAME} from a "
            "first draw; without them draw normally."
        )
    universe = retag_cli.companies()
    items, omissions = collect_items(extra._read(bodies_path), [name], CompanyResolver(universe))
    ids, new = merge_new_extractor(extra._read(ids_path), items, args.seed)
    print(
        f"{len(items)} texts of {name}: {len(items) - len(new)} already in the file, {len(new)} new"
    )
    kept = extra._read(target)
    backup = target.with_name(f"{target.stem}.before_{name}{target.suffix}")
    if backup.exists():
        sys.exit(
            f"{backup} exists: the add of {name} was interrupted. Compare it with {target.name}: if "
            f"{target.name} has the same marks, delete the copy and run again; if marks are "
            f"missing, copy it over {target.name} first."
        )
    done = retag_cli.add_spanish(
        out / CACHE_DIR,
        to_translate(new),
        retag_cli._translator(),
        yes=args.yes,
        max_new=args.max_new,
    )
    names_of = {c.ticker: c.name for c in universe}
    added = tag_rows(new, {r["article_id"]: r for r in done}, names_of)
    coverage_path = out / COVERAGE_NAME
    try:
        old_coverage = gold_metrics.read_coverage(coverage_path)
    except gold_metrics.MarkedFileError as exc:
        sys.exit(str(exc))
    old_ids = extra._read(ids_path)
    known = {tuple(r[k] for k in COVERAGE_FIELDS) for r in old_coverage}
    coverage = old_coverage + [
        o for o in omissions if tuple(o[k] for k in COVERAGE_FIELDS) not in known
    ]
    try:
        retag.require_tagging_format(added, target.name, retag.FAITHFULNESS_COLUMNS)
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    try:
        shutil.copy2(target, backup)  # the owner's marks, as they were
        # The tagging file goes first: if it cannot be written (open in Excel) nothing else changed.
        extra._write(target, retag.spreadsheet_safe(kept + added))  # idempotent on earlier rows
    except OSError as exc:
        backup.unlink(
            missing_ok=True
        )  # the tagging file was not touched: the next run starts clean
        sys.exit(f"cannot write {target.name} ({exc}); close it in Excel and run again")
    try:
        extra._write(ids_path, ids)
        write_coverage(coverage_path, coverage)
    except OSError as exc:
        # The tagging file already has the new rows; put the three back as they were. If that fails
        # too, the copy stays and the message says what state each file is in.
        try:
            shutil.copy2(backup, target)
            extra._write(ids_path, old_ids)
            write_coverage(coverage_path, old_coverage)
        except OSError as again:
            sys.exit(
                f"cannot write {ids_path.name} or {COVERAGE_NAME} ({exc}) and could not undo "
                f"({again}). {target.name} may already hold the new rows; {ids_path.name} and "
                f"{COVERAGE_NAME} may be old or half-written. The marks as they were are in "
                f"{backup.name}: do not delete it. Fix the cause and run again."
            )
        backup.unlink(missing_ok=True)
        sys.exit(f"cannot write {ids_path.name} or {COVERAGE_NAME} ({exc}); nothing was added")
    print(f"{len(added)} new texts appended to {target} (marks untouched; copy in {backup.name})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--extractors", default="baseline,model")
    parser.add_argument(
        "--add-extractor",
        metavar="NAME",
        help="add this extractor's texts to the file already being marked (no mark is touched)",
    )
    parser.add_argument(
        "--attribution-model",
        metavar="DIR_OR_ID",
        help="the trained attribution model (T-148b3): a local directory, or a Hub id with "
        "--attribution-revision; without it only the baseline exists",
    )
    parser.add_argument("--attribution-revision", metavar="SHA")
    parser.add_argument(
        "--stale",
        action="store_true",
        help="which marked rows the code no longer writes; exit code 1 if any, +2 if an extractor of "
        "the ids was not run (model and units need --attribution-model; --only limits the check); "
        "updates faithfulness_stale.json",
    )
    parser.add_argument(
        "--only",
        metavar="NAMES",
        help="with --stale: check only these extractors (comma-separated); the others of the ids "
        "are left out, not counted as unchecked",
    )
    parser.add_argument(
        "--exclude-from",
        type=Path,
        metavar="DIR",
        help="a confirmation draw: leave out the bodies of the draw in DIR (and use its held-out "
        "list); write into --out, a new directory",
    )
    parser.add_argument(
        "--exempt",
        "--footer-only",
        dest="exempt",
        action="store_true",
        help="write the companies each drawn body names that the units extractors are not asked "
        "about (footer, guard) and merge them into the file for the owner to read; --yes pays for "
        "the Spanish of its empty cells",
    )
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    return parser.parse_args()


def main() -> None:
    db.env.load_env_file()  # .env: these scripts do not import pipeline, which loads it elsewhere
    args = parse_args()
    wanted = (
        {"model"}
        if args.stale
        else {args.add_extractor}
        if args.add_extractor
        else set(args.extractors.split(","))
    )
    if args.attribution_model and {"model", *UNITS_READINGS} & {w.strip() for w in wanted}:
        register_model(
            args.attribution_model,
            args.attribution_revision,
            {c.ticker: c.name for c in retag_cli.companies()},
        )
    if args.add_extractor:
        add_extractor(args)
        return
    out: Path = args.out
    if args.stale:
        sys.exit(run_stale(args))
    if args.exempt:
        write_exempt(out, yes=args.yes, max_new=args.max_new)
        return
    names = [e.strip() for e in args.extractors.split(",") if e.strip()]
    missing = [e for e in names if e not in EXTRACTORS]
    if missing:
        sys.exit(
            f"no extractor {missing}: only {sorted(EXTRACTORS)} exist now. The trained model "
            "(T-148b3) needs --attribution-model; use --extractors baseline for the decision "
            "on the baseline (D34)."
        )
    target = out / TO_TAG_NAME
    if extra.has_rows(target):
        sys.exit(f"{target} exists: it is your tagging file and is not overwritten")
    universe = retag_cli.companies()
    resolver = CompanyResolver(universe)
    bodies_path = out / BODIES_NAME
    if bodies_path.exists():
        bodies = extra._read(bodies_path)
        print(f"{bodies_path} exists: the bodies are not drawn again")
    else:
        candidates = candidate_bodies(out, resolver, args.exclude_from)
        bodies = draw_bodies(candidates, args.n, args.seed)
        extra._write(bodies_path, bodies)
        one = sum(b["segments"] == "1" for b in bodies)
        print(
            f"{len(candidates)} narrative multi bodies; {len(bodies)} drawn ({one} one paragraph)"
        )
    items, omissions = collect_items(bodies, names, resolver)
    unextracted = Counter(o["extractor"] for o in omissions)
    random.Random(args.seed).shuffle(items)  # noqa: S311 -- blind: the order hides the extractor
    print("texts for the owner:", len(items), "| left without a text:", dict(unextracted))
    cache = out / CACHE_DIR
    cache.mkdir(exist_ok=True)
    done = retag_cli.add_spanish(
        cache, to_translate(items), retag_cli._translator(), yes=args.yes, max_new=args.max_new
    )
    rows = tag_rows(items, {r["article_id"]: r for r in done}, {c.ticker: c.name for c in universe})
    try:
        retag.require_tagging_format(rows, target.name, retag.FAITHFULNESS_COLUMNS)
        ids = [
            {"review_order": str(n), **{k: i[k] for k in IDS_FIELDS[1:]}}
            for n, i in enumerate(items, 1)
        ]
        write_first_draw(
            {out / IDS_NAME: ids, target: retag.spreadsheet_safe(rows)},
            out / COVERAGE_NAME,
            omissions,
        )
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    except OSError as exc:
        sys.exit(f"cannot write the draw's files ({exc}); close them in Excel and run again")
    print(f"{len(rows)} texts -> {target}")


if __name__ == "__main__":
    main()
