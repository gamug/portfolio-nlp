"""The owner's re-tag of the 720 judged articles under the new taxonomy (Work item 18, T-146b).

    uv run scripts/retag_gold_2026_10_29.py export [--out data]   # writes the file to tag
    uv run scripts/retag_gold_2026_10_29.py translate [--limit N]  # spanish_title/text after text
    uv run scripts/retag_gold_2026_10_29.py guide  [--out data]   # rewrites the guide CSVs only
    uv run scripts/retag_gold_2026_10_29.py normalise [--out data]  # names -> tickers, new file
    uv run scripts/retag_gold_2026_10_29.py check  [--out data] [--file NAME]  # validates a file

``export`` reads ``data/category_gold_review.csv`` and its key (tags v1) and writes
``category_gold_v2_review.csv`` (the article, the v1 label and note and the scope they imply as
guidance, EMPTY ``scope`` / ``companies`` / ``company_labels`` / ``mentions`` / ``sub_industry`` /
``notes``) and
``category_gold_v2_key.csv`` (the v1 key, where the LLM's label stays: the owner is not shown it).
It **refuses to overwrite a file that already has scopes filled in**. ``check`` lists the
untagged rows and every row that breaks the rules (``news_nlp.retag.check_row``).
Column format: ``companies`` = ``AAPL; MSFT``; ``company_labels`` = ``AAPL:product_innovation; MSFT:other``.
The owner may type names or nicknames instead of tickers (``Apple; Google``, ``Apple:other``):
``normalise`` writes ``category_gold_v2_review_tickers.csv`` with tickers and lists every text it
could not resolve or that fits several companies (left as typed, never guessed), and the matches
it inferred (a first word, a brand, a ticker that is also a word) for the owner to confirm; the
review file is not changed. ``check --file category_gold_v2_review_tickers.csv`` then validates that copy.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import company_names, retag, translate
from news_nlp.gold_review import TaggedFileError, tagged_rows, write_review_files
from news_nlp.triage import Company

DEFAULT_MAX_NEW = 800  # the most new translations one run may buy without raising --max-new


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _translator() -> translate.LLMTranslator:
    from dotenv import load_dotenv  # noqa: PLC0415 -- only the translating commands need it

    load_dotenv()
    return translate.LLMTranslator.from_env()


def cmd_export(
    out: Path,
    *,
    universe_companies: list[Company] | None = None,
    translator: translate.LLMTranslator | None = None,
    spanish: bool = True,
    yes: bool = False,
    max_new: int = DEFAULT_MAX_NEW,
) -> None:
    """Export the file to tag. It carries ``spanish_title`` / ``spanish_text`` after ``text``
    (translated once per article, cached) unless ``spanish`` is False."""
    review, key = retag.retag_rows(
        _read(out / "category_gold_review.csv"), _read(out / "category_gold_review_key.csv")
    )
    if spanish:
        if tagged_rows(out / retag.REVIEW_NAME, "scope"):  # refuse before paying for anything
            sys.exit(f"{out / retag.REVIEW_NAME} is already tagged; refusing to overwrite it")
        rows = [{k: str(v) for k, v in r.items()} for r in review]
        got = add_spanish(out, rows, translator or _translator(), yes=yes, max_new=max_new)
        review = got
    try:
        write_review_files(
            out,
            review,
            key,
            review_name=retag.REVIEW_NAME,
            key_name=retag.KEY_NAME,
            tag_column="scope",
        )
    except TaggedFileError as exc:
        sys.exit(f"{exc}. Copy it elsewhere or pass another --out directory.")
    write_guides(out, universe_companies)
    print(
        f"{len(review)} articles -> {out / retag.REVIEW_NAME}; guide and sub-industry list written"
    )


def confirm_spend(new: int, *, yes: bool, max_new: int) -> None:
    """Paid LLM calls need an explicit go: print what is about to be bought and stop unless the
    owner passed ``--yes``; refuse a run above ``--max-new`` unless it was raised on purpose."""
    if new == 0:
        return
    prompt, completion = translate.estimate(new)
    print(
        f"{new} articles to translate: paid LLM calls, about {prompt:,} prompt and "
        f"{completion:,} completion tokens"
    )
    if new > max_new:
        sys.exit(f"{new} is above --max-new {max_new}: raise it on purpose to go on")
    if not yes:
        sys.exit("nothing was bought: pass --yes to pay for them")


def _require_adopted(cache: translate.TranslationCache) -> None:
    if cache.untagged():
        sys.exit(
            f"{cache.untagged()} cached translations have no model/prompt tag: run "
            "'translate --adopt-legacy --yes' once (no cost) if the current model and prompt made them"
        )


def cmd_adopt_legacy(
    out: Path, *, translator: translate.LLMTranslator | None = None, yes: bool = False
) -> None:
    """Stamp the untagged cached translations with the current model and prompt tag. Only for rows
    known to have been made with them (the 720 of 2026-10-30 were); no API call is made. It says
    which model, tag and how many first and does nothing without ``yes``."""
    translator = translator or _translator()
    cache = translate.TranslationCache(out / translate.CACHE_NAME)
    n = cache.untagged()
    if n == 0:
        print("no untagged cached translations")
        return
    print(
        f"{n} cached translations will be recorded as made by model "
        f"{getattr(translator, 'model', '?')!r} with prompt/model tag {translator.tag}"
    )
    if not yes:
        sys.exit("nothing changed: pass --yes if that model and prompt made them")
    cache.adopt_legacy(translator.tag)
    print(f"{n} cached translations stamped")


def _report(done: translate.Outcome, translator: translate.LLMTranslator) -> None:
    stopped = f"; STOPPED: {done.aborted}" if done.aborted else ""
    print(
        f"translated {done.translated} new articles, {len(done.failures)} failed{stopped}. "
        f"Paid calls this run: {translator.calls} ({translator.prompt_tokens} prompt and "
        f"{translator.completion_tokens} completion tokens, retries and rejected replies included)"
    )
    for article_id, why in sorted(done.failures.items()):
        print(f"  article {article_id}: {why}")


def add_spanish(
    out: Path,
    rows: list[dict[str, str]],
    translator: translate.LLMTranslator,
    *,
    limit: int | None = None,
    workers: int = 8,
    yes: bool = False,
    max_new: int = DEFAULT_MAX_NEW,
) -> list[dict[str, str]]:
    """The rows with the Spanish columns after ``text``; translations are cached in ``out`` and
    the token use of the new ones is printed. Nothing is bought without ``yes``. If any article
    failed or the run stopped, it exits non-zero before anything is written; the cache keeps the
    rest, so a re-run pays only for the missing ones."""
    cache = translate.TranslationCache(out / translate.CACHE_NAME)
    _require_adopted(cache)
    todo = translate.pending(rows, cache, translator.tag)
    confirm_spend(
        min(len(todo), limit) if limit is not None else len(todo), yes=yes, max_new=max_new
    )
    done = translate.translate_rows(
        rows, translator.translate, cache, workers=workers, limit=limit, tag=translator.tag
    )
    _report(done, translator)
    if done.failures or done.aborted:
        sys.exit("not written: run it again (only the missing articles are paid for)")
    return translate.with_spanish(rows, done.translations)


def cmd_translate(
    out: Path,
    *,
    translator: translate.LLMTranslator | None = None,
    file: str | None = None,
    limit: int | None = None,
    workers: int = 8,
    yes: bool = False,
    max_new: int = DEFAULT_MAX_NEW,
) -> None:
    """Add ``spanish_title`` / ``spanish_text`` after ``text`` to an existing review file. With
    ``limit`` it is a pilot: the first *limit* articles are translated, printed and cached, and the
    file is not touched. Otherwise the file is backed up to ``backup/`` (with the time in the name,
    never over an earlier backup) and rewritten with every other cell as it was."""
    source = out / (file or retag.REVIEW_NAME)
    try:
        rows = retag.read_tag_file(source)
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    translator = translator or _translator()
    if limit is not None:
        cache = translate.TranslationCache(out / translate.CACHE_NAME)
        _require_adopted(cache)
        todo = translate.pending(rows, cache, translator.tag)
        confirm_spend(min(len(todo), limit), yes=yes, max_new=max_new)
        done = translate.translate_rows(
            rows, translator.translate, cache, limit=limit, tag=translator.tag
        )
        for article_id, pair in list(done.translations.items())[:limit]:
            print(f"--- article {article_id}\n{pair.title}\n{pair.text[:700]}")
        _report(done, translator)
        if done.failures or done.aborted:
            sys.exit("pilot failed")
        return
    result = add_spanish(out, rows, translator, workers=workers, yes=yes, max_new=max_new)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup = out / "backup" / f"{source.stem}_before_translate_{stamp}.csv"
    backup.parent.mkdir(parents=True, exist_ok=True)
    for n in range(2, 100):  # never over an earlier backup, even within the same second
        if not backup.exists():
            break
        backup = backup.with_name(f"{source.stem}_before_translate_{stamp}_{n}.csv")
    shutil.copy2(source, backup)
    write_csv(source, result)
    print(f"{len(result)} rows -> {source} (backup: {backup})")


def write_guides(out: Path, universe_companies: list[Company] | None = None) -> None:
    """The labels guide and the GICS sub-industry list: code-generated, safe to rewrite while the
    review file is being tagged (``cmd_export`` refuses to touch that one)."""
    write_csv(out / "category_gold_v2_guide.csv", retag.guide_rows())
    write_csv(
        out / "gics_sub_industries.csv", retag.sub_industry_rows(universe_companies or companies())
    )


def cmd_guide(out: Path, *, universe_companies: list[Company] | None = None) -> None:
    write_guides(out, universe_companies)
    print(f"{out / 'category_gold_v2_guide.csv'} and {out / 'gics_sub_industries.csv'} rewritten")


def companies() -> list[Company]:
    import news_nlp as db  # noqa: PLC0415 -- only these commands need the SOURCE store

    conn = db.connect_pipeline()
    rows = conn.execute(
        "SELECT DISTINCT ticker, company, gics_sector, gics_sub_industry FROM source.articles "
        "WHERE fetch_status = 'ok' ORDER BY ticker, company"
    ).fetchall()
    return [Company(r[0], r[1] or "", r[2] or "", r[3] or "") for r in rows]


def load_universe() -> retag.Universe:
    return retag.Universe.of(companies())


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    try:
        fh = path.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        sys.exit(
            f"cannot write {path}: it is open in another program (Excel?); close it and run again"
        )
    with fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


TICKERS_NAME = "category_gold_v2_review_tickers.csv"


def cmd_normalise(
    out: Path,
    *,
    universe_companies: list[Company] | None = None,
    file: str | None = None,
    force: bool = False,
    self_check: bool = False,
) -> None:
    """Write the review file with names and nicknames replaced by tickers. It refuses to replace
    a tickers file that was edited after the review file (``force`` overrides)."""
    firms = universe_companies or companies()
    if self_check:
        failures = company_names.self_check(firms, retag.Universe.of(firms).tickers)
        print(
            f"self-check: {len(failures)} names do not resolve to their own ticker",
            *failures,
            sep="\n  ",
        )
        if failures:
            sys.exit(1)
        return
    book = company_names.NameBook(firms, retag.Universe.of(firms).tickers)
    source = out / (file or retag.REVIEW_NAME)
    try:
        rows = retag.read_tag_file(source)
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    result: list[dict[str, str]] = []
    problems: list[str] = []
    confirm: list[str] = []
    for row in rows:
        new, found, inferred = company_names.normalise_row(book, row)
        result.append(new)
        tag = f"article {row.get('article_id')}"
        problems += [f"  {tag}: {p}" for p in found]
        confirm += [f"  {tag}: {c}" for c in inferred]
    target = out / TICKERS_NAME
    if (
        target.exists()
        and not force
        and target.stat().st_mtime >= source.stat().st_mtime
        and _read(target) != [{k: str(v) for k, v in r.items()} for r in result]
    ):
        sys.exit(
            f"{target} was edited after {source.name}: fix the cells in {source.name} and run "
            "normalise again, or pass --force to overwrite the copy"
        )
    write_csv(target, result)
    if problems:
        print("left as typed (fix them in the review file):", *problems, sep="\n")
    if confirm:
        print("converted by inference (please confirm):", *confirm, sep="\n")
    print(
        f"{len(result)} rows -> {out / TICKERS_NAME}; {len(problems)} texts left as typed, "
        f"{len(confirm)} to confirm"
    )


def cmd_check(
    out: Path, *, universe: retag.Universe | None = None, file: str | None = None
) -> None:
    expected = [int(r["article_id"]) for r in _read(out / retag.KEY_NAME)]
    try:
        report = retag.load_gold_v2(
            out / (file or retag.REVIEW_NAME), expected, universe or load_universe()
        )
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    print(
        f"tagged and valid {len(report.good)}, with errors {len(report.bad)}, "
        f"untagged {report.untagged}, expected {len(expected)}"
    )
    print("scopes:", dict(Counter(g.scope for g in report.good)))
    for label, ids in (
        ("missing ids", report.missing),
        ("duplicated ids", report.duplicated),
        ("unknown ids", report.unknown),
    ):
        if ids:
            print(f"  {label}: {ids}")
    for article_id, errors in sorted(report.bad.items()):
        print(f"  article {article_id}: {'; '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("export", "guide", "normalise", "check", "translate"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--file", help="review file name inside --out (check, normalise)")
    parser.add_argument("--limit", type=int, help="translate: pilot on the first N, file untouched")
    parser.add_argument(
        "--yes", action="store_true", help="export, translate: pay for new translations"
    )
    parser.add_argument(
        "--max-new", type=int, default=DEFAULT_MAX_NEW, help="export, translate: most new articles"
    )
    parser.add_argument(
        "--adopt-legacy",
        action="store_true",
        help="translate: stamp untagged cache rows with the current model and prompt (no cost; needs --yes)",
    )
    parser.add_argument(
        "--no-spanish", action="store_true", help="export: no spanish_title / spanish_text columns"
    )
    parser.add_argument("--force", action="store_true", help="normalise: overwrite an edited copy")
    parser.add_argument(
        "--self-check", action="store_true", help="normalise: every universe name -> its ticker"
    )
    args = parser.parse_args()
    if args.command == "export":
        cmd_export(args.out, spanish=not args.no_spanish, yes=args.yes, max_new=args.max_new)
    elif args.command == "translate" and args.adopt_legacy:
        cmd_adopt_legacy(args.out, yes=args.yes)
    elif args.command == "translate":
        cmd_translate(
            args.out, file=args.file, limit=args.limit, yes=args.yes, max_new=args.max_new
        )
    elif args.command == "guide":
        cmd_guide(args.out)
    elif args.command == "normalise":
        cmd_normalise(args.out, file=args.file, force=args.force, self_check=args.self_check)
    else:
        cmd_check(args.out, file=args.file)


if __name__ == "__main__":
    main()
