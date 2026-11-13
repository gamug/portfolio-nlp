"""Add the Spanish aid to the owner's mark sheets of 2026-11-13 (main companies, ignored spans, summary slice): one translated column after each English one.

    uv run scripts/translate_marks_sheets_2026_11_13.py [--yes] [--max-new 400]

Paid LLM calls (``LLM_*`` of ``.env``), with the translator and the cache of the other tagging files (``news_nlp.translate``): a text already in the
cache is not paid for again. A sheet is copied to ``<name>.before_translate.csv`` first, a sheet that already has its Spanish columns is skipped, and
every other cell is kept as it is. Nothing is bought without ``--yes``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retag_gold_2026_10_29 as retag_cli  # noqa: E402

import news_nlp as db  # noqa: E402
from news_nlp import translate  # noqa: E402

#: sheet -> ((English column, Spanish column), ...)
SHEETS: dict[Path, tuple[tuple[str, str], ...]] = {
    Path("main_company") / "to_tag.csv": (("texto", "texto_es"),),
    Path("main_company") / "spans_to_tag.csv": (("contexto", "contexto_es"),),
    Path("summary_judge") / "to_tag.csv": (
        ("texto_fuente", "texto_fuente_es"),
        ("resumen", "resumen_es"),
    ),
}


def key_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def with_columns(
    rows: list[dict[str, str]],
    pairs: tuple[tuple[str, str], ...],
    done: dict[str, translate.Translation],
) -> list[dict[str, str]]:
    """The rows with each Spanish column right after its English one."""
    out: list[dict[str, str]] = []
    for row in rows:
        new: dict[str, str] = {}
        for column, value in row.items():
            new[column] = value
            for english, spanish in pairs:
                if column == english:
                    hit = done.get(key_of(value))
                    new[spanish] = hit.text if hit else ""
        out.append(new)
    return out


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--max-new", type=int, default=400)
    args = parser.parse_args()
    loaded: dict[Path, list[dict[str, str]]] = {}
    texts: dict[str, str] = {}
    for name, pairs in SHEETS.items():
        path = args.data / name
        with path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if pairs[0][1] in rows[0]:
            print(f"{path}: already has its Spanish columns")
            continue
        loaded[path] = rows
        for row in rows:
            for english, _ in pairs:
                if row[english].strip():
                    texts[key_of(row[english])] = row[english]
    if not loaded:
        return
    translator = retag_cli._translator()
    cache = translate.TranslationCache(args.data / "marks_translations_es.csv")
    todo_rows = [{"article_id": k, "title": "-", "text": t} for k, t in texts.items()]
    retag_cli.confirm_spend(
        len(translate.pending(todo_rows, cache, translator.tag)), yes=args.yes, max_new=args.max_new
    )
    done = translate.translate_rows(
        todo_rows, translator.translate, cache, workers=8, tag=translator.tag
    )
    retag_cli._report(done, translator)
    if done.failures or done.aborted:
        sys.exit("not written: run it again (only the missing texts are paid for)")
    for path, rows in loaded.items():
        shutil.copy2(path, path.with_suffix(".before_translate.csv"))
        result = with_columns(rows, SHEETS[path.relative_to(args.data)], done.translations)
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
        print(f"{len(result)} rows -> {path}")


if __name__ == "__main__":
    main()
