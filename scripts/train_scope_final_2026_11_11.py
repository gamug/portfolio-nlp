"""T-163: fit the two scope models of the selected classifier and save them (owner 2026-11-11, step 1 of the pipeline).

    uv run python scripts/train_scope_final_2026_11_11.py [--out models/scope_final] [--seed 1]

The recipe and the 365 training bodies are the ones ``docs/splitting_training.md`` measured (83.0% on the held-out 200, which
stay unseen): stage A (company / sector / not_related) on all 365, stage B (single / multi) on the 210 company bodies, both from
the pinned category model, float32 weights, bf16 autocast, seed 1. Writes ``<out>/stage_a`` and ``<out>/stage_b`` (weights,
tokenizer) and ``<out>/training_record.json`` (base revision, rows per class, recipe, seed, git commit). Not pushed anywhere.
The step's revision hashes the record and the weight files, so a refit (GPU training is not bit for bit) is a new revision.
Takes about 5 minutes on a 6 GB RTX 4050 (see ``docs/splitting_training.md``, "Step 1").
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scope_end_to_end_2026_11_10 as e2e
import scope_nli_single_multi_2026_11_10 as sm
import scope_nli_topic_2026_11_10 as tp

import scope_stage

FRAGMENT = 800
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import torch  # noqa: PLC0415 -- heavy imports only where the models run
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "scope_final")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("training needs a GPU (CUDA is not available)")
    sets = e2e.load_sets()
    train = sets["train"] + sets["enrich"] + sets["hook"]
    texts = [
        scope_stage.fragment(r["title"], r["text"], FRAGMENT) for r in train
    ]  # the runner's own function
    cfg = json.loads((ROOT / "config" / "pipeline_models.json").read_text(encoding="utf-8"))[
        "category"
    ]
    print(
        f"{len(train)} bodies: {dict(Counter(r['scope'] for r in train))}; "
        f"sets {len(sets['train'])} + {len(sets['enrich'])} + {len(sets['hook'])}",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])

    def load():
        return AutoModelForSequenceClassification.from_pretrained(
            cfg["model"], revision=cfg["revision"], dtype=torch.float32
        ).cuda()

    args.out.mkdir(parents=True, exist_ok=True)
    model = load()
    tp.fine_tune(model, tok, texts, [e2e.topic_of_scope(r) for r in train], seed=args.seed)
    model.save_pretrained(args.out / "stage_a")
    tok.save_pretrained(args.out / "stage_a")
    print("stage A saved", flush=True)
    del model
    torch.cuda.empty_cache()
    own = [i for i, r in enumerate(train) if r["scope"] in sm.CLASSES]
    model = load()
    sm.fine_tune(
        model, tok, [texts[i] for i in own], [train[i]["scope"] for i in own], seed=args.seed
    )
    model.save_pretrained(args.out / "stage_b")
    tok.save_pretrained(args.out / "stage_b")
    (args.out / "training_record.json").write_text(
        json.dumps(
            {
                "base": f"{cfg['model']}@{cfg['revision']}",
                "bodies": len(train),
                "company_bodies": len(own),
                "scope_counts": dict(Counter(r["scope"] for r in train)),
                "sets": ["category_gold_v2", "scope_enrich", "scope_hook"],
                "fragment_chars": FRAGMENT,
                "stage_a": {"topics": tp.TOPICS, "hypotheses": tp.HYPOTHESES, "recipe": tp.RECIPE},
                "stage_b": {
                    "classes": sm.CLASSES,
                    "hypotheses": sm.HYPOTHESES,
                    "recipe": sm.RECIPE,
                },
                "seed": args.seed,
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git from PATH
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                    check=False,
                ).stdout.strip(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"stage B saved; record -> {args.out / 'training_record.json'}")


if __name__ == "__main__":
    main()
