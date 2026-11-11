"""T-147, the owner's two-stage route (2026-11-10): topic first, then single or multi among the companies.

    uv run python scripts/scope_two_stage_2026_11_10.py [--features data/scope_relevance_features.json] [--chars 800]

Stage A: an embedding model (``--embedder``: bge small, base or large, pinned) turns the title + first ``--chars`` characters into a
vector; a logistic regression (C = 10, fixed in advance) sorts it into ``company`` (single or multi), ``sector`` or
``not_related`` (market_economy, investing_general, unrelated). Stage B: the relevance features of
``scope_relevance_features_2026_11_10.py`` (read from its cache), a logistic regression trained **only on the company
bodies** of the 284, decides ``single`` or ``multi`` for the bodies stage A calls ``company``. Scored on the 200 held-out
the way the other scope scripts are (weighted to the corpus, four classes), with each stage apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scope_relevance_features_2026_11_10 as feat
import scope_title_baseline_2026_11_10 as base

#: The pinned embedders tried (the smallest is the first measurement, the others the owner's "larger model").
EMBEDDERS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
}
PCA_COMPONENTS = (8, 16, 32, 64, 128)
TOPIC = {"single": "company", "multi": "company", "sector": "sector"}


def topic(scope: str) -> str:
    return TOPIC.get(scope, "not_related")


def embed(texts: list[str], embedder: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    import torch  # noqa: PLC0415 -- the heavy imports only where the model runs
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    revision = EMBEDDERS[embedder]
    tok = AutoTokenizer.from_pretrained(embedder, revision=revision)
    model = AutoModel.from_pretrained(embedder, revision=revision).cuda().eval()
    out = []
    for i in range(0, len(texts), 32):
        x = tok(
            texts[i : i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            cls = model(**{k: v.cuda() for k, v in x.items()}).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.vstack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=Path("data/scope_relevance_features.json"))
    parser.add_argument("--chars", type=int, default=800)
    parser.add_argument(
        "--pca",
        action="store_true",
        help="PCA on the embedding; components and C chosen by 5-fold CV on the 284 only",
    )
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()
    sets = {}
    for name, marks, texts in (
        ("train", "category_gold_v2.csv", "category_gold_v2_texts.csv"),
        ("test", "category_gold_v2_test.csv", "category_gold_v2_test_texts.csv"),
    ):
        text_of = {
            (r["article_id"], r["ticker"]): r for r in base.read(feat.GOLD / "texts" / texts)
        }
        sets[name] = [
            {**r, **text_of[(r["article_id"], r["ticker"])]} for r in base.read(feat.GOLD / marks)
        ]
    train, test = sets["train"], sets["test"]
    cached = json.loads(args.features.read_text(encoding="utf-8"))["features"]
    names = feat.FEATURES
    f_tr = np.array([[f[n] for n in names] for f in cached["train"]])
    f_te = np.array([[f[n] for n in names] for f in cached["test"]])
    vec = embed([feat.fragment(r, args.chars) for r in train + test], args.embedder)
    e_tr, e_te = vec[: len(train)], vec[len(train) :]
    # stage A: topic
    y_topic = [topic(r["scope"]) for r in train]
    if args.pca:
        # no label is used by PCA, and it is fitted inside each fold; components and C are chosen by cross-validated
        # topic accuracy on the 284, and the 200 are scored once with the winner
        folds = StratifiedKFold(5, shuffle=True, random_state=1)
        grid = [(k, c) for k in PCA_COMPONENTS for c in (1.0, 10.0)]
        cv = {
            g: cross_val_score(
                make_pipeline(PCA(g[0], random_state=1), LogisticRegression(max_iter=3000, C=g[1])),
                e_tr,
                y_topic,
                cv=folds,
            ).mean()
            for g in grid
        }
        k, c = max(cv, key=lambda g: cv[g])
        print(
            "cross-validated topic accuracy on the 284 (components, C):",
            {g: round(v, 3) for g, v in cv.items()},
        )
        print(f"chosen: {k} components, C = {c}")
        a_model = make_pipeline(PCA(k, random_state=1), LogisticRegression(max_iter=3000, C=c)).fit(
            e_tr, y_topic
        )
    else:
        a_model = LogisticRegression(max_iter=3000, C=10.0).fit(e_tr, y_topic)
    a_pred = [str(p) for p in a_model.predict(e_te)]
    # stage B: single or multi, trained on the company bodies only
    company = np.array([topic(r["scope"]) == "company" for r in train])
    scaler = StandardScaler().fit(f_tr[company])
    b_model = LogisticRegression(max_iter=2000, C=1.0).fit(
        scaler.transform(f_tr[company]),
        [r["scope"] for r, c in zip(train, company, strict=True) if c],
    )
    b_pred = [str(p) for p in b_model.predict(scaler.transform(f_te))]
    pred = [
        b if a == "company" else ("sector" if a == "sector" else "out_of_scope")
        for a, b in zip(a_pred, b_pred, strict=True)
    ]
    print(f"two-stage ({args.embedder}):")
    print("\n".join(base.lines("end to end", test, pred)), end="\n\n")
    gold_topic = [topic(r["scope"]) for r in test]
    print(
        "stage A confusion (rows = gold topic, columns = predicted), order company / sector / not_related"
    )
    order = ("company", "sector", "not_related")
    count = Counter(zip(gold_topic, a_pred, strict=True))
    for g in order:
        print(f"{g:>12}", [count[(g, p)] for p in order])
    own = [
        (r["scope"], b) for r, g, b in zip(test, gold_topic, b_pred, strict=True) if g == "company"
    ]
    print(
        f"stage B on the {len(own)} bodies that really are company: plain agreement {np.mean([g == b for g, b in own]):.3f}"
    )
    count = Counter(own)
    for g in ("single", "multi"):
        print(f"{g:>12}", [count[(g, p)] for p in ("single", "multi")])


if __name__ == "__main__":
    main()
