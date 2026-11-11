"""Step 1 of the stepped pipeline: the scope of a news body, from two fine-tuned NLI models (T-147 selected, T-163).

Stage A reads the title + the first 800 characters against three hypotheses (company / sector / not_related); stage B, the same
text against two (single / multi), is asked only of the bodies stage A calls ``company``. The scope is ``sector`` when A says
sector, ``out_of_scope`` when A says not_related, and B's ``single`` or ``multi`` otherwise. The models, the hypotheses and the
recipe come from the ``training_record.json`` next to the weights (``scripts/train_scope_final_2026_11_11.py``); the
revision stored with every result is the hash of that record, so a retrained model is a new revision.

``docs/splitting_training.md`` tells how the models were chosen and what they get wrong (an out_of_scope body is often read as
single: the accepted trade-off).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BATCH = 32


@dataclass(frozen=True)
class ScopeResult:
    scope: str
    scores: dict[str, float]  # the entailment margin of every hypothesis asked


def combine(topic: str, single_multi: str | None) -> str:
    """Stage A's topic and, for a company body, stage B's answer into the four-class scope."""
    if topic == "sector":
        return "sector"
    if topic == "not_related":
        return "out_of_scope"
    if single_multi not in ("single", "multi"):
        raise ValueError(f"a company body needs stage B's answer, got {single_multi!r}")
    return single_multi


def fragment(title: str, text: str, chars: int) -> str:
    return f"{title}. {text[:chars]}"


def revision_of(model_dir: Path) -> str:
    """``scope-nli-<12 hex>``: the hash of the training record (data, recipe, base revision) and of the saved weights.

    GPU training is not bit for bit, so two fits with one record give different weights; the weight files are part of the
    revision so that a refit is a new revision and its bodies are done again.
    """
    digest = hashlib.sha256((model_dir / "training_record.json").read_bytes())
    for weights in sorted(model_dir.glob("stage_*/*.safetensors")):
        digest.update(weights.relative_to(model_dir).as_posix().encode())
        with weights.open("rb") as fh:
            while chunk := fh.read(1 << 24):
                digest.update(chunk)
    return "scope-nli-" + digest.hexdigest()[:12]


def assemble(
    record: dict[str, Any],
    topic_margins: dict[str, list[float]],
    class_margins: dict[str, list[float]],
    topics: list[str],
    company: list[int],
) -> list[ScopeResult]:
    """The four-class scope of every text from the stage margins.

    ``company`` lists the positions stage A called company; ``class_margins`` holds stage B's margins for those only, in that order.
    """
    a, b = record["stage_a"], record["stage_b"]
    row_of = {i: j for j, i in enumerate(company)}
    results: list[ScopeResult] = []
    for i, topic in enumerate(topics):
        scores = {f"topic:{name}": topic_margins[name][i] for name in a["topics"]}
        second: str | None = None
        if topic == "company":
            j = row_of[i]
            scores.update({f"count:{name}": class_margins[name][j] for name in b["classes"]})
            second = max(b["classes"], key=lambda name: class_margins[name][j])
        results.append(ScopeResult(combine(topic, second), scores))
    return results


PIN_PATH = Path(__file__).resolve().parent.parent / "config" / "scope_model.json"


def pin_status(revision: str, pin_path: Path = PIN_PATH) -> str:
    """One line saying whether *revision* is the pinned scope model (``config/scope_model.json``, T-147b)."""
    if not pin_path.exists():
        return f"model revision {revision}: no pin recorded"
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    if pin.get("step_revision") == revision:
        return f"model revision {revision}: pinned ({pin['model']}@{pin['revision'][:8]})"
    return (
        f"model revision {revision}: NOT the pinned one ({pin.get('step_revision')}); "
        "the results are not traceable to the Hub copy"
    )


def fetch_pinned(model_dir: Path, pin_path: Path = PIN_PATH) -> str:
    """Download the pinned scope model from the Hub into *model_dir* and check it is the pinned step revision.

    The repository is private: ``HF_TOKEN`` (read scope) must be set. Returns the step revision; raises when the downloaded
    files do not hash to the revision recorded in the pin (a wrong or altered copy is never used silently).
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415 -- only the download needs it

    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    snapshot_download(pin["model"], revision=pin["revision"], local_dir=str(model_dir))
    got = revision_of(model_dir)
    if got != pin["step_revision"]:
        raise RuntimeError(
            f"{pin['model']}@{pin['revision'][:8]} downloaded to {model_dir} has the step revision {got}, "
            f"not the pinned {pin['step_revision']}"
        )
    return got


class ScopeClassifier:
    """Both stages loaded on the GPU; ``classify`` reads a batch of (title, body) pairs."""

    def __init__(self, model_dir: Path) -> None:
        import torch  # noqa: PLC0415 -- heavy imports only where the models run
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("the scope step needs a GPU (CUDA is not available)")
        self.record: dict[str, Any] = json.loads(
            (model_dir / "training_record.json").read_text(encoding="utf-8")
        )
        self.revision = revision_of(model_dir)
        self.chars = int(self.record["fragment_chars"])
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir / "stage_a")
        self.a = (
            AutoModelForSequenceClassification.from_pretrained(
                model_dir / "stage_a", dtype=torch.float32
            )
            .cuda()
            .eval()
        )
        self.b = (
            AutoModelForSequenceClassification.from_pretrained(
                model_dir / "stage_b", dtype=torch.float32
            )
            .cuda()
            .eval()
        )

    def _margins(
        self, model: Any, texts: Sequence[str], hypotheses: dict[str, str], max_length: int
    ) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {name: [] for name in hypotheses}
        for name, hypothesis in hypotheses.items():
            for i in range(0, len(texts), BATCH):
                chunk = list(texts[i : i + BATCH])
                x = self.tok(
                    chunk,
                    [hypothesis] * len(chunk),
                    truncation="only_first",
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
                with (
                    self._torch.no_grad(),
                    self._torch.autocast("cuda", dtype=self._torch.bfloat16),
                ):
                    logits = model(**{k: v.cuda() for k, v in x.items()}).logits.float()
                out[name].extend((logits[:, 0] - logits[:, 1]).cpu().tolist())
        return out

    def classify(self, items: Sequence[tuple[str, str]]) -> list[ScopeResult]:
        texts = [fragment(t, b, self.chars) for t, b in items]
        a = self.record["stage_a"]
        am = self._margins(self.a, texts, a["hypotheses"], int(a["recipe"]["max_length"]))
        topics = [max(a["topics"], key=lambda name: am[name][i]) for i in range(len(texts))]
        company = [i for i, t in enumerate(topics) if t == "company"]
        b = self.record["stage_b"]
        bm = self._margins(
            self.b, [texts[i] for i in company], b["hypotheses"], int(b["recipe"]["max_length"])
        )
        return assemble(self.record, am, bm, topics, company)
