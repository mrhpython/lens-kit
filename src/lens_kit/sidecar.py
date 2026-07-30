"""Provenance-sidecar generator: a full self-identifying record next to a checkpoint.

WHY (the argument a customer reads in the README): a checkpoint is a pile of
optimized prompts. On its own it cannot tell you which model produced it, which
labeled data it was tuned and scored on, what it scored, or — most important —
what it is NOT a claim of. A bare accuracy number travels; a sidecar makes the
number refuse to travel past its evidence. The `not_a_claim_of` list is the
load-bearing field: it ships the honest boundary WITH the artifact, so a 88%
agency-holdout figure can never be quoted as a transfer promise or an accuracy
floor on a customer's own data.

This generator produces the FULL sidecar. C2 shipped a MINIMAL stub
(checkpoint.save_checkpoint, provenance_stub=True) that only guaranteed identity
+ lineage; this upgrades it. Backward compatibility is a hard contract: every
stub field is still present (same key, same meaning), `provenance_stub` flips to
False, and a richer block is added around it. Tooling that read the stub keeps
working; tooling that wants the full record gets it.

No LLM calls. Pure function of (checkpoint file on disk, profile, optional
dataset + eval-results files). Deterministic except for the `generated` date
stamp.
"""
from __future__ import annotations

import json
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

from .checkpoint import sha256_file
from .config import Profile

SIDECAR_SCHEMA_VERSION = 1

# Default not_a_claim_of entries. These are the honest-boundary defaults that
# ship with EVERY sidecar — the score-transfer ban made mechanical. A caller may
# extend this list (extra_not_a_claim_of) but may never shrink the defaults: the
# whole point is that the boundary cannot be quietly dropped to make a number
# look more impressive than its evidence.
DEFAULT_NOT_A_CLAIM_OF = (
    "transfer to any other domain or dataset",
    "an accuracy floor on customer data the gate was never tuned on",
    "concurrency, sustained load, or production scale",
    "per-call bit-reproducibility",
    # The honest sycophancy boundary (arXiv 2507.07484 / 2502.08177 / 2501.08617):
    # the kit is EXTERNALLY MEASURED against sycophancy/rubber-stamping
    # (calibration fn_rate, mutation control, variance envelopes) — NOT immune to
    # it. The gate itself runs on an LLM under a compiled reward, so it is subject
    # to approval-vs-accuracy drift; the claim is "we measure for it", never
    # "the gate cannot do it".
    "immunity from sycophancy or rubber-stamping (the kit is externally measured "
    "against it, not free of it; the gate runs on an LLM under a compiled reward)",
)

# Added ONLY when no variance envelope (reruns>1) backs the eval metrics. A
# single eval run is a point estimate, not a bound; saying so is the discipline.
SINGLE_RUN_CAVEAT = "a bound — single eval run, no variance envelope (re-run with --reruns N)"


def _dist_version(name: str) -> str | None:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def _provider_note(model: str) -> str:
    """One-line note tying a litellm model string to its provider.

    Not authoritative — the provider may silently serve a different model on the
    same id (the Stage-A substitution episode). The note records what was
    *configured*; a re_measurements entry records what was actually served.
    """
    prefix = model.split("/", 1)[0] if "/" in model else ""
    known = {
        "xai": "xAI",
        "openai": "OpenAI or any OpenAI-compatible host (check api_base)",
        "anthropic": "Anthropic",
        "gemini": "Google",
        "ollama_chat": "local Ollama",
        "ollama": "local Ollama",
    }
    who = known.get(prefix, "unknown provider (litellm-routed)")
    return (f"litellm model string '{model}' -> {who}. The served model can "
            f"differ from this id; record drift in re_measurements, not here.")


def _eval_block(eval_results_path: Path) -> dict:
    """Pull metrics + provenance out of an eval-harness results or envelope JSON.

    Accepts either a single-run eval doc (has 'summary') or an envelope doc (has
    'metrics' with per-metric stats). Returns the metrics plus the eval file's
    own identity so the sidecar's numbers are traceable to a receipt on disk.
    """
    doc = json.loads(eval_results_path.read_text())
    block: dict = {
        "eval_file": eval_results_path.name,
        "eval_file_sha256": sha256_file(eval_results_path),
    }
    if isinstance(doc, dict) and "summary" in doc:           # single-run eval doc
        block["metrics"] = doc["summary"]
        block["has_envelope"] = False
        block["reruns"] = doc.get("reruns", 1)
        block["data_file"] = doc.get("data_file")
        block["data_sha256"] = doc.get("data_sha256")
    elif isinstance(doc, dict) and "metrics" in doc:         # envelope doc
        block["metrics"] = {m: (st["worst_of_n"] if isinstance(st, dict) else st)
                            for m, st in doc["metrics"].items()}
        block["envelope"] = doc["metrics"]
        block["has_envelope"] = True
        block["reruns"] = doc.get("reruns")
        block["data_file"] = doc.get("data_file")
        block["data_sha256"] = doc.get("data_sha256")
    else:
        block["metrics"] = None
        block["has_envelope"] = False
        block["note"] = ("eval-results file had neither 'summary' nor 'metrics' "
                         "— metrics not recorded")
    return block


def _dataset_block(dataset_path: Path) -> dict:
    """Counts + labels-version for the dataset a checkpoint was tuned/scored on.

    A label edit forces a new run designation (the RUNS.md rule); recording the
    labels version + a sha here makes 'which labels produced this' answerable
    without trusting a filename.
    """
    doc = json.loads(dataset_path.read_text())
    if isinstance(doc, dict):
        examples = doc.get("examples", [])
        meta = doc.get("_metadata", {}) or {}
    else:
        examples, meta = doc, {}
    return {
        "dataset_file": dataset_path.name,
        "dataset_sha256": sha256_file(dataset_path),
        "example_count": len(examples),
        "labels_version": meta.get("labels_version") or meta.get("version"),
        "created": meta.get("created"),
    }


def build_sidecar(
    checkpoint_path: str | Path,
    *,
    profile: Profile,
    train_size: int | None = None,
    val_size: int | None = None,
    dataset_path: str | Path | None = None,
    eval_results_path: str | Path | None = None,
    extra_not_a_claim_of: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    """Build (but do NOT write) the full provenance-sidecar dict for a checkpoint.

    Backward-compatible superset of checkpoint.save_checkpoint's stub: every stub
    field is present with the same meaning; provenance_stub=False; richer blocks
    added. See module docstring for the contract.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    digest = sha256_file(checkpoint_path)

    eval_block = None
    if eval_results_path is not None:
        eval_block = _eval_block(Path(eval_results_path))

    dataset_block = None
    if dataset_path is not None:
        dataset_block = _dataset_block(Path(dataset_path))

    # not_a_claim_of: defaults are immutable; caller adds, never removes. The
    # single-run caveat is appended unless an envelope-backed eval is present.
    not_a_claim_of = list(DEFAULT_NOT_A_CLAIM_OF)
    has_envelope = bool(eval_block and eval_block.get("has_envelope"))
    if eval_block is not None and not has_envelope:
        not_a_claim_of.append(SINGLE_RUN_CAVEAT)
    for item in (extra_not_a_claim_of or []):
        if item not in not_a_claim_of:
            not_a_claim_of.append(item)

    sidecar = {
        # ── C2 stub fields (UNCHANGED keys + meaning; provenance_stub flips) ──
        "provenance_stub": False,
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "checkpoint": checkpoint_path.name,
        "sha256": digest,
        "model": profile.llm.model,
        "profile_name": profile.name,
        "dspy_version": _dist_version("dspy"),
        "gepa_version": _dist_version("gepa"),
        "train_size": train_size,
        "val_size": val_size,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # ── Full-sidecar additions ──
        "provider_note": _provider_note(profile.llm.model),
        "size_bytes": checkpoint_path.stat().st_size,
        "dependency_versions": {
            "dspy": _dist_version("dspy"),
            "gepa": _dist_version("gepa"),
            "litellm": _dist_version("litellm"),
        },
        "dataset": dataset_block,
        "eval": eval_block,
        "not_a_claim_of": not_a_claim_of,
        # Empty list to be appended-to on future drift (model substitution,
        # re-eval under new deps). Same shape as the exemplar's re_measurements.
        "re_measurements": [],
    }
    if extra:
        sidecar.update(extra)
    return sidecar


def write_sidecar(checkpoint_path: str | Path, sidecar: dict) -> Path:
    """Write a sidecar dict to <ckpt>.provenance.json. Returns the path."""
    checkpoint_path = Path(checkpoint_path)
    prov_path = checkpoint_path.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return prov_path


def generate_sidecar(
    checkpoint_path: str | Path,
    *,
    profile: Profile,
    train_size: int | None = None,
    val_size: int | None = None,
    dataset_path: str | Path | None = None,
    eval_results_path: str | Path | None = None,
    extra_not_a_claim_of: list[str] | None = None,
    extra: dict | None = None,
) -> tuple[dict, Path]:
    """Build + write the full sidecar. Returns (sidecar_dict, written_path)."""
    sidecar = build_sidecar(
        checkpoint_path, profile=profile, train_size=train_size, val_size=val_size,
        dataset_path=dataset_path, eval_results_path=eval_results_path,
        extra_not_a_claim_of=extra_not_a_claim_of, extra=extra,
    )
    path = write_sidecar(checkpoint_path, sidecar)
    return sidecar, path
