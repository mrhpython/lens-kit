"""Turn lens-kit artifact JSON into flat reviewable records.

A *record* is the review surface's unit of human attention — the analogue of
the upstream eval-viewer's "run". Each carries a stable ``id`` (used as the
feedback.json key), an artifact ``kind``, a short ``title``, and a ``data``
payload the HTML renders with kind-specific chips.

Artifact-type detection sniffs the top-level keys of each parsed object;
schema_version keys are authoritative, with a structural fallback so that
hand-trimmed fixtures (or future versions) still classify. No file is read
for its bytes beyond ``json.loads`` — the raw artifact stays out of the
record except the fields a reviewer needs.
"""
from __future__ import annotations

import json
from pathlib import Path

# Artifacts the review surface understands. Each value is the human label.
KIND_EVAL = "eval"
KIND_CALIBRATION = "calibration"
KIND_MUTATION = "mutation"

# Filenames we never treat as artifacts (the surface's own output, caches).
_EXCLUDE_NAMES = {"feedback.json"}


class ArtifactError(ValueError):
    """A results file/dir could not be parsed into review records.

    Raised for: not-found paths, unreadable/invalid JSON, and inputs that
    match no known artifact shape. The CLI maps this to exit code 2.
    """


# ── Type detection ────────────────────────────────────────────────

def detect_artifact_type(obj: object) -> str | None:
    """Classify one parsed JSON object. Returns a KIND_* or None.

    Authoritative signal: the schema_version key each writer stamps
    (eval_harness/calibration/mutation). Structural fallback keeps
    trimmed fixtures and version drift classifiable.
    """
    if not isinstance(obj, dict):
        return None

    if "eval_schema_version" in obj or "per_example" in obj:
        return KIND_EVAL
    if "mutation_schema_version" in obj:
        return KIND_MUTATION
    if "calib_schema_version" in obj:
        return KIND_CALIBRATION

    # Structural fallback: a results list whose rows reveal the kind.
    rows = obj.get("results")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        row = rows[0]
        if "caught" in row and "marker" in row:
            return KIND_MUTATION
        if "acceptable" in row and "class" in row:
            return KIND_CALIBRATION
    return None


# ── Loading ───────────────────────────────────────────────────────

def load_artifacts(path: str | Path) -> list[tuple[Path, str, dict]]:
    """Load one results file or a directory of them.

    Returns a list of ``(source_path, kind, obj)`` for every recognised
    artifact, sorted by filename. A directory with no recognised artifact,
    or a single file that doesn't classify, raises ArtifactError so the CLI
    surfaces a usage error rather than serving an empty page.
    """
    p = Path(path)
    if not p.exists():
        raise ArtifactError(f"path not found: {p}")

    files: list[Path]
    if p.is_dir():
        files = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() == ".json"
            and f.name not in _EXCLUDE_NAMES
        )
        if not files:
            raise ArtifactError(f"no .json artifacts in directory: {p}")
    else:
        files = [p]

    loaded: list[tuple[Path, str, dict]] = []
    parse_errors: list[str] = []
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            parse_errors.append(f"{f.name}: {e}")
            continue
        kind = detect_artifact_type(obj)
        if kind is None:
            parse_errors.append(f"{f.name}: not a recognised lens-kit artifact")
            continue
        loaded.append((f, kind, obj))

    if not loaded:
        # A single explicit file that failed is a hard error; a directory
        # with only junk is too. Report the first cause for the CLI message.
        detail = parse_errors[0] if parse_errors else "no recognised artifacts"
        raise ArtifactError(f"could not load artifacts from {p}: {detail}")
    return loaded


# ── Record building ───────────────────────────────────────────────

def build_records(
    loaded: list[tuple[Path, str, dict]]
) -> tuple[list[dict], list[dict]]:
    """Flatten loaded artifacts into (records, summaries).

    ``records`` is the per-item review list (one chip card each).
    ``summaries`` is one header block per source artifact (accuracy/catch/FP,
    acceptable/fp/fn rate, mutants-missed, plus the variance envelope when
    the artifact carries one).
    """
    records: list[dict] = []
    summaries: list[dict] = []
    for src, kind, obj in loaded:
        stem = src.stem
        if kind == KIND_EVAL:
            summaries.append(_eval_summary(stem, obj))
            records.extend(_eval_records(stem, obj))
        elif kind == KIND_CALIBRATION:
            summaries.append(_calib_summary(stem, obj))
            records.extend(_calib_records(stem, obj))
        elif kind == KIND_MUTATION:
            summaries.append(_mutation_summary(stem, obj))
            records.extend(_mutation_records(stem, obj))
    return records, summaries


def _eval_summary(stem: str, obj: dict) -> dict:
    s = obj.get("summary") or {}
    return {
        "source": stem,
        "kind": KIND_EVAL,
        "checkpoint": obj.get("checkpoint"),
        "model": obj.get("model"),
        "profile": obj.get("profile"),
        "total_examples": obj.get("total_examples"),
        "errors": obj.get("errors"),
        "metrics": {
            "overall_accuracy": s.get("overall_accuracy"),
            "catch_rate": s.get("catch_rate"),
            "fp_rate": s.get("fp_rate"),
        },
        # An eval rerun file does not itself hold the envelope (that lands in a
        # sibling *-envelope.json). If the caller passed the envelope file it
        # classifies as eval too and carries "metrics" per-stat; surface it.
        "envelope": _envelope_block(obj),
    }


def _envelope_block(obj: dict) -> dict | None:
    """Extract a variance-envelope block if the artifact is an envelope file.

    Eval/calibration envelope files carry a top-level ``metrics`` mapping
    metric -> {mean, stddev, min, max, worst_of_n}. A per-run results file
    instead has ``summary`` (scalars), so this returns None for those.
    """
    metrics = obj.get("metrics")
    if isinstance(metrics, dict) and metrics and all(
        v is None or isinstance(v, dict) for v in metrics.values()
    ):
        return {
            "reruns": obj.get("reruns"),
            "fresh_cache": obj.get("fresh_cache"),
            "cache_caveat": obj.get("cache_caveat"),
            "metrics": metrics,
        }
    return None


def _eval_records(stem: str, obj: dict) -> list[dict]:
    out: list[dict] = []
    for ex in obj.get("per_example", []) or []:
        idx = ex.get("index")
        confusion = ex.get("lens_confusion")  # {lens: tp|tn|fp|fn|None} or None
        out.append({
            "id": f"eval:{stem}#{idx}",
            "kind": KIND_EVAL,
            "source": stem,
            "title": _eval_title(ex),
            "data": {
                "index": idx,
                "score": ex.get("score"),
                "domain": ex.get("domain"),
                "source_name": ex.get("source"),
                "source_id": ex.get("source_id"),
                "elapsed": ex.get("elapsed"),
                "lens_confusion": confusion,
                "mismatches": ex.get("mismatches", []),
            },
        })
    return out


def _eval_title(ex: dict) -> str:
    sid = ex.get("source_id") or ex.get("source") or ""
    base = f"#{ex.get('index')}"
    if sid:
        base += f"  {sid}"
    score = ex.get("score")
    if score is not None:
        base += f"  ({score:.0%})" if isinstance(score, (int, float)) else ""
    return base


def _calib_summary(stem: str, obj: dict) -> dict:
    s = obj.get("summary") or {}
    return {
        "source": stem,
        "kind": KIND_CALIBRATION,
        "model": obj.get("model"),
        "profile": obj.get("profile"),
        "battery_dir": obj.get("battery_dir"),
        "metrics": {
            "acceptable_rate": s.get("acceptable_rate"),
            "on_target_rate": s.get("on_target_rate"),
            "fp_rate": s.get("fp_rate"),
            "fn_rate": s.get("fn_rate"),
            "unknown_count": s.get("unknown_count"),
        },
        "envelope": _envelope_block(obj),
    }


def _calib_records(stem: str, obj: dict) -> list[dict]:
    out: list[dict] = []
    for row in obj.get("results", []) or []:
        fixture = row.get("fixture", "?")
        out.append({
            "id": f"calib:{stem}#{fixture}",
            "kind": KIND_CALIBRATION,
            "source": stem,
            "title": f"{fixture}  [{row.get('class')}]",
            "data": {
                "fixture": fixture,
                "class": row.get("class"),
                "verdict": row.get("verdict"),
                "acceptable": row.get("acceptable"),
                "on_target": row.get("on_target"),
                "halted": row.get("halted"),
                "lens_confusion": row.get("lens_confusion"),
                "mismatches": row.get("mismatches", []),
            },
        })
    return out


def _mutation_summary(stem: str, obj: dict) -> dict:
    return {
        "source": stem,
        "kind": KIND_MUTATION,
        "model": obj.get("model"),
        "profile": obj.get("profile"),
        "holdout": obj.get("holdout"),
        "seed": obj.get("seed"),
        "metrics": {
            "total": obj.get("total"),
            "missed": obj.get("missed"),
        },
        "envelope": None,
    }


def _mutation_records(stem: str, obj: dict) -> list[dict]:
    out: list[dict] = []
    for row in obj.get("results", []) or []:
        rid = row.get("id", "?")
        # NOTE: the current mutation writer (mutation.run_mutant) persists the
        # mutated body under "text"; artifacts from older kit versions carry
        # only the planted-defect marker. When "text" is absent the card shows
        # the marker + an honest "(mutant text not stored in results)" note
        # rather than inventing text.
        out.append({
            "id": f"mut:{stem}#{rid}",
            "kind": KIND_MUTATION,
            "source": stem,
            "title": f"{rid}  [{row.get('mutation_type')}]",
            "data": {
                "mutant_id": rid,
                "mutation_type": row.get("mutation_type"),
                "verdict": row.get("verdict"),
                "caught": row.get("caught"),
                "marker": row.get("marker"),
                "expected_one_of": row.get("expected_one_of", []),
                "text": row.get("text"),  # absent only in pre-C5 artifacts
                "error": row.get("error"),
            },
        })
    return out
