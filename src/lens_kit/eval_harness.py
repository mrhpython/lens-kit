"""Holdout eval harness: per-lens confusion, mismatch strings, variance envelope.

Receipt conventions mirror the production harnesses (schema-v2 style):
  - per-example mismatch strings: "<lens>: FN (expected FAIL, got PASS)"
    (positive class = FAIL = defect present; FN is the forensics row)
  - per-lens confusion counts tp/tn/fp/fn, FN-forensics buildable by grep
  - results JSON is self-describing and written to VERSIONED filenames —
    never overwritten
  - --reruns N banks a variance envelope: per-metric min/max/mean/pstdev +
    direction-aware worst-of-N floor (fp_rate floor = MAX observed;
    accuracy/catch floor = MIN observed)

Cache caveat (encoded as a printed warning + --fresh-cache): dspy's disk
cache replays identical responses for identical prompts, which can make
"reruns" byte-identical — a variance battery we ran was voided by exactly
this. --fresh-cache points the dspy cache at a fresh temp dir per rerun
(and disables the in-process memory cache) so variance runs are real.
"""
import datetime
import json
import statistics
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from .checkpoint import load_checkpoint, sha256_file
from .config import Profile, lm_context
from .gate import LensGate, validate_lens_labels

EVAL_SCHEMA_VERSION = 2

# Variance-envelope metrics and which direction "worse" points.
# higher_is_better -> worst-of-N floor = MIN; lower_is_better -> floor = MAX.
METRIC_DIRECTIONS = {
    "overall_accuracy": "higher_is_better",
    "catch_rate": "higher_is_better",
    "fp_rate": "lower_is_better",
}

CACHE_WARNING = (
    "WARNING: dspy's disk cache replays identical responses for identical\n"
    "prompts — reruns can be byte-identical and the 'variance' you measure is\n"
    "zero by construction (this voided a real variance battery). Pass\n"
    "--fresh-cache to point the cache at a fresh temp dir per rerun."
)


# ── Data ─────────────────────────────────────────────────────────

def load_holdout(path: str | Path) -> tuple[list[dict], dict]:
    """Accepts {'_metadata':..., 'examples':[...]} or a plain JSON list.

    Lens labels are validated at load: a per_lens key outside the canonical
    lens set is a HARD ERROR (fail closed) — a typo'd label can never match
    gate output and would otherwise silently score 0.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    if isinstance(doc, dict):
        examples, meta = doc.get("examples", []), doc.get("_metadata", {})
    elif isinstance(doc, list):
        examples, meta = doc, {}
    else:
        raise ValueError(f"Holdout must be a JSON list or {{'examples': [...]}}: {path}")
    for i, item in enumerate(examples):
        validate_lens_labels(item.get("per_lens", {}),
                             where=f"{path.name} example {i}")
    return examples, meta


# ── Assessment ───────────────────────────────────────────────────

def run_assessment(examples: list[dict], gate, progress: bool = True) -> dict:
    """Run the gate over every example; collect per-lens confusion +
    per-example mismatch strings. Positive class = FAIL (violation)."""
    results = {
        "total": len(examples),
        "per_lens": defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}),
        "overall_correct": 0,
        "overall_total": 0,
        "per_example": [],
        "errors": 0,
    }

    for i, item in enumerate(examples):
        text = item.get("text", "")
        domain = item.get("domain", "general")
        expected = item.get("per_lens", {})
        source = item.get("source", "unknown")
        source_id = item.get("source_id", "")

        try:
            start = time.time()
            result = gate(text=text, domain=domain)
            elapsed = time.time() - start
            actual = result.per_lens

            correct = total = 0
            mismatches = []
            lens_confusion = {}
            for lens, expected_pass in expected.items():
                if lens not in actual:
                    # Not evaluated (e.g. relevance without context, post-HALT)
                    lens_confusion[lens] = None
                    continue
                actual_pass = actual[lens]
                total += 1
                if actual_pass == expected_pass:
                    correct += 1
                    if expected_pass:
                        results["per_lens"][lens]["tn"] += 1
                        lens_confusion[lens] = "tn"
                    else:
                        results["per_lens"][lens]["tp"] += 1
                        lens_confusion[lens] = "tp"
                elif expected_pass:
                    results["per_lens"][lens]["fp"] += 1
                    lens_confusion[lens] = "fp"
                    mismatches.append(f"{lens}: FP (expected PASS, got FAIL)")
                else:
                    results["per_lens"][lens]["fn"] += 1
                    lens_confusion[lens] = "fn"
                    mismatches.append(f"{lens}: FN (expected FAIL, got PASS)")

            results["overall_correct"] += correct
            results["overall_total"] += total
            score = correct / max(total, 1)
            results["per_example"].append({
                "index": i, "source": source, "source_id": source_id,
                "domain": domain, "score": score,
                "mismatches": mismatches,
                "lens_confusion": lens_confusion,
                "elapsed": round(elapsed, 2),
            })
            if progress:
                print("." if score == 1.0 else "X", end="", flush=True)
        except Exception as e:
            results["errors"] += 1
            results["per_example"].append({
                "index": i, "source": source, "source_id": source_id,
                "domain": domain, "score": 0.0,
                "mismatches": [f"ERROR: {e}"],
                "lens_confusion": None,
                "elapsed": 0,
            })
            if progress:
                print("E", end="", flush=True)

    if progress:
        print()
    return results


def summarize(results: dict) -> dict:
    """Overall accuracy / catch rate / FP rate from the per-lens confusion."""
    tp = sum(s["tp"] for s in results["per_lens"].values())
    tn = sum(s["tn"] for s in results["per_lens"].values())
    fp = sum(s["fp"] for s in results["per_lens"].values())
    fn = sum(s["fn"] for s in results["per_lens"].values())
    return {
        "overall_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "catch_rate": tp / max(tp + fn, 1),
        "fp_rate": fp / max(fp + tn, 1),
    }


def print_report(results: dict, summary: dict, checkpoint_name: str) -> None:
    print(f"\n{'Lens':<16} {'Acc':>6} {'Catch':>7} {'FP Rate':>8} "
          f"{'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print("-" * 62)
    for lens in sorted(results["per_lens"]):
        s = results["per_lens"][lens]
        n = sum(s.values())
        acc = (s["tp"] + s["tn"]) / max(n, 1)
        catch = s["tp"] / max(s["tp"] + s["fn"], 1) if (s["tp"] + s["fn"]) else None
        fpr = s["fp"] / max(s["fp"] + s["tn"], 1) if (s["fp"] + s["tn"]) else None
        print(f"{lens:<16} {acc:>6.1%} "
              f"{(f'{catch:.1%}' if catch is not None else 'N/A'):>7} "
              f"{(f'{fpr:.1%}' if fpr is not None else 'N/A'):>8} "
              f"{s['tp']:>4} {s['tn']:>4} {s['fp']:>4} {s['fn']:>4}")
    print("-" * 62)
    print(f"{'OVERALL':<16} {summary['overall_accuracy']:>6.1%} "
          f"{summary['catch_rate']:>7.1%} {summary['fp_rate']:>8.1%}   "
          f"[{checkpoint_name}] errors={results['errors']}")


# ── Variance envelope (direction-aware worst-of-N) ───────────────

def build_envelope(summaries: list[dict], directions: dict | None = None) -> dict:
    """Per-metric min/max/mean/pstdev + worst-of-N floor.

    Floor direction follows ``directions`` (default: this module's eval
    METRIC_DIRECTIONS): a higher_is_better metric floors at the MIN observed,
    a lower_is_better metric (e.g. fp_rate) floors at the MAX observed. The
    calibration battery passes its own metric/direction map to reuse this.
    """
    directions = directions if directions is not None else METRIC_DIRECTIONS
    metrics = {}
    for m, direction in directions.items():
        vals = [s[m] for s in summaries if s.get(m) is not None]
        if not vals:
            metrics[m] = None
            continue
        metrics[m] = {
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.mean(vals),
            "stddev": statistics.pstdev(vals),
            "direction": direction,
            "worst_of_n": min(vals) if direction == "higher_is_better" else max(vals),
        }
    return metrics


# ── Versioned, never-overwriting result paths ────────────────────

def next_free_run_path(dirpath: Path, date: str, prefix: str = "eval") -> Path:
    """<prefix>-<date>-runK.json with the lowest unused K — never overwrites."""
    k = 1
    while (dirpath / f"{prefix}-{date}-run{k}.json").exists():
        k += 1
    return dirpath / f"{prefix}-{date}-run{k}.json"


def next_free_envelope_path(dirpath: Path, date: str, prefix: str = "eval") -> Path:
    """<prefix>-<date>-envelope.json, then -envelope-2.json, ... — never overwrites."""
    p = dirpath / f"{prefix}-{date}-envelope.json"
    k = 2
    while p.exists():
        p = dirpath / f"{prefix}-{date}-envelope-{k}.json"
        k += 1
    return p


# ── Gate construction + cache control ────────────────────────────

def make_gate(profile: Profile, checkpoint: str | Path | None):
    """Bare LensGate; checkpoint loaded via the remapping loader when given."""
    gate = LensGate(profile=profile)
    if checkpoint:
        load_checkpoint(gate, checkpoint)
    return gate


def configure_fresh_cache() -> str:
    """Point the dspy cache at a brand-new temp dir; kill the memory cache.

    Returns the temp dir path. Per-rerun fresh dirs are what make a
    variance battery real instead of a cache replay. Process-global —
    callers must wrap in preserve_dspy_cache() to avoid clobbering the
    embedding application's cache config.
    """
    import dspy

    cache_dir = tempfile.mkdtemp(prefix="lens-kit-fresh-cache-")
    dspy.configure_cache(
        enable_disk_cache=True,
        disk_cache_dir=cache_dir,
        enable_memory_cache=False,
    )
    return cache_dir


@contextmanager
def preserve_dspy_cache():
    """Restore the process-global dspy cache object on exit.

    dspy.configure_cache() replaces the global ``dspy.cache``; library
    consumers importing run_eval_command must get their prior cache config
    back, including on exception.
    """
    import dspy

    prior = getattr(dspy, "cache", None)
    try:
        yield
    finally:
        dspy.cache = prior


# ── Command entry ────────────────────────────────────────────────

def run_eval_command(holdout_path: str | Path, profile: Profile, *,
                     checkpoint: str | Path | None = None,
                     reruns: int = 1, fresh_cache: bool = False,
                     output_dir: str | Path = ".",
                     command_str: str = "lens-kit eval") -> int:
    """Full eval run (1..N reruns + envelope). Returns CLI exit code."""
    holdout_path = Path(holdout_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, meta = load_holdout(holdout_path)
    if not examples:
        print(f"ERROR: no examples in {holdout_path}")
        return 2
    checkpoint_name = Path(checkpoint).name if checkpoint else "baseline (uncompiled)"
    checkpoint_sha = sha256_file(checkpoint) if checkpoint else None
    print(f"Holdout: {len(examples)} examples ({holdout_path.name}; "
          f"created {meta.get('created', 'unknown')})")
    print(f"Mode: {checkpoint_name}")

    if reruns > 1 and not fresh_cache:
        print("\n" + CACHE_WARNING + "\n")

    date = datetime.date.today().isoformat()
    run_id = None
    summaries, run_files = [], []
    total_errors = 0

    with preserve_dspy_cache():
        for k in range(reruns):
            if fresh_cache:
                cache_dir = configure_fresh_cache()
                print(f"[rerun {k + 1}/{reruns}] fresh dspy cache: {cache_dir}")
            gate = make_gate(profile, checkpoint)
            with lm_context(profile):
                results = run_assessment(examples, gate)
            summary = summarize(results)
            total_errors += results["errors"]
            print_report(results, summary, checkpoint_name)

            doc = {
                "eval_schema_version": EVAL_SCHEMA_VERSION,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": checkpoint_sha,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "command": command_str,
                "profile": profile.name,
                "model": profile.llm.model,
                "data_file": str(holdout_path),
                "data_sha256": sha256_file(holdout_path),
                "run_index": k + 1,
                "reruns": reruns,
                "fresh_cache": fresh_cache,
                "summary": summary,
                "total_examples": results["total"],
                "errors": results["errors"],
                "per_lens": {lens: dict(s) for lens, s in results["per_lens"].items()},
                "per_example": results["per_example"],
            }
            out_path = next_free_run_path(output_dir, date)
            out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            print(f"Results saved: {out_path}")
            summaries.append(summary)
            run_files.append(out_path.name)

    metrics_str = "; ".join(f"{m}={summaries[-1][m]:.3f}" for m in METRIC_DIRECTIONS)
    if reruns > 1:
        env_path = next_free_envelope_path(output_dir, date)
        envelope = {
            "eval_schema_version": EVAL_SCHEMA_VERSION,
            "date": date,
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha,
            "data_file": str(holdout_path),
            "data_sha256": sha256_file(holdout_path),
            "reruns": reruns,
            "fresh_cache": fresh_cache,
            "cache_caveat": None if fresh_cache else (
                "reruns shared the default dspy cache — variance may be "
                "understated by cache replay; re-run with --fresh-cache"),
            "run_files": run_files,
            "run_summaries": summaries,
            "metrics": build_envelope(summaries),
        }
        env_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        print(f"\nEnvelope saved: {env_path}")
        for m, st in envelope["metrics"].items():
            if st is None:
                continue
            print(f"  {m:<18} mean {st['mean']:.4f}  stddev {st['stddev']:.4f}  "
                  f"range [{st['min']:.4f}, {st['max']:.4f}]  "
                  f"worst-of-N {st['worst_of_n']:.4f}")
        floors = envelope["metrics"]
        metrics_str = "; ".join(
            f"{m}_floor={floors[m]['worst_of_n']:.3f}" for m in METRIC_DIRECTIONS
            if floors.get(m))

    from . import ledger
    run_id = ledger.new_run_id("eval")
    ledger.append_run(run_id=run_id, command=command_str,
                      checkpoint_sha=checkpoint_sha, data_file=holdout_path.name,
                      metrics=metrics_str,
                      status="EVAL" if total_errors == 0 else f"EVAL ({total_errors} errors)")
    print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")
    return 0
