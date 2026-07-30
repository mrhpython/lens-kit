"""Mutation control: the "is my gate actually alive?" check.

WHY (the argument a customer reads in the README): a gate that PASSes everything
proves nothing — it might be catching real flaws, or it might be a rubber stamp.
The only way to know is to plant a flaw you KNOW is there and confirm the gate
fails it. This is the seeded-broken-variant pattern: take a real (clean) holdout
example, deterministically inject ONE known defect — a fabricated specific
statistic, a self-contradicting number, a stripped source attribution — and run
it through the gate. The gate MUST block every mutant. If any mutant sails
through, the gate is rubber-stamping on that defect class, and `lens-kit mutate`
exits non-zero naming the missed mutants. Run this after every compile or model
change, before you trust a green eval.

Mutant construction is PURE PYTHON: no LLM, deterministic by seed. Same
(holdout, seed) -> byte-identical mutants. The defect each mutation plants is
recorded (mutation_type + the expected catching verdict), so a miss is
attributable to a specific lens family, not a vague "gate seems weak".

Verdict mapping reuses calibration.map_verdict (SHIP / HOLD / NEEDS_SOURCES).
Each mutation type carries the set of verdicts that count as "caught".
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from .calibration import map_verdict
from .config import Profile, lm_context
from .gate import LensGate

MUTATION_SCHEMA_VERSION = 1

# Exit code shared with the calibration battery: "the gate failed to catch what
# it must". 5 (not 3 — 3 is the compile costing-gate stop). Mirrors
# calibration.EXIT_BATTERY_FAIL; kept as its own constant so the meaning reads
# locally.
EXIT_RUBBER_STAMP = 5

# ── Mutation types ───────────────────────────────────────────────
# Each type: a deterministic transform that injects ONE known defect into a
# clean example's text, plus the verdicts that count as "caught". A mutant is a
# MISS when its verdict is outside this caught-set (typically a silent SHIP).
#
#   fabricated_stat   -> append a specific, uncited statistic presented as fact.
#                        Caught = NEEDS_SOURCES or HOLD (it needs a citation or a
#                        block; a silent SHIP is the failure).
#   contradiction     -> append a claim numerically incompatible with one already
#                        in the text (or a planted pair). Caught = HOLD.
#   stripped_source   -> remove inline source attributions ("according to X",
#                        "[source]", "per Y") leaving the claims bare. Caught =
#                        NEEDS_SOURCES or HOLD.

MUTATION_TYPES = ("fabricated_stat", "contradiction", "stripped_source")

# verdicts that count as "the gate caught this mutant", per type.
CAUGHT_VERDICTS: dict[str, set[str]] = {
    "fabricated_stat": {"NEEDS_SOURCES", "HOLD"},
    "contradiction": {"HOLD"},
    "stripped_source": {"NEEDS_SOURCES", "HOLD"},
}

# Deterministic, non-LLM injected fragments. Phrased to be unambiguously the
# planted defect for that class, so a customer can read the mutant and confirm
# the flaw is present (it is also recorded in the mutant record's marker).
_FABRICATED_STAT_FRAGMENTS = (
    " Independent research confirms a 73.4% improvement across all accounts.",
    " Studies show 9 out of 10 teams double their output within 30 days.",
    " Our internal data proves a 4.8x return on every dollar spent.",
)
_CONTRADICTION_FRAGMENTS = (
    " This process is fully automated with zero human involvement; "
    "every item is also manually reviewed by our team before release.",
    " The service is completely free of charge; your card is billed "
    "$199 per month after the first day.",
    " Results appear within 24 hours; full onboarding takes 90 days "
    "before the first result is produced.",
)
# Source-attribution patterns to strip (case-insensitive, in priority order).
_SOURCE_PATTERNS = (
    "according to ", "as reported by ", "per ", "[source]", "[citation]",
    "sources say ", "studies show ", "research from ",
)
_STRIPPED_SOURCE_FALLBACK = (
    " The figures above are stated without any source or citation.")


def _strip_sources(text: str) -> tuple[str, bool]:
    """Remove inline source attributions. Returns (mutated_text, stripped?).

    Deterministic: lowercases for matching only; removes the attribution phrase
    plus the rest of that clause up to the next sentence boundary. If nothing
    matched, appends a marker line so the mutant still carries an unsourced-claim
    defect (the gate must still see SOMETHING to catch); that fallback path
    returns stripped=False so the recorded marker can describe what was
    actually done.
    """
    out = text
    changed = False
    low = out.lower()
    for pat in _SOURCE_PATTERNS:
        idx = low.find(pat)
        while idx != -1:
            # Drop from the attribution phrase to the next sentence end.
            end = len(out)
            for stop in (". ", ".\n", "! ", "? "):
                s = out.find(stop, idx)
                if s != -1:
                    end = min(end, s + 1)
            out = (out[:idx] + out[end:]).replace("  ", " ")
            changed = True
            low = out.lower()
            idx = low.find(pat)
    if not changed:
        out = out.rstrip() + _STRIPPED_SOURCE_FALLBACK
        return out, False
    return out, True


def mutate_text(text: str, mutation_type: str, rng: random.Random) -> tuple[str, str]:
    """Apply one mutation to text. Returns (mutated_text, marker).

    `marker` is a human-readable description of the planted defect, recorded in
    the mutant record (NEVER injected into the text). rng is seeded by the
    caller for determinism.
    """
    if mutation_type == "fabricated_stat":
        frag = rng.choice(_FABRICATED_STAT_FRAGMENTS)
        return text.rstrip() + frag, f"appended uncited statistic:{frag.strip()}"
    if mutation_type == "contradiction":
        frag = rng.choice(_CONTRADICTION_FRAGMENTS)
        return text.rstrip() + frag, f"appended self-contradicting claim:{frag.strip()}"
    if mutation_type == "stripped_source":
        mutated, stripped = _strip_sources(text)
        if stripped:
            return mutated, "stripped inline source attributions"
        return mutated, "appended unsourced-claim marker (no attributions to strip)"
    raise ValueError(f"Unknown mutation type: {mutation_type!r}")


# ── Mutant generation ────────────────────────────────────────────

def _load_examples(holdout_path: str | Path) -> list[dict]:
    doc = json.loads(Path(holdout_path).read_text())
    if isinstance(doc, dict):
        return doc.get("examples", [])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Holdout must be a JSON list or {{'examples': [...]}}: {holdout_path}")


def generate_mutants(holdout_path: str | Path, *, seed: int = 0,
                     per_type: int = 1) -> list[dict]:
    """Deterministically seed mutants from real holdout examples.

    For each mutation type, pick `per_type` source examples (preferring those
    with text long enough to mutate) and apply that mutation. Same
    (holdout, seed, per_type) -> byte-identical mutants.

    Each mutant record:
        id            stable "<mutation_type>-<NN>"
        mutation_type one of MUTATION_TYPES
        source_index  index into the holdout
        text          mutated text (the planted defect is in here)
        marker        human description of the defect (NOT in text)
        caught_verdicts  verdicts that count as catching this mutant
    """
    examples = _load_examples(holdout_path)
    usable = [(i, e) for i, e in enumerate(examples)
              if len(str(e.get("text", "")).strip()) >= 20]
    if not usable:
        raise ValueError(
            f"No usable examples (>=20 chars of text) in {holdout_path} — "
            f"cannot seed mutants from an empty/short holdout.")

    mutants: list[dict] = []
    for mtype in MUTATION_TYPES:
        rng = random.Random(f"{seed}:{mtype}")
        # Deterministic selection: shuffle a copy of indices, take per_type.
        pool = list(usable)
        rng.shuffle(pool)
        chosen = pool[:per_type] if per_type <= len(pool) else (
            pool + [rng.choice(usable) for _ in range(per_type - len(pool))])
        for n, (src_idx, ex) in enumerate(chosen, start=1):
            text = str(ex.get("text", ""))
            mutated, marker = mutate_text(text, mtype, rng)
            mutants.append({
                "id": f"{mtype}-{n:02d}",
                "mutation_type": mtype,
                "source_index": src_idx,
                "source_id": ex.get("source_id", ex.get("source", "")),
                "domain": ex.get("domain", "general"),
                "context": ex.get("context", ""),
                "text": mutated,
                "marker": marker,
                "caught_verdicts": sorted(CAUGHT_VERDICTS[mtype]),
            })
    return mutants


# ── Running mutants through the gate ─────────────────────────────

def make_gate(profile: Profile):
    """Bare LensGate for the mutation check. Separated so tests can monkeypatch."""
    return LensGate(profile=profile)


def run_mutant(mutant: dict, gate) -> dict:
    """Run one mutant through the gate; classify caught/missed.

    A crash is a MISS (a gate that errors on a mutant did not catch it). The
    verdict is mapped via calibration.map_verdict.
    """
    caught_set = set(mutant["caught_verdicts"])
    try:
        result = gate(text=mutant["text"], domain=mutant.get("domain", "general"),
                      context=mutant.get("context", ""))
        verdict = map_verdict(result)
        caught = verdict in caught_set
        error = None
    except Exception as e:  # an error is not a catch
        verdict, caught, error = "ERROR", False, repr(e)
    row = {
        "id": mutant["id"],
        "mutation_type": mutant["mutation_type"],
        "verdict": verdict,
        "caught": caught,
        "expected_one_of": sorted(caught_set),
        "marker": mutant["marker"],
        # The mutated body itself, so the review surface (lens-kit review) can
        # show the human the actual text the gate judged, not just the marker.
        # Backward-compatible artifact addition: consumers ignore unknown keys.
        "text": mutant["text"],
    }
    if error:
        row["error"] = error
    return row


def run_mutation_control(holdout_path: str | Path, profile: Profile, *,
                         seed: int = 0, per_type: int = 1,
                         output_dir: str | Path | None = None,
                         command_str: str = "lens-kit mutate",
                         gate=None) -> int:
    """Seed mutants, run them through the gate, fail loud on any miss.

    Returns a CLI exit code:
        0 — every mutant caught (the gate is alive on every defect class)
        5 — one or more mutants slipped through (rubber-stamping); the missed
            mutants are named and a results file is written.

    `gate` may be supplied pre-built (tests); otherwise one is built from the
    profile and run under lm_context.
    """
    mutants = generate_mutants(holdout_path, seed=seed, per_type=per_type)
    print(f"Mutation control: {len(mutants)} seeded mutants "
          f"({Path(holdout_path).name}, seed={seed})")
    print(f"Profile: {profile.name}  model: {profile.llm.model}")

    if gate is not None:
        rows = [run_mutant(m, gate) for m in mutants]
    else:
        built = make_gate(profile)
        with lm_context(profile):
            rows = [run_mutant(m, built) for m in mutants]

    missed = [r for r in rows if not r["caught"]]
    print(f"\n{'mutant':<20} {'type':<18} {'verdict':<14} caught")
    for r in rows:
        print(f"{r['id']:<20} {r['mutation_type']:<18} {r['verdict']:<14} "
              f"{'Y' if r['caught'] else 'X'}")

    doc = {
        "mutation_schema_version": MUTATION_SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": command_str,
        "profile": profile.name,
        "model": profile.llm.model,
        "holdout": str(holdout_path),
        "seed": seed,
        "per_type": per_type,
        "total": len(rows),
        "missed": len(missed),
        "results": rows,
    }
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mutation-{time.strftime('%Y%m%d-%H%M%S')}.json"
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults saved: {out_path}")

    from . import ledger
    run_id = ledger.new_run_id("mutate")
    ledger.append_run(
        run_id=run_id, command=command_str, checkpoint_sha=None,
        data_file=Path(holdout_path).name,
        metrics=f"mutants={len(rows)}; missed={len(missed)}",
        status="MUTATE" if not missed else "MUTATE (rubber-stamp)")
    print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")

    if missed:
        names = ", ".join(f"{r['id']} (verdict={r['verdict']}, "
                          f"expected one of {r['expected_one_of']})" for r in missed)
        print(f"\nRUBBER-STAMP WARNING: {len(missed)} of {len(rows)} planted "
              f"mutants slipped through the gate.\n"
              f"  Missed: {names}\n"
              f"A gate that cannot fail a planted flaw proves nothing. "
              f"Investigate the lens for each missed defect class before "
              f"trusting this gate's PASS verdicts.")
        return EXIT_RUBBER_STAMP
    print(f"\nAll {len(rows)} planted mutants caught — the gate is alive on "
          f"every defect class.")
    return 0
