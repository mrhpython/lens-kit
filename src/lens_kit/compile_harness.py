"""GEPA compile harness for the lens-kit gate.

Flow (run_compile):
    1. load training data (plain JSON list of {text, per_lens, ...};
       skip text < 20 chars or empty per_lens), deterministic shuffle,
       train/val split
    2. COSTING GATE — projected rollouts via GEPA's own budget accounting,
       live pace probe, printed projection; HARD STOP (exit 3) when the
       projection exceeds the approval threshold without --approve-cost
    3. GEPA compile over LensGateWrapper under lm_context(profile), with a
       pace monitor that aborts (exit 4) + writes a structured kill record
       if measured pace blows the approved projection
    4. post-compile val score, checkpoint save (sha256 sibling + FULL
       provenance sidecar, dataset block from the training file), RUNS.md
       ledger row

Provider-agnostic: task LM and reflection LM both come from the profile.
The reflection LM defaults to the profile's task LM at temperature 1.0;
override with a `compile.reflection` block when you want a stronger model.
"""
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import dspy

from . import ledger
from .checkpoint import save_checkpoint
from .config import Profile, lm_context, make_lm
from .costing import (CostProjection, PaceExceeded, PaceMonitor, PaceStopper,
                      gepa_projected_metric_calls, measure_pace,
                      measure_pace_fresh_cache, monitored_metric)
from .gate import CANONICAL_LENSES, LensGate, validate_lens_labels

# The 7 lenses with gold training labels. Excludes relevance/judgment/
# cross_check/fixer, which have no training signal and waste reflection cycles.
SCORED_LENSES = (
    "truth", "causality", "contradiction", "extrapolation",
    "rights", "structure", "consistency",
)

EXIT_COST_STOP = 3
EXIT_PACE_KILL = 4


# ── Data loading ─────────────────────────────────────────────────

def load_training_examples(path: str | Path) -> list[dspy.Example]:
    """Plain JSON list of {text, per_lens, domain?, source?} -> dspy.Examples.

    Skips items with text < 20 chars or empty per_lens. Non-canonical
    per_lens keys are a load-time hard error (validate_lens_labels) — a
    typo'd label would silently score 0 across the whole run. Deterministic
    shuffle (seed 42) so train/val splits are reproducible.
    Also accepts the holdout wrapper shape {"examples": [...]}.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("examples", [])
    if not isinstance(data, list):
        raise ValueError(f"Training data must be a JSON list (or {{'examples': [...]}}): {path}")

    examples = []
    for i, item in enumerate(data):
        text = item.get("text", "")
        if len(text) < 20:
            continue
        per_lens = item.get("per_lens", {})
        if not per_lens:
            continue
        validate_lens_labels(per_lens, where=f"{path.name} item {i}")
        examples.append(dspy.Example(
            text=text,
            domain=item.get("domain", "general"),
            expected_per_lens=json.dumps(per_lens),
        ).with_inputs("text", "domain"))

    random.Random(42).shuffle(examples)
    return examples


def split_examples(examples: list, val_split: float) -> tuple[list, list]:
    split_idx = max(3, int(len(examples) * (1 - val_split)))
    return examples[:split_idx], examples[split_idx:]


# ── Metric (F2, recall-weighted) — ported from the production optimizer ──

def validate_weight_map(weights, *, where: str = "") -> dict | None:
    """Validate an optional per-lens severity weight map -> clean dict or None.

    None -> None (the default, byte-identical-behavior path). Otherwise every
    key must be a canonical lens and every value a non-negative number; an
    unknown lens key or a negative/non-numeric weight is a ValueError (fail
    closed — a typo'd or negative weight would silently mis-score). A lens
    absent from the map weighs 1.0 (unweighted) — the map only RE-weights the
    lenses it names. Partial maps are the intended use ("just upweight truth").
    """
    if weights is None:
        return None
    if not isinstance(weights, dict):
        raise ValueError(f"weight map must be a mapping of lens->weight{_loc(where)}")
    unknown = [k for k in weights if k not in CANONICAL_LENSES]
    if unknown:
        raise ValueError(
            f"Unknown lens key(s) {unknown} in weight map{_loc(where)} — "
            f"canonical lenses: {', '.join(CANONICAL_LENSES)}.")
    clean = {}
    for k, v in weights.items():
        try:
            w = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"weight for '{k}' is not a number ({v!r}){_loc(where)}")
        if w < 0:
            raise ValueError(f"weight for '{k}' is negative ({w}){_loc(where)} — "
                             f"severity weights must be >= 0.")
        clean[k] = w
    return clean


def _loc(where: str) -> str:
    return f" in {where}" if where else ""


def lens_gate_metric(example, pred, trace=None, pred_name=None, pred_trace=None,
                     weights=None):
    """Score gate output with F2 (recall 2x precision; positive = FAIL).

    Blended: F2 on examples with planted violations; TN-rate on all-PASS
    examples (otherwise clean examples score 0 and GEPA learns to flag
    everything). GEPA-compatible (returns ScoreWithFeedback when pred_name
    is given); also a plain 3-arg metric for dspy.Evaluate.

    ``weights`` (optional, profile-driven): a per-lens severity weight map that
    SCALES the existing per-lens contributions — a high-severity lens (e.g.
    truth, contradiction) can count more than a low one (e.g. structure). It is
    a weighting of the SAME F2/TN-rate computation, not a new metric: each lens
    contributes its weight (default 1.0) to the tp/fp/fn/tn and correct/total
    tallies instead of a flat 1. ``weights=None`` (the default) keeps integer
    arithmetic untouched and is byte-identical to the pre-change behavior.
    """
    expected = json.loads(example.expected_per_lens)
    if not (hasattr(pred, "per_lens") and isinstance(pred.per_lens, dict)):
        return 0.0
    actual = pred.per_lens

    # Default path: flat weight 1 per lens (integer tallies). The weighted path
    # scales each lens's contribution by its severity weight. Default == None is
    # kept on the integer path so the regression is byte-identical.
    wmap = validate_weight_map(weights) if weights is not None else None

    tp = fp = fn = tn = 0
    correct = total = 0
    mismatches = []
    for lens, expected_pass in expected.items():
        if lens not in actual:
            continue
        w = 1 if wmap is None else wmap.get(lens, 1.0)
        total += w
        actual_pass = actual[lens]
        if actual_pass == expected_pass:
            correct += w
        if not expected_pass and not actual_pass:
            tp += w
        elif expected_pass and actual_pass:
            tn += w
        elif expected_pass and not actual_pass:
            fp += w
        else:
            fn += w
        if actual_pass != expected_pass:
            mismatches.append(f"{lens}: expected {'PASS' if expected_pass else 'FAIL'}, "
                              f"got {'PASS' if actual_pass else 'FAIL'}")

    beta_sq = 4  # β=2
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f2 = ((1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
          if precision + recall > 0 else 0.0)
    accuracy = correct / max(total, 1)

    has_violations = (tp + fn) > 0
    score = f2 if has_violations else tn / max(tn + fp, 1)

    if pred_name is not None:
        if mismatches:
            feedback = (f"F2={f2:.2f} (prec={precision:.2f}, recall={recall:.2f}), "
                        f"acc={accuracy:.2f}. TP={tp} FP={fp} FN={fn} TN={tn}. "
                        f"Mismatches: {'; '.join(mismatches)}")
            relevant = [m for m in mismatches if pred_name.lower() in m.lower()]
            if relevant:
                feedback = (f"Predictor '{pred_name}' F2={f2:.2f}. "
                            f"Issues: {'; '.join(relevant)}")
        elif has_violations:
            feedback = (f"Correct: all planted violations caught. F2={f2:.2f} "
                        f"(prec={precision:.2f}, recall={recall:.2f}), "
                        f"acc={accuracy:.2f}.")
        else:
            # Clean example scored via TN-rate — an "F2=0.00" here would be
            # an incoherent signal for the reflection LM.
            feedback = (f"Correct: clean content passed all lenses "
                        f"(TN={tn}/{total}, acc={accuracy:.2f}). "
                        f"No violations expected, none flagged.")
        from dspy.teleprompt.gepa.gepa import ScoreWithFeedback
        return ScoreWithFeedback(score=score, feedback=feedback)
    return score


def _weighted_metric(weights):
    """Bind a per-lens weight map to lens_gate_metric -> a GEPA-shaped metric.

    ``weights is None`` -> the bare lens_gate_metric (byte-identical default;
    no wrapper at all, so nothing about the unweighted path changes). Otherwise
    the map is validated eagerly (CANONICAL keys, >= 0; raises ValueError before
    any billed rollout) and a closure forwards every GEPA argument plus the
    bound weights.
    """
    if weights is None:
        return lens_gate_metric
    validated = validate_weight_map(weights, where="compile.metric_weights")

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        return lens_gate_metric(example, pred, trace, pred_name, pred_trace,
                                weights=validated)
    return metric


# ── Wrapper module ───────────────────────────────────────────────
# GEPA needs a module returning dspy.Prediction, not LensResult.

class LensGateWrapper(dspy.Module):
    """Wraps LensGate so GEPA/Evaluate see a dspy.Prediction."""

    def __init__(self, profile: Profile | None = None, auto_fix: bool = True,
                 fast_mode: bool = False):
        super().__init__()
        self.gate = LensGate(profile=profile, auto_fix=auto_fix, fast_mode=fast_mode)

    def forward(self, text: str, domain: str = "general") -> dspy.Prediction:
        result = self.gate(text=text, domain=domain)
        return dspy.Prediction(
            passed=result.passed,
            per_lens=result.per_lens,
            violations=[{"lens": v.lens, "severity": v.severity, "issue": v.issue}
                        for v in result.violations],
            consciousness_flags=result.consciousness_flags,
            halted=result.halted,
        )


def scored_lens_selector(state, trajectories, scores, candidate_idx, candidate):
    """Restrict GEPA mutation to the 7 lenses that have training labels."""
    return [p for p in candidate.keys() if any(lens in p for lens in SCORED_LENSES)]


# ── Compile run ──────────────────────────────────────────────────

def _reflection_lm(profile: Profile):
    cfg = profile.compile.reflection
    if cfg is None:
        cfg = replace(profile.llm, temperature=1.0)
    return make_lm(cfg)


def _run_gepa(gate, train_set, val_set, *, metric, reflection_lm, threads,
              auto, max_evals, patience, monitor):
    """Build and run the GEPA optimizer. Separated for testability."""
    from dspy.teleprompt import GEPA
    from gepa.utils.stop_condition import NoImprovementStopper, SignalStopper

    stop_callbacks = [
        NoImprovementStopper(max_iterations_without_improvement=patience),
        PaceStopper(monitor),
    ]
    signal_stopper = SignalStopper()
    stop_callbacks.append(signal_stopper)

    gepa_kwargs = {
        "metric": metric,
        "num_threads": threads,
        "add_format_failure_as_feedback": True,
        "skip_perfect_score": True,
        "reflection_lm": reflection_lm,
        "component_selector": scored_lens_selector,
        "use_merge": False,
        "track_stats": True,
        "track_best_outputs": True,
        "gepa_kwargs": {"stop_callbacks": stop_callbacks},
    }
    if max_evals:
        gepa_kwargs["max_full_evals"] = max_evals
    else:
        gepa_kwargs["auto"] = auto

    optimizer = GEPA(**gepa_kwargs)
    try:
        return optimizer.compile(gate, trainset=train_set, valset=val_set or None)
    finally:
        signal_stopper.cleanup()


def _val_score(program, val_set, metric, threads) -> float | None:
    if not val_set:
        return None
    evaluator = dspy.Evaluate(devset=val_set, metric=metric,
                              num_threads=threads, display_progress=True)
    result = evaluator(program)
    score = getattr(result, "score", None)
    if score is None:
        try:
            score = float(result)
        except (TypeError, ValueError):
            return None
    return float(score)


def run_compile(training_path: str | Path, profile: Profile, output_path: str | Path,
                *, auto: str | None = None, max_evals: int | None = None,
                threads: int = 1, val_split: float = 0.2, patience: int = 10,
                approve_cost: bool = False, fresh_cache: bool = False,
                warm_probe_cache: bool = False,
                command_str: str = "lens-kit compile") -> int:
    """Full compile run. Returns a CLI exit code (0/2/3/4).

    Cache control (see costing.measure_pace_fresh_cache for the why):
      - The pace probe is measured against an ISOLATED fresh dspy cache by
        DEFAULT, so a warm cache from a prior run can't replay sub-second
        pace and arm a false pace kill. Pass ``warm_probe_cache=True`` to
        opt out (measure against the live cache — only sensible when you
        deliberately want to price cached pace).
      - ``fresh_cache=True`` additionally points the MAIN GEPA compile run at
        a fresh cache (mirrors ``eval --fresh-cache``): no rollout is served
        from a stale cache, so the run is fully billed and the pace monitor
        sees real pace throughout.
    """
    output_path = Path(output_path)
    cc = profile.compile

    # Validate the optional per-lens severity weight map BEFORE any work — a bad
    # map (unknown lens / negative weight) must fail fast (exit 2), never after a
    # billed rollout. None (every older profile) -> unweighted, unchanged.
    try:
        validate_weight_map(cc.metric_weights, where="compile.metric_weights")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    examples = load_training_examples(training_path)
    print(f"Training examples: {len(examples)} (after <20-char / empty-per_lens skip)")
    if len(examples) < 5:
        print("ERROR: need at least 5 usable examples to compile.")
        return 2
    train_set, val_set = split_examples(examples, val_split)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    if auto is None and max_evals is None:
        auto = "light"
    budget_str = f"max_evals={max_evals}" if max_evals else f"auto={auto}"

    gate = LensGateWrapper(profile=profile)
    num_preds = len(gate.predictors())

    gepa_calls = gepa_projected_metric_calls(
        num_preds=num_preds, train_size=len(train_set), val_size=len(val_set),
        auto=auto, max_evals=max_evals)
    probe_n = max(1, cc.probe_examples)

    run_id = ledger.new_run_id("compile")
    data_name = Path(training_path).name

    from .eval_harness import configure_fresh_cache, preserve_dspy_cache

    with lm_context(profile), preserve_dspy_cache():
        # ── Costing gate (the hard stop lives BEFORE any compile call) ──
        # The probe runs against an isolated fresh cache by default: a warm
        # cache replays sub-second pace and arms a FALSE pace kill (the
        # warm-cache incident). --warm-probe-cache opts out.
        probe_cache_note = ("live cache" if warm_probe_cache
                            else "isolated fresh cache")
        print(f"\nProbing pace on {probe_n} examples (live calls, "
              f"{probe_cache_note})...")
        # Reference the module-level measure_pace at call time (a single
        # monkeypatch point) for BOTH the warm and the fresh-probe path.
        if warm_probe_cache:
            sec_per_rollout = measure_pace(gate, train_set, probe_n=probe_n)
        else:
            sec_per_rollout = measure_pace_fresh_cache(
                gate, train_set, probe_n=probe_n, _measure=measure_pace)
        projection = CostProjection(
            projected_rollouts=gepa_calls + probe_n + len(val_set),
            budget=budget_str,
            train_size=len(train_set), val_size=len(val_set),
            seconds_per_rollout=sec_per_rollout, threads=threads,
            cost_per_rollout_usd=cc.cost_per_rollout_usd,
            threshold_hours=cc.approval_threshold_hours,
            probe_rollouts=probe_n,
        )
        print("\n" + projection.format())

        if projection.requires_approval and not approve_cost:
            print(f"\nHARD STOP: projected {projection.projected_hours:.2f}h exceeds the "
                  f"{cc.approval_threshold_hours:.2f}h approval threshold.\n"
                  f"Re-run with --approve-cost to accept this projection "
                  f"(or lower the budget / raise compile.approval_threshold_hours).")
            ledger.append_run(run_id=run_id, command=command_str, checkpoint_sha=None,
                              data_file=data_name,
                              metrics=f"projected {projection.projected_rollouts} rollouts / "
                                      f"{projection.projected_hours:.2f}h — not approved",
                              status="STOPPED_COST")
            return EXIT_COST_STOP

        # The probe restored the live cache on exit; if the caller asked for a
        # fresh main run, point the compile at a brand-new cache now (mirrors
        # eval --fresh-cache). Restored by the enclosing preserve_dspy_cache().
        if fresh_cache:
            main_cache_dir = configure_fresh_cache()
            print(f"Fresh dspy cache for the compile run: {main_cache_dir}")

        monitor = PaceMonitor(
            projected_rollouts=projection.projected_rollouts,
            approved_wall_seconds_per_rollout=projection.wall_seconds_per_rollout,
            kill_factor=cc.pace_kill_factor,
            min_calls=cc.pace_min_calls,
        )
        # Bind the optional per-lens severity weights from the profile. None
        # (every older profile) -> the unweighted metric, unchanged. Validated
        # eagerly here so a bad weight map fails before any billed rollout.
        scoring_metric = _weighted_metric(cc.metric_weights)
        metric = monitored_metric(scoring_metric, monitor)
        reflection_lm = _reflection_lm(profile)

        print(f"\nCompiling with GEPA ({budget_str}, threads={threads}, "
              f"patience={patience}, pace kill at "
              f"{cc.pace_kill_factor}x approved pace)...")
        monitor.start()
        try:
            compiled = _run_gepa(gate, train_set, val_set, metric=metric,
                                 reflection_lm=reflection_lm, threads=threads,
                                 auto=auto, max_evals=max_evals, patience=patience,
                                 monitor=monitor)
            if monitor.killed:  # raise swallowed by a parallel executor
                raise PaceExceeded(monitor.kill_record)
        except PaceExceeded as e:
            kill_path = Path(str(output_path) + ".kill.json")
            kill_path.parent.mkdir(parents=True, exist_ok=True)
            kill_path.write_text(json.dumps(e.record, indent=2) + "\n", encoding="utf-8")
            print(f"\nPACE KILL: {e.record['reason']}")
            print(f"  {e.record['percent_done']}% done, "
                  f"{e.record['elapsed_seconds']}s elapsed, "
                  f"projected remaining {e.record['projected_remaining_seconds']}s")
            print(f"  Kill record: {kill_path}")
            ledger.append_run(run_id=run_id, command=command_str, checkpoint_sha=None,
                              data_file=data_name,
                              metrics=f"killed at {e.record['percent_done']}% "
                                      f"({e.record['rollouts_done']} rollouts)",
                              status="KILLED")
            return EXIT_PACE_KILL

        # ── Past this point `compiled` is the artifact a BILLED run paid
        # for. Nothing below may lose it silently: a val-score failure
        # logs + continues to the save; a save failure attempts an
        # emergency fallback save; a ledger row is written whatever happens.
        print("\nEvaluating compiled gate on val split...")
        score = None
        try:
            score = _val_score(compiled, val_set, scoring_metric, threads)
            score_str = f"{score:.2f}" if score is not None else "n/a (empty val)"
        except Exception as e:
            score_str = "None (val eval failed)"
            print(f"WARNING: val-score eval failed ({e!r}). The compiled program "
                  f"is the paid artifact — continuing to checkpoint save; "
                  f"val_score recorded as None.")
        print(f"Val score (F2 blend): {score_str}")

    rollout_str = (f"rollouts~{monitor.calls + probe_n}/"
                   f"{projection.projected_rollouts}")
    try:
        stub = save_checkpoint(compiled, output_path, profile=profile,
                               train_size=len(train_set), val_size=len(val_set),
                               dataset_path=training_path,
                               extra={"budget": budget_str, "val_score": score,
                                      "costing": projection.to_dict(),
                                      "rollouts_observed": monitor.calls + probe_n})
    except Exception as e:
        fallback = Path.cwd() / f"lens-kit-emergency-{time.strftime('%Y%m%d-%H%M%S')}.json"
        try:
            compiled.save(str(fallback))
            fb_msg = f"emergency fallback saved: {fallback}"
        except Exception as e2:
            fb_msg = f"emergency fallback ALSO failed ({e2!r})"
        print(f"ERROR: checkpoint save failed ({e!r}); {fb_msg}")
        ledger.append_run(run_id=run_id, command=command_str, checkpoint_sha=None,
                          data_file=data_name,
                          metrics=f"save failed: {e!r}; {fb_msg}; "
                                  f"val_score={score_str}; {rollout_str}",
                          status="FAILED")
        print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")
        return 1

    print(f"Checkpoint saved: {output_path}")
    print(f"  sha256: {stub['sha256']}")
    print(f"  provenance stub: {output_path.with_suffix('.provenance.json')}")

    ledger.append_run(run_id=run_id, command=command_str,
                      checkpoint_sha=stub["sha256"], data_file=data_name,
                      metrics=f"val_score={score_str}; {rollout_str}",
                      status="KEEP")
    print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")
    return 0
