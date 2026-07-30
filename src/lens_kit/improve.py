"""lens-kit improve — the autoresearch challenger/promote loop.

WHY (the argument a customer reads in the README): a gate improves by HINDSIGHT.
Every defect the validator agent catches is recorded (``catches``); those caught
defects are the most valuable training signal there is — they are exactly the
flaws THIS gate let through or nearly did. ``improve`` closes that loop: it builds
a CHALLENGER gate by adding the exported catch-labels to the base trainset,
recompiles (through the existing costing gate — no spend escapes it), and
promotes the challenger over the incumbent baseline ONLY if it wins HONESTLY:
better than baseline beyond the measured variance envelope, AND alive on planted
flaws (mutation control). A within-noise "win", or a checkpoint that wins on
accuracy but rubber-stamps a planted mutant, is NOT promoted. Both outcomes get a
ledger row and a provenance sidecar — a kept-baseline is a receipt too.

This is the RLHS application (arXiv 2501.08617): the reward is conditioned on the
downstream OUTCOME (does the recompiled gate actually catch more, beyond noise),
not on the appearance of improvement.

DOCTRINE this module enforces:
  - the costing gate is inherited, never bypassed: the challenger compile runs
    through ``compile_harness.run_compile``, so its >2h projection hard-stop
    (exit 3) and pace-kill (exit 4) propagate unchanged.
  - both baseline and challenger are EVAL'd on the frozen holdout with a FRESH
    cache. A measurement correction is not an improvement: if baseline and
    challenger differed only because one replayed a stale cache, that delta is an
    artifact, not a win. Fresh-cache on both is how that artifact is excluded.
  - a within-envelope difference is KEPT-BASELINE. Honesty about variance is the
    point: a gate that promotes on noise is sycophantic toward its own progress.
  - promotion is a TWO-AXIS rule, not catch_rate alone. catch_rate is gameable —
    a flag-EVERYTHING gate scores catch_rate 1.0 (and trivially passes mutation
    control) while its false-positive rate explodes, directly breaking the "we
    don't over-flag" claim. So the challenger must win on catch_rate AND not
    materially regress on fp_rate, both judged at their WORST across reruns. See
    beats_baseline. fp_regression_tolerance (profile.improve, default 0.0) sets
    how much FP rise is allowed; the default allows none.
  - mutation control gates promotion: a winner that cannot fail a planted flaw is
    not safe to ship, no matter its accuracy. A mutation-control failure returns
    the integrity signal (exit 5) and keeps the baseline. (Mutation control does
    NOT rescue the flag-everything case — that is why the fp_rate bound lives in
    the promotion rule itself, not just here.)

SEAMS (offline-testable): the three operations that make LLM calls are isolated
behind module-level functions — ``_compile_challenger`` (wraps
compile_harness.run_compile), ``_eval_checkpoint`` (wraps the eval harness to a
structured MeasuredEval), and ``_mutation_control`` (wraps mutation.run_mutation_
control). Tests monkeypatch these three so the suite never touches the network;
the command's first REAL run (which reaches the real seams) is a future operator-
approved spend on a pinned model id.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from . import ledger
from .config import Profile

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_COST_STOP = 3      # inherited from compile_harness
EXIT_PACE_KILL = 4      # inherited from compile_harness
EXIT_RUBBER_STAMP = 5   # inherited from mutation control
# An inconclusive integrity check (the challenger won but mutation control
# returned a result that was neither clean (0) nor the rubber-stamp signal (5) —
# e.g. a mutation usage/setup error). Promotion is blocked: we never promote on
# an integrity check we could not conclude. 6 is the existing integrity/consistency
# family code (see cli.py exit-code doctrine), not a new code.
EXIT_INCONCLUSIVE_INTEGRITY = 6

# The PRIMARY metric the promotion decision is made on. catch_rate = tp/(tp+fn) =
# the share of real defects the gate caught. This is the gate's reason to exist:
# a higher catch rate means fewer planted flaws slipping through. It is also the
# RLHS-aligned outcome (did the recompiled gate actually catch MORE), not a proxy
# like overall_accuracy (which a flag-nothing gate can inflate on clean examples).
#
# But catch_rate ALONE is not enough, and getting this wrong contradicts the
# product's core claim ("we don't over-flag"). A gate that flags EVERYTHING has
# catch_rate 1.0 — it would always win on this axis — while its false-positive
# rate explodes and its accuracy collapses. Mutation control does NOT rescue this
# (a flag-everything gate catches every planted mutant and passes mutation
# control). So the promotion rule is TWO-AXIS: the challenger must win on
# catch_rate AND not materially regress on fp_rate, and BOTH axes are judged
# pessimistically (catch at its worst = MIN, fp at its worst = MAX across reruns).
PROMOTION_METRIC = "catch_rate"
FP_METRIC = "fp_rate"

# Default false-positive regression tolerance: 0.0 = no FP increase is allowed.
# This is the strict, defensible default — the product claim is "we don't
# over-flag", so a challenger that catches more by flagging more (any FP increase)
# is not an honest improvement. A profile may loosen it (compile-adjacent
# `improve.fp_regression_tolerance`) with a small documented epsilon when a tiny
# FP rise buys a large catch gain, but the default refuses any FP regression.
DEFAULT_FP_REGRESSION_TOLERANCE = 0.0

# What the promotion decision DID and did NOT check — shipped in every improve
# sidecar's not_a_claim_of so the receipt is honest about its own scope. The
# promotion gate checks two AGGREGATE axes (overall catch_rate up, overall fp_rate
# not worse, both pessimistic); it does NOT check per-lens regressions, latency,
# cost, or transfer. A promoted checkpoint can still have regressed an INDIVIDUAL
# lens while the aggregate improved.
_PROMOTION_BOUNDARY = (
    "an improvement on any axis OTHER than the two the promotion gate checked: "
    "the gate verified overall catch_rate went UP (judged at its worst) AND "
    "overall fp_rate did NOT materially regress (judged at its worst, within the "
    "configured tolerance) — it did NOT check per-lens regressions, latency, "
    "cost, or transfer; an individual lens may have regressed while the aggregate "
    "improved",
)


# ── Structured eval result (what a seam returns) ─────────────────

@dataclass
class MeasuredEval:
    """A single checkpoint's holdout measurement.

    ``summary``: the headline metrics (overall_accuracy / catch_rate / fp_rate),
    one point estimate (the last rerun, or the only run).
    ``envelope``: the variance envelope when reruns>1 (per-metric stats incl.
    worst_of_n), else None — a single run has no envelope and the promotion
    decision carries that caveat honestly.
    ``reruns``: how many measurements backed it.
    """
    summary: dict
    envelope: dict | None = None
    reruns: int = 1

    def headline(self, metric: str) -> float:
        return float(self.summary.get(metric, 0.0))

    def worst(self, metric: str) -> float | None:
        """worst-of-N floor for ``metric`` from the envelope, or None.

        For a higher-is-better metric this is the MIN observed across reruns —
        the floor the challenger must clear, not its lucky best.
        """
        if not self.envelope:
            return None
        block = self.envelope.get(metric)
        if not isinstance(block, dict):
            return None
        return block.get("worst_of_n")


# ── Promotion rule (pure, honesty-critical) ──────────────────────

@dataclass
class PromotionVerdict:
    promote: bool
    reason: str
    metric: str = PROMOTION_METRIC
    baseline_value: float = 0.0
    challenger_value: float = 0.0
    single_measurement: bool = False
    # The false-positive axis — surfaced on EVERY verdict so the receipt is
    # honest about what the FP bound checked. fp_*_value are judged
    # pessimistically (baseline headline vs challenger WORST = max-across-reruns).
    fp_baseline: float = 0.0
    fp_challenger: float = 0.0
    fp_delta: float = 0.0
    fp_tolerance: float = DEFAULT_FP_REGRESSION_TOLERANCE


def _worst_or_headline(meas: MeasuredEval, metric: str) -> float:
    """The pessimistic value for ``metric``: the envelope's worst-of-N when an
    envelope is present, else the single-run headline.

    For catch_rate (higher_is_better) worst-of-N is the MIN observed; for fp_rate
    (lower_is_better) it is the MAX observed — both the value that makes the
    challenger look LEAST good, by construction of build_envelope's direction map.
    A single-run challenger has no envelope, so its headline is its only estimate.
    """
    w = meas.worst(metric)
    return float(w) if w is not None else meas.headline(metric)


def beats_baseline(baseline: MeasuredEval, challenger: MeasuredEval,
                   *, metric: str = PROMOTION_METRIC,
                   fp_regression_tolerance: float = DEFAULT_FP_REGRESSION_TOLERANCE
                   ) -> PromotionVerdict:
    """Decide promotion. TWO axes, BOTH judged pessimistically. The exact rule:

    PRIMARY (catch_rate, higher-is-better): the challenger must clear the
    baseline's HEADLINE catch_rate with its WORST-CASE catch_rate (worst-of-N =
    MIN observed when an envelope exists; the single-run headline otherwise),
    STRICTLY. A tie or a within-envelope nominal lead is NOT a win.

    SECONDARY / GUARD (fp_rate, lower-is-better): the challenger must NOT
    materially regress on false positives. The challenger's WORST-CASE fp_rate
    (worst-of-N = MAX observed when an envelope exists; the single-run headline
    otherwise) must satisfy ``chal_fp_worst <= base_fp_headline + tolerance``.
    ``tolerance`` defaults to 0.0 (DEFAULT_FP_REGRESSION_TOLERANCE) — ANY FP
    increase blocks promotion, because the product claim is "we don't over-flag".
    A profile may set a small documented epsilon to allow a tiny FP rise that buys
    a large catch gain.

    WHY both axes, both pessimistic: catch_rate alone is gameable — a gate that
    flags EVERYTHING scores catch_rate 1.0 and would always "win", while its FP
    rate explodes and accuracy collapses (and it would still pass mutation
    control, which a flag-everything gate trivially does). The FP guard closes
    that hole IN THE RULE. Judging the challenger at its WORST on BOTH axes (best
    draw never flatters it) and the baseline at its stated value means we promote
    only on a win that survives a bad draw and does not buy catch with flags.

    The fp_* fields are populated on EVERY verdict (promote or keep) so the
    printed verdict, ledger metrics, and sidecar all carry the base->challenger FP
    delta honestly — the receipt names exactly what was checked.
    """
    tol = float(fp_regression_tolerance)
    base_val = baseline.headline(metric)
    chal_worst = challenger.worst(metric)
    chal_headline = challenger.headline(metric)
    single = chal_worst is None

    # FP axis (computed for the receipt regardless of the catch outcome).
    base_fp = baseline.headline(FP_METRIC)
    chal_fp_worst = _worst_or_headline(challenger, FP_METRIC)
    fp_delta = chal_fp_worst - base_fp
    fp_ok = chal_fp_worst <= base_fp + tol
    fp_note = (f"fp_rate base={base_fp:.4f} -> chal(worst)={chal_fp_worst:.4f} "
               f"(delta {fp_delta:+.4f}, tol {tol:.4f})")

    def verdict(promote: bool, reason: str, chal_val: float) -> PromotionVerdict:
        return PromotionVerdict(
            promote=promote, reason=reason, metric=metric,
            baseline_value=base_val, challenger_value=chal_val,
            single_measurement=single, fp_baseline=base_fp,
            fp_challenger=chal_fp_worst, fp_delta=fp_delta, fp_tolerance=tol)

    # ── PRIMARY axis: does the challenger win on catch_rate, pessimistically? ──
    chal_catch = chal_worst if not single else chal_headline
    catch_wins = chal_catch > base_val
    bound_word = "worst-of-N" if not single else "single-measurement"

    if not catch_wins:
        if single:
            reason = (f"challenger {metric}={chal_catch:.4f} does not beat "
                      f"baseline {metric}={base_val:.4f} (single measurement); "
                      f"keep baseline. {fp_note}")
        else:
            reason = (f"challenger worst-of-N {metric}={chal_catch:.4f} does NOT "
                      f"clear baseline {metric}={base_val:.4f} (within/below the "
                      f"variance envelope) — within-noise is not a win; keep "
                      f"baseline. {fp_note}")
        return verdict(False, reason, chal_catch)

    # ── GUARD axis: catch won — but did FP materially regress? ──
    if not fp_ok:
        reason = (f"challenger wins on {metric} ({bound_word} {chal_catch:.4f} > "
                  f"{base_val:.4f}) BUT false-positives regressed: {fp_note}. A "
                  f"gate that catches more by FLAGGING MORE is over-flagging, not "
                  f"improving — keep baseline.")
        return verdict(False, reason, chal_catch)

    # ── Both axes satisfied -> promote. ──
    if single:
        reason = (f"challenger {metric}={chal_catch:.4f} > baseline "
                  f"{metric}={base_val:.4f} on a SINGLE measurement (no variance "
                  f"envelope — re-run with --reruns N to bound it), and FP did not "
                  f"regress: {fp_note}")
    else:
        reason = (f"challenger worst-of-N {metric}={chal_catch:.4f} clears "
                  f"baseline {metric}={base_val:.4f} beyond the variance envelope, "
                  f"and FP did not regress: {fp_note}")
    return verdict(True, reason, chal_catch)


# ── Seams (the only LLM-touching operations; monkeypatched in tests) ──

def _compile_challenger(training_path, profile: Profile, output_path, *,
                        approve_cost: bool = False, fresh_cache: bool = True,
                        threads: int = 1, auto=None, max_evals=None,
                        val_split: float = 0.2, patience: int = 10,
                        command_str: str = "lens-kit improve (compile)") -> int:
    """Compile the challenger THROUGH the existing costing gate.

    Thin pass-through to compile_harness.run_compile — the costing gate
    (projection + >2h hard-stop exit 3, pace-kill exit 4) is INHERITED here, not
    re-implemented. ``fresh_cache=True`` by default: the challenger must be
    measured on real pace/rollouts, not a stale cache replay. Returns run_compile's
    exit code unchanged (0 / 2 / 3 / 4).
    """
    from .compile_harness import run_compile

    return run_compile(
        training_path, profile, output_path,
        auto=auto, max_evals=max_evals, threads=threads, val_split=val_split,
        patience=patience, approve_cost=approve_cost, fresh_cache=fresh_cache,
        command_str=command_str)


def _eval_checkpoint(checkpoint, holdout_path, profile: Profile, *,
                     reruns: int = 1, fresh_cache: bool = True) -> MeasuredEval:
    """Measure one checkpoint on the frozen holdout -> a MeasuredEval (in-process).

    Uses the eval-harness primitives directly (rather than run_eval_command,
    which only returns an exit code and writes files) so the structured summary +
    variance envelope come back to the caller. Both baseline and challenger are
    measured with ``fresh_cache=True`` by default: a shared/stale cache replay
    would make a measurement-artifact look like an improvement — fresh-on-both is
    how that is excluded (the doctrine in the module docstring).
    """
    from .config import lm_context
    from .eval_harness import (build_envelope, configure_fresh_cache, load_holdout,
                               make_gate, preserve_dspy_cache, run_assessment,
                               summarize)

    examples, _meta = load_holdout(holdout_path)
    summaries = []
    with preserve_dspy_cache():
        for _ in range(max(1, reruns)):
            if fresh_cache:
                configure_fresh_cache()
            gate = make_gate(profile, checkpoint)
            with lm_context(profile):
                results = run_assessment(examples, gate, progress=False)
            summaries.append(summarize(results))
    envelope = build_envelope(summaries) if len(summaries) > 1 else None
    return MeasuredEval(summary=summaries[-1], envelope=envelope,
                        reruns=len(summaries))


def _mutation_control(checkpoint, holdout_path, profile: Profile, *,
                      seed: int = 0, per_type: int = 1,
                      output_dir=None,
                      command_str: str = "lens-kit improve (mutation-control)") -> int:
    """Run mutation control on a checkpoint -> exit code (0 ok / 5 rubber-stamp).

    The challenger checkpoint is loaded into the gate first, so the mutants are
    judged by the CHALLENGER, not the bare baseline gate. Thin pass-through to
    mutation.run_mutation_control with that checkpoint-loaded gate.
    """
    from .config import lm_context
    from .eval_harness import make_gate
    from .mutation import run_mutation_control

    gate = make_gate(profile, checkpoint)
    with lm_context(profile):
        return run_mutation_control(
            holdout_path, profile, seed=seed, per_type=per_type,
            output_dir=output_dir, command_str=command_str, gate=gate)


# ── Trainset merge (base + exported catch labels) ────────────────

def build_challenger_trainset(base_trainset_path, catches_path, *,
                              domain_filter=None) -> tuple[list, int, int]:
    """base trainset + catch-labels, deduped by text against the base.

    Returns ``(merged_rows, added, skipped)``:
      merged_rows -> the base rows followed by the eligible, non-duplicate catch
                     labels (each a {text, domain, per_lens, source, source_id}).
      added       -> how many catch labels were actually appended (after dedupe).
      skipped     -> catches that were not eligible (no snippet / lenses_failed),
                     as reported by build_export_labels — NOT a silent cap.

    Dedupe is by the label ``text`` against the base trainset texts: a catch
    whose snippet is already a base example adds no new signal. (build_export_
    labels has already deduped its OWN output by source_id.)
    """
    from .catches import build_export_labels, load_catches

    base = json.loads(Path(base_trainset_path).read_text())
    if isinstance(base, dict):
        base = base.get("examples", [])
    base_texts = {r.get("text", "") for r in base}

    records = load_catches(catches_path)
    labels, skipped = build_export_labels(records, domain_filter=domain_filter)

    merged = list(base)
    added = 0
    for label in labels:
        if label["text"] in base_texts:
            continue  # already represented in the base trainset
        merged.append(label)
        base_texts.add(label["text"])
        added += 1
    return merged, added, skipped


# ── Orchestration ────────────────────────────────────────────────

def _validate_inputs(paths: dict) -> str | None:
    """Return an error message for the first missing input, or None."""
    for label, p in paths.items():
        if not Path(p).exists():
            return f"{label} not found: {p}"
    return None


def _write_outcome(output_dir: Path, *, run_id: str, command_str: str,
                   data_name: str, status: str, metrics: str,
                   checkpoint_sha: str | None,
                   ledger_path=None) -> None:
    ledger.append_run(
        run_id=run_id, command=command_str, checkpoint_sha=checkpoint_sha,
        data_file=data_name, metrics=metrics, status=status,
        ledger_path=ledger_path)
    print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")


def run_improve(*, trainset, holdout, baseline_checkpoint, catches, output_dir,
                profile: Profile, domain_filter=None, approve_cost: bool = False,
                threads: int = 1, auto=None, max_evals: int | None = None,
                val_split: float = 0.2, patience: int = 10, reruns: int = 1,
                allow_single_measurement: bool = False,
                command_str: str = "lens-kit improve",
                ledger_path=None) -> int:
    """One challenger/promote iteration. Returns a CLI exit code.

    0  -> ran cleanly; PROMOTED or KEPT-BASELINE (both honest outcomes).
    2  -> usage error (a missing input).
    3  -> the challenger compile hit the costing-gate hard stop (>2h projection).
    4  -> the challenger compile pace-killed.
    5  -> the challenger WON on the metric but failed mutation control
          (rubber-stamp) — baseline kept, integrity signal returned.
    6  -> the challenger won but mutation control was INCONCLUSIVE (a non-{0,5}
          result we can't read as clean-or-rubber-stamp) — baseline kept; we
          never promote on an integrity check we could not conclude.

    ``allow_single_measurement`` (default False): a challenger with NO variance
    envelope (reruns==1) is a single point estimate, not a bound. "A measurement
    correction is not an improvement" — a single lucky draw must not silently
    promote. Without this opt-in and with no envelope, a nominal single-run win is
    KEPT-BASELINE with a printed note that an envelope (--reruns N) or the explicit
    --allow-single-measurement opt-in is required. WITH the opt-in, the win
    promotes but the single-measurement caveat is still recorded honestly in the
    ledger row + sidecar.

    Structure is single-iteration by design (the plan's v1); multi-iteration is a
    future extension (loop over this body, feeding the promoted checkpoint back as
    the new baseline), deliberately not built here.
    """
    err = _validate_inputs({
        "trainset": trainset, "holdout": holdout,
        "baseline-checkpoint": baseline_checkpoint, "catches": catches})
    if err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_USAGE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_name = Path(holdout).name
    run_id = ledger.new_run_id("improve")

    # ── 1. Build the challenger trainset (base + catch labels) ──
    merged, added, skipped = build_challenger_trainset(
        trainset, catches, domain_filter=domain_filter)
    print(f"Challenger trainset: {len(merged)} examples "
          f"({added} catch-label(s) added, {skipped} catch(es) skipped — "
          f"missing snippet/lenses_failed)")
    if added == 0:
        print("Note: 0 catch-labels added — recompiling on the base trainset "
              "alone (a base-only recompile is still a valid challenger).")

    challenger_trainset_path = output_dir / "challenger-trainset.json"
    challenger_trainset_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── 2. Compile the challenger THROUGH the costing gate ──
    challenger_ckpt = output_dir / "challenger.json"
    print(f"\nCompiling challenger (through the costing gate)...")
    rc = _compile_challenger(
        challenger_trainset_path, profile, challenger_ckpt,
        approve_cost=approve_cost, threads=threads, auto=auto, max_evals=max_evals,
        val_split=val_split, patience=patience,
        command_str=command_str + " (compile)")
    if rc != EXIT_OK:
        # The costing gate stopped (3) or pace-killed (4), or a usage/save error
        # (2/1). No promotion; baseline untouched. compile_harness already wrote
        # its own ledger row for the stop; we add the improve-level outcome row.
        status = {EXIT_COST_STOP: "STOPPED_COST (challenger compile)",
                  EXIT_PACE_KILL: "KILLED (challenger compile)"}.get(
                      rc, f"COMPILE_FAILED (rc={rc})")
        _write_outcome(output_dir, run_id=run_id, command_str=command_str,
                       data_name=data_name, status=status,
                       metrics=f"challenger compile returned {rc}; no promotion",
                       checkpoint_sha=None, ledger_path=ledger_path)
        print(f"\nChallenger compile returned {rc} — no promotion, baseline "
              f"untouched.")
        return rc

    # ── 3. Eval both on the frozen holdout, fresh cache on BOTH ──
    print("\nEvaluating baseline on the frozen holdout (fresh cache)...")
    base_eval = _eval_checkpoint(baseline_checkpoint, holdout, profile,
                                 reruns=reruns, fresh_cache=True)
    print("Evaluating challenger on the frozen holdout (fresh cache)...")
    chal_eval = _eval_checkpoint(challenger_ckpt, holdout, profile,
                                 reruns=reruns, fresh_cache=True)

    fp_tolerance = profile.improve.fp_regression_tolerance
    verdict = beats_baseline(base_eval, chal_eval,
                             fp_regression_tolerance=fp_tolerance)
    print(f"\nPromotion rule: catch_rate UP (pessimistic) AND fp_rate not worse "
          f"(tolerance {verdict.fp_tolerance:.4f})")
    print(f"  baseline   {verdict.metric}={verdict.baseline_value:.4f}  "
          f"fp_rate={verdict.fp_baseline:.4f}")
    print(f"  challenger {verdict.metric}={verdict.challenger_value:.4f}"
          + (" (worst-of-N)" if not verdict.single_measurement else
             " (single measurement)")
          + f"  fp_rate={verdict.fp_challenger:.4f} (worst-of-N)")
    print(f"  fp delta: {verdict.fp_delta:+.4f}")
    print(f"  decision: {verdict.reason}")

    metrics_summary = (
        f"{verdict.metric}: base={verdict.baseline_value:.4f} "
        f"chal={verdict.challenger_value:.4f}; "
        f"fp_rate: base={verdict.fp_baseline:.4f} "
        f"chal={verdict.fp_challenger:.4f} (delta {verdict.fp_delta:+.4f}, "
        f"tol {verdict.fp_tolerance:.4f})"
        + ("; SINGLE-MEASUREMENT (no envelope)" if verdict.single_measurement
           else "; envelope-bounded"))

    # ── 4. Keep-baseline path (loss / tie / within-envelope) ──
    if not verdict.promote:
        _keep_baseline(
            output_dir, run_id=run_id, command_str=command_str,
            data_name=data_name, profile=profile, challenger_ckpt=challenger_ckpt,
            challenger_trainset_path=challenger_trainset_path, base_eval=base_eval,
            chal_eval=chal_eval, verdict=verdict, metrics_summary=metrics_summary,
            reason="metric not beaten beyond envelope", ledger_path=ledger_path)
        return EXIT_OK

    # ── 4b. Single-measurement gate (FIX 2): a nominal win with NO variance
    # envelope is a single point estimate. "A measurement correction is not an
    # improvement" — a single lucky draw must not silently promote. Without the
    # explicit opt-in, KEEP BASELINE and tell the operator to bound it. ──
    if verdict.single_measurement and not allow_single_measurement:
        print("\nSINGLE-MEASUREMENT win: the challenger beat baseline on ONE "
              "measurement with no variance envelope. A single draw is not a "
              "bound — re-run with --reruns N to measure an envelope, or pass "
              "--allow-single-measurement to promote on this point estimate "
              "anyway (the caveat is recorded either way). Not promoting; "
              "baseline kept.")
        _keep_baseline(
            output_dir, run_id=run_id, command_str=command_str,
            data_name=data_name, profile=profile, challenger_ckpt=challenger_ckpt,
            challenger_trainset_path=challenger_trainset_path, base_eval=base_eval,
            chal_eval=chal_eval, verdict=verdict,
            metrics_summary=metrics_summary + "; SINGLE-MEASUREMENT not opted in",
            reason="won on a single measurement, no variance envelope, and "
                   "--allow-single-measurement was not set",
            ledger_path=ledger_path)
        return EXIT_OK

    # ── 5. Mutation control GATES promotion ──
    print("\nChallenger won on the metric — running mutation control before "
          "promoting (a winner that can't fail a planted flaw is not safe)...")
    mut_rc = _mutation_control(challenger_ckpt, holdout, profile,
                               output_dir=output_dir,
                               command_str=command_str + " (mutation-control)")
    if mut_rc == EXIT_RUBBER_STAMP:
        print("\nMUTATION CONTROL FAILED: the challenger let a planted mutant "
              "through. It wins on accuracy but rubber-stamps a defect class — "
              "NOT promoted. Baseline kept.")
        _keep_baseline(
            output_dir, run_id=run_id, command_str=command_str,
            data_name=data_name, profile=profile, challenger_ckpt=challenger_ckpt,
            challenger_trainset_path=challenger_trainset_path, base_eval=base_eval,
            chal_eval=chal_eval, verdict=verdict,
            metrics_summary=metrics_summary + "; MUTATION-CONTROL FAILED",
            reason="won on metric but failed mutation control (rubber-stamp)",
            ledger_path=ledger_path)
        return EXIT_RUBBER_STAMP
    if mut_rc != EXIT_OK:
        # An unexpected non-zero (not the clean 0, not the rubber-stamp 5): we
        # could NOT conclude the integrity check (e.g. a mutation usage/setup
        # error). All such results map to ONE documented outcome — do not promote
        # on an integrity check we could not conclude. Exit 6 (integrity family),
        # never the raw mutation rc (FIX 3).
        print(f"\nMutation control returned {mut_rc} — neither clean (0) nor the "
              f"rubber-stamp signal (5). The integrity check is INCONCLUSIVE; we "
              f"never promote on an integrity check we could not conclude. "
              f"Baseline kept.")
        _keep_baseline(
            output_dir, run_id=run_id, command_str=command_str,
            data_name=data_name, profile=profile, challenger_ckpt=challenger_ckpt,
            challenger_trainset_path=challenger_trainset_path, base_eval=base_eval,
            chal_eval=chal_eval, verdict=verdict,
            metrics_summary=metrics_summary
            + f"; MUTATION-CONTROL INCONCLUSIVE (rc={mut_rc})",
            reason=f"mutation control inconclusive (rc={mut_rc}) — could not "
                   f"conclude the integrity check",
            ledger_path=ledger_path)
        return EXIT_INCONCLUSIVE_INTEGRITY

    # ── 6. Promote ──
    return _promote(
        output_dir, run_id=run_id, command_str=command_str, data_name=data_name,
        profile=profile, challenger_ckpt=challenger_ckpt,
        challenger_trainset_path=challenger_trainset_path, chal_eval=chal_eval,
        verdict=verdict, metrics_summary=metrics_summary, ledger_path=ledger_path)


def _eval_results_path_for_sidecar(output_dir: Path, meas: MeasuredEval,
                                   stem: str) -> Path | None:
    """Persist a MeasuredEval as a sidecar-readable eval doc, return its path.

    The sidecar generator reads an eval-results/envelope JSON off disk. We write
    the MeasuredEval in a shape sidecar._eval_block accepts (single-run docs carry
    a 'summary'; multi-run docs carry 'metrics'). Returns None on failure (the
    sidecar is still generated, just without the eval block).
    """
    try:
        if meas.envelope is not None:
            doc = {"metrics": meas.envelope, "reruns": meas.reruns}
        else:
            doc = {"summary": meas.summary, "reruns": meas.reruns}
        path = output_dir / f"{stem}-eval.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return path
    except Exception:
        return None


def _promote(output_dir: Path, *, run_id, command_str, data_name, profile,
             challenger_ckpt: Path, challenger_trainset_path: Path,
             chal_eval: MeasuredEval, verdict: PromotionVerdict,
             metrics_summary: str, ledger_path) -> int:
    """Write the promoted checkpoint + sidecar + a PROMOTED ledger row."""
    import shutil

    from .checkpoint import sha256_file
    from .sidecar import generate_sidecar

    promoted = output_dir / "promoted.json"
    shutil.copy2(challenger_ckpt, promoted)
    sha = sha256_file(promoted)

    extra_caveats = list(_PROMOTION_BOUNDARY)
    if verdict.single_measurement:
        extra_caveats.append(
            "promoted on a SINGLE eval measurement — no variance envelope "
            "backed the win (re-run improve with --reruns N to bound it)")
    eval_doc = _eval_results_path_for_sidecar(output_dir, chal_eval, "promoted")
    _sidecar, sidecar_path = generate_sidecar(
        promoted, profile=profile, dataset_path=challenger_trainset_path,
        eval_results_path=eval_doc, extra_not_a_claim_of=extra_caveats,
        extra={"improve_outcome": "PROMOTED",
               "promotion_metric": verdict.metric,
               "baseline_value": verdict.baseline_value,
               "challenger_value": verdict.challenger_value,
               "fp_baseline": verdict.fp_baseline,
               "fp_challenger": verdict.fp_challenger,
               "fp_delta": verdict.fp_delta,
               "fp_regression_tolerance": verdict.fp_tolerance,
               "single_measurement": verdict.single_measurement,
               "promotion_reason": verdict.reason})

    print(f"\nPROMOTED: challenger -> {promoted}")
    print(f"  sha256: {sha}")
    print(f"  sidecar: {sidecar_path}")
    _write_outcome(output_dir, run_id=run_id, command_str=command_str,
                   data_name=data_name, status="PROMOTED",
                   metrics=metrics_summary, checkpoint_sha=sha,
                   ledger_path=ledger_path)
    return EXIT_OK


def _keep_baseline(output_dir: Path, *, run_id, command_str, data_name, profile,
                   challenger_ckpt: Path, challenger_trainset_path: Path,
                   base_eval: MeasuredEval, chal_eval: MeasuredEval,
                   verdict: PromotionVerdict, metrics_summary: str, reason: str,
                   ledger_path) -> None:
    """Keep the incumbent: NO promoted checkpoint, but the challenger artifact is
    retained for inspection and gets its own (clearly not-promoted) sidecar + a
    KEPT-BASELINE ledger row. The baseline checkpoint is never touched."""
    from .checkpoint import sha256_file
    from .sidecar import generate_sidecar

    sha = sha256_file(challenger_ckpt) if challenger_ckpt.exists() else None
    extra_caveats = ["NOT PROMOTED — this challenger was kept for inspection "
                     "only; the incumbent baseline remains the gate of record"]
    extra_caveats.extend(_PROMOTION_BOUNDARY)
    if verdict.single_measurement:
        extra_caveats.append(
            "comparison was a SINGLE eval measurement — no variance envelope")
    eval_doc = _eval_results_path_for_sidecar(output_dir, chal_eval, "challenger")
    try:
        _sidecar, sidecar_path = generate_sidecar(
            challenger_ckpt, profile=profile,
            dataset_path=challenger_trainset_path, eval_results_path=eval_doc,
            extra_not_a_claim_of=extra_caveats,
            extra={"improve_outcome": "KEPT-BASELINE",
                   "kept_baseline_reason": reason,
                   "promotion_metric": verdict.metric,
                   "baseline_value": verdict.baseline_value,
                   "challenger_value": verdict.challenger_value,
                   "fp_baseline": verdict.fp_baseline,
                   "fp_challenger": verdict.fp_challenger,
                   "fp_delta": verdict.fp_delta,
                   "fp_regression_tolerance": verdict.fp_tolerance,
                   "single_measurement": verdict.single_measurement})
        print(f"  challenger artifact retained (not promoted): {challenger_ckpt}")
        print(f"  challenger sidecar: {sidecar_path}")
    except Exception as e:
        print(f"  (challenger sidecar generation skipped: {e!r})")

    print(f"\nKEPT-BASELINE: {reason}. The incumbent baseline is unchanged.")
    _write_outcome(output_dir, run_id=run_id, command_str=command_str,
                   data_name=data_name, status="KEPT-BASELINE",
                   metrics=metrics_summary, checkpoint_sha=sha,
                   ledger_path=ledger_path)


# ── --status ─────────────────────────────────────────────────────

def run_improve_status(*, ledger_path=None) -> int:
    """Read the ledger; report the last improve outcome(s). Mirrors auto_improve
    --status, kit-native: the incumbent + the last challenger result + whether it
    was promoted, straight from the RUNS.md rows this command appended."""
    path = Path(ledger_path) if ledger_path else Path.cwd() / ledger.LEDGER_FILENAME
    if not path.exists():
        print("No RUNS.md ledger found — no improve runs recorded yet.")
        return EXIT_OK

    rows = _improve_rows(path)
    if not rows:
        print("No improve runs recorded yet (no PROMOTED / KEPT-BASELINE rows in "
              "the ledger).")
        return EXIT_OK

    print("=== lens-kit improve — recent outcomes ===")
    for row in rows[-5:]:
        print(f"  {row['run_id']:<28} {row['status']:<28} {row['metrics']}")
    last = rows[-1]
    print(f"\nLast outcome: {last['status']}")
    print(f"  run: {last['run_id']}")
    print(f"  metrics: {last['metrics']}")
    if last["status"].startswith("PROMOTED"):
        print("  -> the challenger BEAT the incumbent and is the new gate of "
              "record.")
    else:
        print("  -> the incumbent baseline was KEPT (challenger did not win "
              "honestly).")
    return EXIT_OK


def _improve_rows(ledger_file: Path) -> list[dict]:
    """Parse RUNS.md rows whose run id is an improve run. Markdown-table parse —
    same shape ledger.append_run writes (| run_id | date | command | sha | data |
    metrics | status |)."""
    rows = []
    for line in ledger_file.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| improve-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        rows.append({"run_id": cells[0], "date": cells[1], "command": cells[2],
                     "sha": cells[3], "data": cells[4], "metrics": cells[5],
                     "status": cells[6]})
    return rows
