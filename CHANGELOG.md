# Changelog

## 2026-08-10 — Truth lens: BestOfN strictness sampling removed (N=1)

The truth lens previously ran `dspy.BestOfN(N=3)` with a strictness reward
that preferred the sample reporting the most violations — a design from the
era when a missing citation was itself the violation standard. Two things
made it wrong to keep: the standard changed (an uncited figure is no longer
a defect, so preferring the strictest sample now amplifies false positives),
and every evaluation number we bank measures the plain single-sample path —
the gate was shipping a deliberately stricter truth lens than the one being
measured. Truth now runs a plain `Predict`, identical to the measured path.
Extrapolation keeps BestOfN(N=3) pending its own review; `fast_mode` still
selects N=1 there. Checkpoint loading handles both wrapped-era and bare
key shapes. No absolute rates here per `docs/CLAIMS.md`; receipts live in
the internal run ledger.

## 2026-08-09 — Causality lens: boundary redrawn from omission to wrong-cause

`CausalityValidation`'s instructions previously flagged "recommendations or
cause-effect claims that have NO causal chain, mechanism, or evidence" — an
omission standard under which any recommendation without a spelled-out
mechanism could fire, which made the lens tax ordinary operational prose
(audit summaries, patch plans, marketing plans). The lens now fires only on
asserted wrong causal claims — post-hoc sequence-as-cause, correlation sold
as cause, one case generalised into a law, bare causal laws stated flatly as
fact, specific outcomes asserted as what named initiatives will produce, and
reversed or single-cause stories — and explicitly passes recommendations,
targets and feature-benefit lines that assert no cause, hands forecasts to
the Extrapolation lens, and ignores reported figures and tables.

Validated on our internal labelled sets under a pre-registered accept/revert
rule (dev iteration, then a single holdout confirmation): violation catch
unchanged on the holdout; clean-text false positives for this lens cut by
roughly half. Absolute rates are deliberately omitted — they are properties
of one model on our data on one date, not of this package (`docs/CLAIMS.md`);
receipts live in the internal run ledger, not shipped with this package.

## 2026-06-17 — Opt-in parallel lens execution (default on)

`LensGate(parallel=True)` (now the default) runs independent lenses
concurrently per dependency stage instead of one-at-a-time. Select the
unchanged sequential reference with `LensGate(parallel=False)`.

**Proven verdict-identical, not assumed:**

- **Load-bearing proof — deterministic mocked-LM equivalence**
  (`tests/test_parallel_equivalence.py`): with every lens stubbed to fixed
  outputs, the parallel path is byte-identical to the sequential reference
  on `passed`, `halted`, `halt_reason`, `fixed_text`, `consciousness_flags`,
  `per_lens`, and `violations` (lens, severity, issue, order) — across HALT,
  rights-scrub, truth-mask/fix, extrapolation-fix, auto-fix, no-context,
  scenario-vocab, a lens raising, a claim-extractor raising, and cross-check.
- **Corroboration — live holdout non-inferiority** (n=113, one mid-size
  open-weight model on an OpenAI-compatible endpoint, our own internal
  holdout): Δcatch +0.011 between modes, inside the measured run-to-run noise
  band of 0.011; false-positive rate identical to three decimal places. Single
  run per mode, temp 0.1 — this corroborates the deterministic proof above, it
  is not itself the proof. Absolute rates are deliberately omitted: they are
  properties of that model on our data on that date, not of this package, and
  quoting them here would breach the claim gate in `docs/CLAIMS.md`.
- **Thread-safety** (`tests/test_parallel_thread_safety.py`): worker threads
  capture `dspy.settings.lm` and re-enter `dspy.context(lm=…)` (dspy's LM is
  thread-local; a naked worker would lose the caller's scoped LM). The test is
  discriminating — it fails on the naked dispatch.

**Divergence found and fixed during rollout:** `claim_extractor.extract` is
un-wrapped in sequential `forward()`, so a failure there propagates out of
the gate (the eval harness counts it as an error). The first parallel draft
swallowed that exception to "no claims" and continued. Fixed to propagate;
added a regression fixture (`test_equiv_claim_extractor_raises_propagates_both_paths`).

**Measured wall-time** (one violation-dense finance input, a mid-size
open-weight model on a hosted OpenAI-compatible endpoint): sequential 35.5s →
parallel 19.6s (1.81×). A single measurement on one input, not a benchmark
suite; the gain scales with how many slow lenses overlap in a stage and with
your endpoint's latency, so treat 1.81× as an illustration, not a spec.

Concurrency mechanism: `concurrent.futures.ThreadPoolExecutor` over the
blocking, IO-bound `dspy.Predict` calls, scheduled in 5 dependency stages so
each lens receives the exact text tier it reads in the sequential path.
The design spec and rollout plan for this change are internal and not
published; the shipped receipts are the two test files named above, which are
the part you can actually run.
