# Changelog

## 0.1.0 — 2026-09-03 — First packaged GitHub release

The first GitHub Release ships eight artifacts: wheel and source archives for
the `lens-kit` core plus installable Codex, Claude Code / Agent SDK, and Grok
stop-hook adapters. GitHub CI runs the core and each adapter suite separately,
then builds, checks and clean-installs the release artifacts. A `v0.1.0` tag
creates the GitHub Release only after those jobs pass and publishes a
`SHA256SUMS` manifest alongside the packages.

The Codex adapter keeps candidate verdict separate from workflow disposition:
one bounded correction for non-Rights findings, risk-sensitive escalation for
unresolved public or high-risk work, and a HALT request for Rights. Host-enforced
continuation caps remain authoritative.

## 2026-08-29 — Stop-hook gate published (Claude Code + Agent SDK)

`integrations/claude-code/` ships the outside-in gate as a Claude Code **Stop
hook** and as a `claude-agent-sdk` hook. It lenses the agent's finished answer;
on HOLD it blocks the turn and feeds the violations back as revision guidance,
capped at one rework so it can never wedge. PASS finishes silently; an
unavailable lens fails open with a visible "not validated" note, or blocks if
you set `LENS_HOOK_FAIL=closed`. The validator is a separate model — the agent
does not grade itself.

Two defects fixed on the way out, both the same shape: a default that could not
work where it mattered.

The profile default was the relative string `profiles/qwen-serve.yaml`, which
only resolved if the process happened to start in one particular directory. A
hook never does, so the default failed in the only context this package exists
for, and the docs carried a "set an absolute path" warning to compensate. It now
falls back to `builtin_profile_path()` — the profile shipped inside `lens_kit` —
so an unset `LENS_HOOK_PROFILE` works out of the box.

The suggested hook `timeout` was `30000` ms, justified by one typical-duration
observation rather than a tail measurement. Gate cost scales with answer length,
so the suggestion was smallest exactly where answers are most worth gating. It is
now `120000` ms with the sizing rule written down: **a hook timeout is a ceiling,
not a delay**, so an oversized one costs nothing when the gate is fast, whereas
an undersized one returns an unavailable result. The configured fail and risk
policies then decide whether the adapter delivers with a visible warning or
requests another block. Measure it on your own provider before trusting a
smaller number.

The test harness also assumed a directory that exists only in our working repo,
which errored every test at import in a clean checkout. It now falls back to the
packaged profile, so the no-credential adapter suite can run from a clean
checkout.

## 2026-08-10 — Structure lens: runs unconditionally; scoped precondition/rollback filter

Two defects fixed, found by a mechanical trace of the serving path. First,
the structure lens used a keyword predicate (`text_has_recommendations`) to
decide whether to run at all, then reported the skipped case as a clean
pass — on a realistic corpus that silently disabled the lens for most
documents while the wire format showed `passed: true`. A detector must not
decide whether to look using a substring list: structure now runs on every
request, and the predicate no longer gates dispatch. Second,
`filter_structure_complete` returned an empty list whenever the text
contained both a precondition keyword and a rollback keyword — deleting
every structure finding, related or not. It now drops only violations whose
issue text is about those topics, and a regression test pins the scoped
behaviour. Cost note: this can add one structure-lens LLM call for inputs that
the old keyword predicate would have skipped; inputs that already ran the
structure lens are unaffected. Extrapolation keeps BestOfN(N=3): a same-day
paired two-arm measurement showed the wrapper earns its strictness on catch,
unlike truth's case — receipts in the internal run ledger.

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

The public package makes no performance claim for this instruction change.
Users should evaluate it on their own labelled data as described in
`docs/CLAIMS.md`.

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
- **Thread-safety** (`tests/test_parallel_thread_safety.py`): worker threads
  capture `dspy.settings.lm` and re-enter `dspy.context(lm=…)` (dspy's LM is
  thread-local; a naked worker would lose the caller's scoped LM). The test is
  discriminating — it fails on the naked dispatch.

**Divergence found and fixed during rollout:** `claim_extractor.extract` is
un-wrapped in sequential `forward()`, so a failure there propagates out of
the gate (the eval harness counts it as an error). The first parallel draft
swallowed that exception to "no claims" and continued. Fixed to propagate;
added a regression fixture (`test_equiv_claim_extractor_raises_propagates_both_paths`).

Concurrency mechanism: `concurrent.futures.ThreadPoolExecutor` over the
blocking, IO-bound `dspy.Predict` calls, scheduled in 5 dependency stages so
each lens receives the exact text tier it reads in the sequential path.
The design spec and rollout plan for this change are internal and not
published; the shipped receipts are the two test files named above, which are
the part you can actually run.
