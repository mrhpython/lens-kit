# Validator agent — receipt templates

These are the verdict receipts a validator agent renders. A receipt is the
record of a validation: what was checked, what the deterministic kit said, what
the agent reasoned, and what the verdict of record is. The gate's verdict is the
score of record; the agent's reasoning supplements it but never overrides a FAIL
into a PASS.

Copy a template verbatim and fill the bracketed slots. Keep the field order —
downstream readers and any tooling depend on it.

Three receipt shapes:

1. **PASS / FAIL receipt** — the per-lens table plus cross-checks (the common case).
2. **CONDITIONAL receipt** — a fixable verdict with a line-located fix, a
   `[PROJECTION]` post-fix score, and a sized re-check.
3. **Paired v1 + v2 verdicts** — the convention for preserving a fix cycle:
   both verdicts are kept, never overwrite one with the other.

---

## 1. PASS / FAIL receipt (per-lens table)

The gate (`lens-kit validate <file> --profile <yaml> --json`) returns
`passed`, `halted`, `per_lens` (a `{lens: bool}` map), `violations` (each with
`lens`, `severity`, `issue`), and `consciousness_flags`. Render it as a table.

```
## VERDICT — <artifact path>
verdict: PASS | FAIL | HALT | UNKNOWN
domain: <domain>    artifact_type: <type>    customer-facing: yes | no
gate: lens-kit validate (profile: <yaml>)    exit: <0|1|2>

| lens          | gate   | note |
|---------------|--------|------|
| rights        | PASS   |      |
| truth         | PASS   |      |
| causality     | PASS   |      |
| definitionalIntegrity | PASS | undefined-term / equivocation; blocking |
| contradiction | PASS   |      |
| extrapolation | PASS   |      |
| structure     | PASS   |      |
| consistency   | PASS   |      |
| relevance     | PASS   | warning-only; needs --context |
| consciousScan | (flags)| <N> flag(s), non-blocking |

violations (gate): <N>
  [<lens>/<severity>] <issue>
  ...

cross-checks (deterministic, no LLM):
  consistency markers : clean | VIOLATION (<N>)   exit <0|6>
  consistency leaks   : clean | VIOLATION (<N>)   exit <0|6>
  consistency numbers : clean | VIOLATION (<N>)   exit <0|6>

downstream consequence if wrong (ADVISORY — attention, not a score input):
  <the load-bearing claim, what it costs downstream if it is wrong, and that it
   got the strictest reading; this never changes the verdict arithmetic>

agent reasoning (SUPPLEMENT — does not override the gate):
  <cross-file / arithmetic / policy notes the single-file gate cannot see>
  <each claim about a file/number is backed by a direct check, not plausibility>

verdict of record: PASS | FAIL | HALT | UNKNOWN
  <one line: the gate's verdict, plus any consistency exit-6 that forces FAIL>
```

Rules for filling it:

- **A lens not in `per_lens` was NOT evaluated** — on a Rights HALT, `per_lens`
  holds only `rights`; mark the rest `(not evaluated)`, never `PASS`.
- **`consciousScan` emits flags, not pass/fail** — record the flag count.
- **`definitionalIntegrity` is a blocking lens** (added 2026-06-14): FAILs on an
  undefined load-bearing term or equivocation; runs after causality (causality keeps
  derivation / missing-mechanism).
- **`relevance` is warning-only** and needs `--context`; with no context it does
  not block.
- **A consistency exit 6 (a dropped marker, a leak, an orphan number) forces the
  verdict to FAIL** regardless of the gate score — no score overrides a leak.
- **Gate unreachable / error → verdict UNKNOWN**, never PASS.
- **The `downstream consequence if wrong` line is ADVISORY.** It records where the
  strictest reading went and what being wrong would cost; it changes attention
  and ordering, never the verdict arithmetic. A high consequence does not turn a
  gate PASS into a FAIL, and a low one does not rescue a FAIL.

---

## 2. CONDITIONAL receipt (line-located fix + projected score + sized re-check)

Use this when the verdict is fixable in one bounded round: the offense is
located, quoted, tied to a rule, and a rewrite is proposed. The post-fix score
is a **projection**, not a measurement — mark it `[PROJECTION]`. State the size
of the re-check so the next pass is scoped, not a full re-run.

```
## VERDICT v1 — <artifact path>   (CONDITIONAL)
verdict: CONDITIONAL — fixable in one bounded round
domain: <domain>    artifact_type: <type>    customer-facing: yes | no
gate: lens-kit validate (profile: <yaml>)    exit: <1>    blocking lens: <lens>

offense:
  file:line   <path>:<line>
  quoted      "<the exact offending text>"
  rule        <the forward rule this breaks — from a prior catch or a lens>
  rewrite     "<the proposed corrected text>"

projected post-fix:
  score       [PROJECTION] ~<NN>/100   (a projection, not a measurement —
              re-run the gate to confirm)
  basis       <which lens clears and why, e.g. "Truth clears: the unsourced
              stat gains a named source; no other lens was blocking">

sized re-check:
  scope       <e.g. "re-validate the one changed section; regression-grep the
              forbidden phrase across all rendered files">
  command     lens-kit validate <file> --profile <yaml>
              lens-kit consistency leaks <files...> --profile <yaml>
  cost        <e.g. "one gate call + one deterministic scan — ~30s">
```

Discipline: **one bounded fix round per verdict cycle.** If the v2 re-check does
not clear, escalate to a human — do not keep iterating.

---

## 3. Paired v1 + v2 verdicts (both preserved)

When a CONDITIONAL is fixed and re-checked, the second verdict is **appended**,
not substituted. Both receipts live in the record so the fix cycle is auditable.
v2 is a delta verdict: it states what changed, shows the regression check over
the offending term, carries the untouched lenses forward, and gives the measured
score (no longer a projection).

```
## VERDICT v2 — <artifact path>   (delta — supersedes v1's CONDITIONAL, v1 kept above)
verdict: PASS | FAIL
gate: lens-kit validate (profile: <yaml>)    exit: <0|1>

changed since v1:
  <path>:<line>   "<old text>"  ->  "<new text>"

regression check:
  <e.g. "grep '<forbidden phrase>' across <N> rendered files -> 0 hits">
  <e.g. "lens-kit consistency leaks <files> --profile <yaml> -> exit 0 (clean)">

carried over from v1 (not re-touched):
  <lenses that were clean in v1 and unaffected by the fix>

measured post-fix:
  verdict     <PASS | FAIL | HALT | UNKNOWN — as returned this round>
  per_lens    <the changed lens now PASS; the rest unchanged>
  violations  <count by severity, from the receipt — not estimated>

verdict of record: PASS | FAIL
```

Conventions:

- **Never overwrite a verdict.** v1's CONDITIONAL stays in the record above v2.
  A reader must be able to see the cycle, not just the endpoint.
- **One lens per round.** v2 fixes the one blocking lens v1 named; it does not
  re-open clean lenses. If a fix touches a second lens, that is a new cycle.
- **The measured verdict replaces the projection.** v1 may carry a
  `[PROJECTION]`; v2 carries only what the gate actually returned this round.
  Do not invent a numeric score: the gate returns a verdict, a per-lens map and
  violations — it does not emit a 0-100 score, so writing one down would be the
  exact fabrication this kit exists to catch.
- **If v2 does not clear, stop and escalate.** The paired record then reads
  v1 CONDITIONAL → v2 still-failing → handed to a human; that is a complete,
  honest record, not a failure to finish.
