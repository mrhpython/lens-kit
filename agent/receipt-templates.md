# Validator agent — receipt templates

These are the verdict receipts a validator agent renders. A receipt is the
record of a validation: what was checked, what the deterministic kit said, what
the agent reasoned, the frozen candidate's verdict, and the separate workflow
disposition. The gate's response is preserved literally; the agent's reasoning
supplements it and never rewrites an external FAIL into PASS.

Copy a template verbatim and fill the bracketed slots. Keep the field order —
downstream readers and any tooling depend on it.

Three receipt shapes:

1. **Verdict receipt** — the per-lens table, cross-checks, and disposition.
2. **REVISE receipt** — bounded findings returned to a separate producer.
3. **Paired v1 + v2 verdicts** — the convention for preserving a correction cycle:
   both verdicts are kept, never overwrite one with the other.

---

## 1. Verdict receipt (per-lens table + disposition)

The gate (`lens-kit validate <file> --profile <yaml> --json`) returns
`passed`, `halted`, `per_lens` (a `{lens: bool}` map), `violations` (each with
`lens`, `severity`, `issue`), and `consciousness_flags`. Render it as a table.

```
## VERDICT — <artifact path>
verdict: PASS | FAIL | HALT | UNKNOWN
disposition: SHIP | REVISE | DELIVER_WITH_WARNINGS | ESCALATE | HALT
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
  consistency markers : N/A | clean | TRIPWIRE (<N>)   exit <0|6>
    declared relation : <source -> render | N/A>
    adjudication      : <confirmed defect | procedure defect | literal false positive | N/A>
  consistency leaks   : N/A | clean | TRIPWIRE (<N>)   exit <0|6>
    declared relation : <customer/public-facing files | N/A>
    adjudication      : <confirmed defect | procedure defect | literal false positive | N/A>
  consistency numbers : N/A | clean | TRIPWIRE (<N>)   exit <0|6>
    declared relation : <summary -> body | N/A>
    adjudication      : <confirmed defect | procedure defect | literal false positive | N/A>

downstream consequence if wrong (ADVISORY — attention, not a score input):
  <the load-bearing claim, what it costs downstream if it is wrong, and that it
   got the strictest reading; this never rewrites the external verdict>

agent reasoning (SUPPLEMENT — does not override the gate):
  <cross-file / arithmetic / policy notes the single-file gate cannot see>
  <each claim about a file/number is backed by a direct check, not plausibility>

candidate verdict: PASS | FAIL | HALT | UNKNOWN
workflow disposition: SHIP | REVISE | DELIVER_WITH_WARNINGS | ESCALATE | HALT
  <one line preserving the external verdict and explaining the disposition>
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
- **Consistency exit 6 is a literal tripwire, not an automatic verdict.** Run a
  check only for its declared relationship and inspect every firing. A confirmed
  defect supports FAIL/REVISE; an inapplicable comparison or documented literal
  false positive is a validator procedure defect. Confirmed protected-data or
  credential exposure is Rights HALT.
- **Gate unreachable / error → verdict UNKNOWN**, never PASS.
- **Keep verdict separate from disposition.** A completed non-Rights FAIL, or
  PASS with actionable warnings, normally maps to REVISE for one bounded producer
  correction. The validator never edits.
- **The `downstream consequence if wrong` line is ADVISORY.** It records where the
  strictest reading went and what being wrong would cost; it changes attention
  and ordering, never the external verdict. A high consequence does not turn a
  gate PASS into a FAIL, and a low one does not rescue a FAIL.

---

## 2. REVISE receipt (line-located finding + bounded producer correction)

Use this when a completed non-Rights FAIL, or PASS with an actionable warning,
has bounded findings. Locate and quote the offense, tie it to a rule, and return
it to a separate producer. The changed candidate must receive a new SHA-256 and
a full validation; a section or partial re-check does not inherit this receipt.

```
## VERDICT v1 — <artifact path>
verdict: PASS | FAIL
disposition: REVISE — bounded producer correction permitted once
candidate_sha256: <sha256>
domain: <domain>    artifact_type: <type>    customer-facing: yes | no
gate: lens-kit validate (profile: <yaml>)    exit: <1>    blocking lens: <lens>

offense:
  file:line   <path>:<line>
  quoted      "<the exact offending text>"
  rule        <the forward rule this breaks — from a prior catch or a lens>
  correction  <bounded guidance; the validator must not write the replacement>

required revalidation:
  scope       <freeze and fully validate the changed candidate; regression-check
              the bounded finding as additional evidence>
  command     lens-kit validate <file> --profile <yaml>
              lens-kit consistency leaks <files...> --profile <yaml>
  hash        <compute a new candidate SHA-256 before dispatch>
```

Discipline: **one bounded producer correction per verdict cycle.** After v2,
unresolved internal low-risk work may be DELIVER_WITH_WARNINGS with all findings;
unresolved public/high-risk work is ESCALATE and does not ship. Rights is HALT.

---

## 3. Paired v1 + v2 verdicts (both preserved)

When a producer correction is made and fully revalidated, the second verdict is **appended**,
not substituted. Both receipts live in the record so the correction cycle is
auditable. v2 records what changed, shows the bounded regression check, and
contains a new full external verdict for the changed candidate.

```
## VERDICT v2 — <artifact path>   (new candidate; v1 kept above)
verdict: PASS | FAIL | HALT | UNKNOWN
disposition: SHIP | DELIVER_WITH_WARNINGS | ESCALATE | HALT
candidate_sha256: <new sha256>
gate: lens-kit validate (profile: <yaml>)    exit: <0|1>

changed since v1:
  <path>:<line>   "<old text>"  ->  "<new text>"

regression check:
  <e.g. "grep '<forbidden phrase>' across <N> rendered files -> 0 hits">
  <e.g. "lens-kit consistency leaks <files> --profile <yaml> -> exit 0 (clean)">

measured post-fix:
  verdict     <PASS | FAIL | HALT | UNKNOWN — as returned this round>
  per_lens    <all rows returned by the full v2 validation>
  violations  <count by severity, from the receipt — not estimated>

candidate verdict: PASS | FAIL | HALT | UNKNOWN
workflow disposition: SHIP | DELIVER_WITH_WARNINGS | ESCALATE | HALT
```

Conventions:

- **Never overwrite a verdict.** v1 and its REVISE disposition stay in the record above v2.
  A reader must be able to see the cycle, not just the endpoint.
- **Full validation is required.** The correction may be bounded, but v2 runs
  the complete gate over the exact changed candidate; clean v1 lenses are not
  assumed to carry over.
- **The v2 verdict is measured.** Do not invent a numeric score: the gate returns
  a verdict, a per-lens map, and violations. If the runtime emits no numeric
  score, report `score: unavailable`.
- **If v2 does not clear, use risk-sensitive disposition.** Internal reversible
  low-risk work may be DELIVER_WITH_WARNINGS; public, customer-facing,
  high-stakes, security-sensitive, or irreversible-use work is ESCALATE and does
  not ship. Rights is always HALT.
