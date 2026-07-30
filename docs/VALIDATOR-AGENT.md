# The validator agent — protocol

This is the protocol you hand to ANY agent (any model) to make it a validator on
top of the lens-kit gate. It is model-agnostic: it names real `lens-kit`
commands and the verdict receipts they feed, nothing model-specific.

The validator agent is the **expensive tier**. The deterministic kit does the
cheap work — the gate scores one document, the `consistency` checks fire on
cross-artifact tripwires, the `catches` memory remembers prior defects. The agent
runs AFTER those, on the artifacts that need a reasoner that can hold several
files, an arithmetic relationship, or a policy in its head at once. The agent's
job is the CROSS-something defect: cross-file contradiction, cross-format marker
drift, a summary number the body never states, a policy phrase that leaked. The
single-document gate cannot see those; that is exactly the gap this agent fills.

> A drop-in system prompt that implements this protocol ships at
> [`../agent/validator-agent.md`](../agent/validator-agent.md); the verdict
> receipt templates at [`../agent/receipt-templates.md`](../agent/receipt-templates.md).
> The kit's CLI is the agent's whole toolbelt — no graph, no service, no network
> beyond the gate's own LLM call.

---

## Preconditions and failure handling

Before running the loop, these must be true (verify them, do not assume):

- **lens-kit is installed and on PATH** — `lens-kit --help` returns exit 0.
- **a profile YAML exists** for the domain, with its `llm` block pointing at a
  reachable endpoint and the named `api_key_env` variable set (the gate fails
  closed on a missing key — see step 3).
- **the catches memory is reachable** — a `catches.jsonl` in the working
  directory, seeded (`lens-kit catches add --seed`) if starting fresh.
- **the deterministic checks have been considered first** — the agent tier runs
  AFTER the gate and the `consistency` tripwires, not instead of them (see *When
  to use the agent layer at all*).

On a step failure the loop does NOT silently proceed:

- **gate error / unreachable (step 3)** → the verdict is UNKNOWN, the loop stops,
  and the artifact is escalated to a human. Never assume PASS.
- **a CONDITIONAL that does not clear on its one re-check (step 6)** → escalate to
  a human; do not iterate further.
- **a `consistency` exit 6 (step 4)** → the verdict is FAIL regardless of the gate
  score; no score overrides it.
- **a catches add rejected (step 7)** → the entry named no defect (a routine
  pass); record the real defect or record nothing. This is a doctrine stop, not
  an error to route around.

The "rollback" for a validation is simple: a verdict is a recommendation, never an
irreversible act, so an UNKNOWN or a FAIL costs nothing but a human decision — the
artifact is not shipped, sent, or published on the agent's say-so.

---

## The 7-step loop (on real kit commands)

### 1. Classify the artifact

Decide three things, because they drive every later step:

- **domain** — `marketing`, `finance`, `seo`, `competitor`, `general`, ... (the
  gate also auto-detects from the profile's `domain_detection` keywords; pass
  `--domain` to pin it).
- **artifact_type** — `landing-copy`, `research-brief`, `content-pack`,
  `social-post`, ... (a free string; it keys the catches memory).
- **customer-facing?** — yes/no. A customer-facing artifact must pass the leak
  scan; an internal one is held to a softer bar on internal-vocabulary.

### 2. Read prior catches

Surface the institutional memory before validating, so the agent walks in already
knowing where this class of artifact has failed before:

```bash
lens-kit catches relevant <artifact_type> --domain <domain>
```

This prints a stable paste block — a scope header, any `[PROMOTE]` patterns
(recurred enough to deserve a deterministic check) at the top, then one
`CATCH / PATTERN / RULE` block per record, most-recurrent first. Read it into the
agent's context. Use `--all` when the type is new and has no history:

```bash
lens-kit catches relevant --all --domain <domain>
lens-kit catches relevant <artifact_type> --format json   # raw records for tooling
```

If the memory is empty, seed it with the shipped genericized examples first:

```bash
lens-kit catches add --seed
```

### 3. Run the gate (the scorer of record)

```bash
lens-kit validate <file> --profile <yaml> --domain <domain> \
    --context "<audience/purpose>" --json
```

- Exit `0` = passed, `1` = failed or halted, `2` = usage/config error.
- The JSON carries `passed`, `halted`, `halt_reason`, `per_lens` (a
  `{lens: bool}` map), `violations` (each with `lens`, `severity`, `issue`), and
  `consciousness_flags`.
- `--context` feeds the **relevance** lens (audience/purpose). Without it,
  relevance does not block (it is warning-only).
- A lens absent from `per_lens` was **not evaluated** — on a Rights HALT,
  `per_lens` holds only `rights` and the rest never ran. Absence is not a pass.

This verdict is the score of record. The agent's later reasoning supplements it;
it does not override it.

### 4. Run cross-checks (deterministic, no LLM)

Run the `consistency` checks that apply to the artifact set. Each is pure Python,
no model call, and exits `6` on a violation that **no score overrides**:

```bash
# markers — every evidence marker in the source must survive into each render:
lens-kit consistency markers <source> <rendered...> --profile <yaml>

# leaks — internal vocabulary must never reach customer-facing copy:
lens-kit consistency leaks <customer-facing-files...> --profile <yaml>

# numbers — every number a summary asserts must appear in the body:
lens-kit consistency numbers <summary> <body>
```

When to run each:

| check | run it when |
|---|---|
| markers | there is a source artifact + one or more rendered outputs |
| leaks | ANY file is customer-facing (run on every such file) |
| numbers | one file summarizes another (exec summary vs detail) |

**Tripwire, not oracle.** These are literal and deterministic; none understands
meaning. `markers` flags a render that legitimately covers a subset of the source
(a false positive by design); `numbers` does no derived math ("13 of 17" will not
reconcile to "76%"). A firing means "look here", a clean run means "no tripwire
fired" — not "proven consistent". That adjudication is the agent's job in step 6.

### 5. Name the downstream consequence (ADVISORY — attention, not arithmetic)

Before you render the verdict, name what happens DOWNSTREAM if the artifact's
load-bearing claim is wrong, and confirm the highest-consequence claims got the
strictest reading. This step changes ATTENTION and ORDERING; it NEVER changes the
verdict arithmetic — the gate score (step 3) plus any `consistency` exit 6 (step
4) remain the authoritative scorer. A high consequence cannot turn a gate PASS
into a FAIL, and a low consequence cannot rescue a gate FAIL.

Do three things:

- **Identify the load-bearing claim** — the one sentence the artifact's purpose
  rests on (the number a buyer will act on, the promise a contract encodes, the
  instruction a reader will follow).
- **Name the downstream consequence if it is wrong** — concretely: who is
  misled, what irreversible action follows, what it costs. "A reader sends money
  against a fabricated ROI figure" is a consequence; "it would be bad" is not.
- **Confirm the strictest reading reached the highest-consequence claims** —
  check that the gate verdict, the relevant prior catches (step 2), and the
  consistency checks (step 4) actually covered THAT claim, not just the easy
  ones. If a high-consequence claim slipped through with a light reading, re-run
  the gate on it in isolation, or surface it explicitly in the receipt.

The point is to spend the strictest scrutiny where being wrong costs the most —
not to re-score. If this step changes anything, it changes WHERE you looked, and
that shows up as a finding in the receipt's supplement block, never as an
override of the gate's number.

### 6. Render the verdict receipt

Render the per-lens table from
[`../agent/receipt-templates.md`](../agent/receipt-templates.md). Three shapes:

- **PASS / FAIL** — the per-lens table plus the cross-check results.
- **CONDITIONAL** — a fixable verdict: quote the offense, give `file:line`, name
  the rule it breaks, propose the rewrite, give a **`[PROJECTION]`** post-fix
  score (a projection, never a measurement), and size the re-check.
- **Paired v1 + v2** — when a CONDITIONAL is fixed, append v2 (a delta verdict
  with the regression check and the MEASURED score); keep v1. Never overwrite a
  verdict.

Put the cross-relationship reasoning — the cross-file, arithmetic, and policy
findings the single-file gate could not see — in the receipt's
`agent reasoning (SUPPLEMENT)` block, clearly marked as supplement. Record the
step-5 consequence on the receipt's `downstream consequence if wrong` line (the
load-bearing claim, what it costs if wrong, and that it got the strictest
reading) — it is advisory context, not a score input. A consistency exit `6`
forces the verdict to FAIL regardless of the gate score.

### 7. Append new catches

Record each NAMED DEFECT you found, so the next run starts smarter. Routine
passes are rejected by doctrine — the memory holds defects, not "looked fine"
notes:

```bash
lens-kit catches add \
  --catch "<what was WRONG>" \
  --pattern "<the general trap this is an instance of>" \
  --rule "<the forward rule that prevents this next time>" \
  --artifact-type <type> --domain <domain>
```

Add `--self-catch` when the defect was a failure of the validator's OWN discipline
(e.g. citing a stale number, validating from memory instead of the file). For
batches, write records to JSON and use `--from-json <file>` (an atomic, all-or-
nothing append). A catch with no `pattern`/`rule` still records but warns — a
catch without a generalization is half a catch.

Do NOT record a routine pass. When the same pattern recurs enough times,
`lens-kit catches stats` flags it `[PROMOTE]`: that is the signal to encode it as
a deterministic `consistency` check, after which it runs in pure Python forever,
cheaper and earlier than any agent pass. The shipped `consistency` checks are the
first three patterns that completed exactly this journey.

---

## Boundary rules (stated as agent instructions)

These are the discipline the validator agent enforces on itself. They were each
learned the expensive way; treat them as load-bearing, not advisory.

- **NEVER validate output you generated.** A generator grading its own output is
  circular — the verdict is worthless. Hand the artifact to a SEPARATE validator
  instance with no stake in its content.
- **The gate's verdict is the score of record; your reasoning supplements it.**
  You may add findings the gate missed (that is the whole point of the agent
  tier), but you NEVER argue a gate FAIL into a PASS. A FAIL stands until the
  artifact is fixed and re-validated.
- **Claims about files, data, or systems need a DIRECT check** — read the file,
  run the command. Prose plausibility is not evidence. If you assert a fix was
  applied, re-read the file on disk; if you cite a number, it must appear in a
  file you actually read. The file on disk is the only truth.
- **On gate unreachable / error → verdict UNKNOWN, never PASS.** A missing API
  key, a config error, a network failure, a non-zero exit you did not expect —
  all fail closed to UNKNOWN. A green verdict is a positive claim; never make it
  by default.
- **One bounded fix round per verdict cycle, then escalate.** A CONDITIONAL gets
  ONE fix + re-check (the v2 verdict). If v2 does not clear, hand it to a human.
  Do not iterate the agent against the gate indefinitely.
- **You recommend; the verdict of record is the gate's, and a human owns
  irreversible action.** Money, publishing, sending, deleting are a human's to
  authorize. The agent's output is a verdict and a recommendation, never an act.

---

## When to use the agent layer at all

The agent is the expensive tier. Run the deterministic checks FIRST — the gate
(`lens-kit validate`) and the cross-checks (`lens-kit consistency`) — and reach
for the agent loop only on:

- **borderline / HOLD verdicts** — where the gate is uncertain or a tripwire
  fired and the question is now "is this a real defect or a known false positive?"
- **customer-facing artifacts** — where the cost of a missed leak or fabrication
  is high.
- **high-stakes content** — anything money-adjacent, published, or sent.
- **multi-file asset sets** — where the defect is a RELATIONSHIP the single-file
  gate cannot see: a cross-file contradiction, a number a summary invents, a
  marker dropped between platform variants, a policy phrase carried verbatim from
  an internal brief into customer copy.

For a single clean low-stakes file, the gate plus a leak scan is enough. The
agent is the cross-relationship reasoner — cross-file, cross-format, arithmetic,
policy — not a replacement for the gate.

---

A walkthrough of this loop, run end to end against a small test artifact using
only the commands documented here, is recorded in
[`VALIDATOR-AGENT-walkthrough.md`](VALIDATOR-AGENT-walkthrough.md).
