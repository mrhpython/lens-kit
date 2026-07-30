# Validator agent — acceptance walkthrough

This is the acceptance test for the validator agent protocol (success criterion
3): a fresh-context walkthrough of the 7-step loop in
[`VALIDATOR-AGENT.md`](VALIDATOR-AGENT.md), run against a small test artifact
using ONLY the commands as documented — no kit-internals knowledge. Every command
is shown with its actual exit code. Paths are scrubbed to `<work>` (the working
directory) and `<profile>` (the shipped `profiles/agency-example.yaml`).

The loop was run in a clean working directory so the memory and ledger it
produced are the walkthrough's own.

---

## The test artifact

A small customer-facing landing copy, plus the source brief it was rendered from:

`<work>/source.md`
```
# Source brief

Headline claim: 40% faster review cycle [ESTIMATE].
Secondary: sign-off in under 3 days [UNVERIFIED].
```

`<work>/landing.md`
```
# Acme Reviews — faster client sign-off

Teams using Acme cut their review cycle by 40% [ESTIMATE].
Our pilot customers report sign-off in under 3 days.
```

The render kept the `[ESTIMATE]` marker but dropped `[UNVERIFIED]` from the
"under 3 days" claim — a planted cross-format marker drop, exactly the class the
agent layer exists to catch.

## Step 1 — classify

| field | value |
|---|---|
| domain | `marketing` |
| artifact_type | `landing-copy` |
| customer-facing? | **yes** (must pass the leak scan) |

## Step 2 — read prior catches

Seed the memory with the shipped genericized examples, then surface the catches
for this artifact type:

```
$ lens-kit catches add --seed
Seeded 6 example catch(es) into catches.jsonl (documented examples — replace
with your own as you record real catches)
exit 0

$ lens-kit catches relevant landing-copy --domain marketing
## PRIOR CATCHES (institutional memory)
scope: artifact_type=landing-copy, domain=marketing; 1 catch(es)
Read these before validating. Most-recurrent patterns first. A pattern marked
PROMOTE has recurred enough to deserve a deterministic check.

- CATCH: The Northwind Analytics landing page footer carried an internal scoring
  phrase ('validator score 9 of 9') in customer-facing copy, inherited verbatim
  from the source brief.
  PATTERN: renders faithfully inherit internal vocabulary from the source brief
  and carry it into customer-facing output
  RULE: run the deny-list scrub on the source brief, not only on the final
  render; one internal-vocabulary hit is a defect no quality score overrides
exit 0
```

The memory walked us in knowing this artifact class has leaked source-brief
vocabulary before — so the leak scan in step 4 is not optional.

## Step 3 — run the gate (one live call)

```
$ lens-kit validate <work>/landing.md --profile <profile> --domain marketing \
    --context "agency owners evaluating AI review tools" --json
{
  "passed": false,
  "halted": false,
  "halt_reason": "",
  "violations": [
    {"lens": "truth", "severity": "high",
     "issue": "\"under 3 days\" has no named source or [ESTIMATE] marker"},
    {"lens": "causality", "severity": "high",
     "issue": "cause-effect claim with no mechanism, chain, or evidence"},
    {"lens": "relevance", "severity": "warning",
     "issue": "Uses placeholder [ESTIMATE] instead of verified data; agency
      owners need concrete, sourced metrics to make decisions."},
    {"lens": "relevance", "severity": "warning",
     "issue": "Uses placeholder for source; readers cannot trust or verify the
      claim."}
  ],
  "consciousness_flags": [],
  "per_lens": {
    "rights": true, "truth": false, "causality": false, "contradiction": true,
    "extrapolation": true, "structure": true, "consistency": true,
    "relevance": false
  }
}
exit 1
```

The gate is the scorer of record: it FAILED the artifact (exit 1), catching the
unsourced "under 3 days" claim under Truth and the unsupported cause-effect under
Causality — the exact defect the dropped marker laundered.

## Step 4 — cross-checks (deterministic, no LLM)

```
$ lens-kit consistency markers <work>/source.md <work>/landing.md --profile <profile>
[consistency:markers] VIOLATION — source markers (source.md): [UNVERIFIED]=1,
  [ESTIMATE]=1, [PROJECTION]=0; 1 rendered file(s) checked
  landing.md: [UNVERIFIED] count 0 < source 1 (markers dropped in render, OR
  this output covers a subset of the source — review)
Ledger row appended: RUNS.md (consistency-<id>)
exit 6

$ lens-kit consistency leaks <work>/landing.md --profile <profile>
[consistency:leaks] clean — 8 deny term(s) x 1 file(s) scanned
Ledger row appended: RUNS.md (consistency-<id>)
exit 0
```

The markers tripwire fired (exit 6): the render carries zero `[UNVERIFIED]` where
the source carried one. The agent adjudicates — here it is a true defect (the
dropped marker is exactly what the gate flagged under Truth), not the legitimate
subset case the tripwire warns about. The leak scan is clean (exit 0): no internal
vocabulary in the customer-facing copy.

## Step 5 — name the downstream consequence (advisory)

Before rendering, name what happens if the load-bearing claim is wrong:

- **Load-bearing claim:** "sign-off in under 3 days" — a concrete speed promise
  a buyer will act on (it is the page's reason to convert).
- **Consequence if wrong:** an agency owner signs a client SLA against a 3-day
  turnaround the product cannot guarantee, then misses it on real work — a
  contractual, money-adjacent failure, not a cosmetic one.
- **Strictest reading reached it:** yes — this exact claim is what the gate
  failed under Truth (no source / no `[ESTIMATE]`) AND what the markers tripwire
  caught as a dropped `[UNVERIFIED]`. The highest-consequence claim got the
  strictest reading; nothing changed the verdict arithmetic.

This step changed nothing about the score — the gate FAIL stands on its own
arithmetic. It confirmed the scrutiny landed where being wrong costs the most.

## Step 6 — verdict receipt

```
## VERDICT — <work>/landing.md
verdict: FAIL
domain: marketing    artifact_type: landing-copy    customer-facing: yes
gate: lens-kit validate (profile: agency-example.yaml)    exit: 1

| lens          | gate    | note |
|---------------|---------|------|
| rights        | PASS    |      |
| truth         | FAIL    | "under 3 days" has no source/[ESTIMATE] |
| causality     | FAIL    | cause-effect, no mechanism |
| definitionalIntegrity | PASS    | undefined-term / equivocation; blocking |
| contradiction | PASS    |      |
| extrapolation | PASS    |      |
| structure     | PASS    |      |
| consistency   | PASS    |      |
| relevance     | FAIL    | warning-only; placeholder, not sourced data |
| consciousScan | (flags) | 0 flag(s) |

violations (gate): 4 (2 high, 2 warning)

cross-checks (deterministic, no LLM):
  consistency markers : VIOLATION (1)   exit 6
  consistency leaks   : clean           exit 0

downstream consequence if wrong (ADVISORY — attention, not a score input):
  Load-bearing claim "sign-off in under 3 days" — if wrong, a buyer signs a
  client SLA against a turnaround the product cannot guarantee (money-adjacent,
  contractual). It got the strictest reading: it is the exact claim the gate
  failed under Truth and the markers tripwire caught. No effect on the score.

agent reasoning (SUPPLEMENT — does not override the gate):
  The markers tripwire and the gate's Truth FAIL are the SAME defect seen two
  ways: source.md marked "under 3 days" [UNVERIFIED]; landing.md dropped the
  marker and the claim became a flat customer-facing assertion. Verified by
  reading both files directly, not by inference.

verdict of record: FAIL
  Gate exit 1 (Truth + Causality high). Consistency markers exit 6 independently
  forces FAIL — a dropped marker is a violation no score overrides.
```

## Step 7 — append the new catch

```
$ lens-kit catches add \
    --catch "landing.md asserts 'sign-off in under 3 days' as a flat
      customer-facing claim with no named source or [ESTIMATE] marker; the
      source brief marked the same claim [UNVERIFIED] but the marker was dropped
      in the render" \
    --pattern "evidence markers present in the source brief get dropped when the
      claim is rendered into customer-facing copy, laundering an unverified claim
      into a flat assertion" \
    --rule "diff marker counts source vs every render before shipping; a
      customer-facing claim that lost its source marker is a Truth defect, not a
      style nit" \
    --artifact-type landing-copy --domain marketing
Appended catch <id> to catches.jsonl
exit 0
```

The memory now carries both catches for this artifact type:

```
$ lens-kit catches relevant landing-copy --domain marketing
## PRIOR CATCHES (institutional memory)
scope: artifact_type=landing-copy, domain=marketing; 2 catch(es)
...
- CATCH: The Northwind Analytics landing page footer carried an internal scoring
  phrase ... (the seed catch)
- CATCH: landing.md asserts 'sign-off in under 3 days' as a flat customer-facing
  claim ... (the catch just recorded)
exit 0
```

And the doctrine boundary holds — a routine-pass catch is rejected:

```
$ lens-kit catches add --catch "all lenses passed, no issues found" \
    --artifact-type landing-copy --domain marketing
error: catch must NAME A DEFECT — empty or pass-like entries are rejected by
doctrine (routine passes are excluded to keep the memory high-signal). Record
what was WRONG, or do not record.
exit 2
```

---

## Result

Every command in the documented loop ran exactly as written, with no
kit-internals knowledge:

| step | command | exit |
|---|---|---|
| 2 | `catches add --seed` | 0 |
| 2 | `catches relevant landing-copy --domain marketing` | 0 |
| 3 | `validate landing.md --profile <profile> --domain marketing --context "..." --json` | 1 (gate FAIL — a real verdict) |
| 4 | `consistency markers source.md landing.md --profile <profile>` | 6 (tripwire fired) |
| 4 | `consistency leaks landing.md --profile <profile>` | 0 (clean) |
| 5 | (consequence-naming — advisory, no command) | — |
| 7 | `catches add --catch ... --pattern ... --rule ... --artifact-type ... --domain ...` | 0 |
| (boundary) | `catches add --catch "all lenses passed ..."` | 2 (routine pass rejected) |

The loop closes: prior catches surfaced before the run, the gate scored,
deterministic cross-checks caught the cross-format defect, the downstream
consequence was named (advisory), a verdict receipt was rendered, and the new
catch was banked for the next run.
