# lens-kit (C5 — gate + compile/eval + calibration + mutation/audit/sidecar + review)

Trainable 10-lens validation gate for AI-generated text, as an installable
Python package. Increment **C1** extracted the gate program with all
provider/domain specifics externalized to config; increment **C2** added the
GEPA compile harness (costing gate + pace monitor) and the holdout eval
harness (variance envelopes + versioned receipts); increment **C3** added the
planted-flaw calibration battery (`lens-kit calibrate generate` / `run`);
increment **C4** adds mutation control (`lens-kit mutate`), the gold-label
audit workflow (`lens-kit label-audit`), and the full provenance-sidecar
generator (`lens-kit sidecar`, also written automatically by every compile);
increment **C5** adds the human review surface (`lens-kit review`) — a
self-contained HTML viewer for eval / calibration / mutation artifacts with
auto-saved feedback.json (adapted from Anthropic's eval-viewer, Apache-2.0;
see the attribution block below).

## Documentation

The methodology manual ships inside the kit under [`docs/`](docs/):

| Doc | What it covers |
|---|---|
| [`docs/MANUAL.md`](docs/MANUAL.md) | The training loop end to end — label → baseline → calibrate → compile (costing gate) → eval + variance → FN forensics → one bounded fix → mutation control → sidecar → ship. Each step: why, the exact command, the receipt, the failure it prevents. |
| [`docs/CLAIMS.md`](docs/CLAIMS.md) | The honest-claims doctrine: evidence lanes, the five-field claim-promotion gate, `not_a_claim_of`, era-stamping + re-baseline on provider drift, and the score-transfer ban (numbers do not transfer — measure yours). |
| [`docs/WORKED-EXAMPLE.md`](docs/WORKED-EXAMPLE.md) | A real run retold as a tutorial: a version bump that *looked* like it broke the gate, the cache-collision proof that it didn't, the served-model substitution that did, and the two wrong conclusions the discipline blocked. |
| [`docs/VALIDATOR-AGENT.md`](docs/VALIDATOR-AGENT.md) | The validator-agent protocol — the model-agnostic 7-step loop (classify → read prior catches → run the gate → run cross-checks → name the downstream consequence → render the verdict receipt → append catches) on real kit commands, the boundary rules as agent instructions, and when to spend the agent tier at all. Drop-in system prompt: [`agent/validator-agent.md`](agent/validator-agent.md); receipt templates: [`agent/receipt-templates.md`](agent/receipt-templates.md). |

## Provider-agnostic by design

The customer brings ANY litellm-compatible endpoint — xAI, OpenAI,
Anthropic, Google, a self-hosted OpenAI-compatible server, or local Ollama.
A profile's `llm.model` is a litellm model string passed straight to
`dspy.LM`; `api_base` covers local/self-hosted endpoints; `api_key_env`
names the environment variable holding the key (keys are never stored in
profiles). There is no fallback chain: a missing key or model is a hard
error, never a silent provider switch. Gate base-model tuning is the
customer's loop, not a ship blocker — the in-box argument is the Stage-A
substitution episode, where a provider swapped the served model mid-quarter
under an unchanged model id.

## Usage

```bash
pip install -e .

# 1. Try it with NO credentials and NO network. These run the deterministic
#    parts of the kit and are the fastest way to see it is real:
lens-kit calibrate generate          # writes a 16-fixture calibration battery
lens-kit catches add --seed          # seeds the institutional-memory loop
lens-kit consistency markers         # deterministic marker/leak scan

# 2. Then run the actual gate. It needs ONE model endpoint — any litellm
#    target. Copy the example profile and point `llm.model` at yours:
cp src/lens_kit/profiles/agency-example.yaml my-profile.yaml
echo 'Our Q3 revenue grew 40% year-over-year and will keep growing.' > report.md
lens-kit validate report.md --profile my-profile.yaml --json
```

The gate calls a model, so step 2 costs whatever your endpoint charges — set
`llm.model` to `ollama_chat/…` with `api_key_env:` omitted and it costs nothing
and stays on your machine. There is no bundled key and no default vendor: if the
env var named in your profile is unset, the kit refuses to run rather than
silently reaching for something else.

```python
from lens_kit import LensGate, Profile, lm_context

profile = Profile.load("my-profile.yaml")
gate = LensGate(profile=profile)
with lm_context(profile):   # scoped LM override, restores on exit/exception
    result = gate(text=text, context="CFO audience, needs cost numbers")
result.passed, result.per_lens, result.violations
```

Process-global alternative: `lm, previous_lm = configure_from_profile(profile)`
— restore with `dspy.configure(lm=previous_lm)`. Prefer `lm_context` for
anything scoped (the compile and eval harnesses run inside it).

## Training and evaluating (C2)

```bash
# Compile the gate against your labeled data (GEPA):
lens-kit compile training.json --profile my-profile.yaml --output ckpt.json \
    --auto light --threads 4            # or --max-evals N (mutually exclusive)
# Pace probe runs against a FRESH cache by default (a warm cache replays
# sub-second pace and would arm a false pace kill); --fresh-cache also points
# the whole compile run at a fresh cache; --warm-probe-cache opts the probe out.

# Evaluate on a frozen holdout (NEVER train on it):
lens-kit eval holdout.json --profile my-profile.yaml --checkpoint ckpt.json
lens-kit eval holdout.json --profile my-profile.yaml --baseline   # uncompiled
lens-kit eval holdout.json --profile my-profile.yaml --checkpoint ckpt.json \
    --reruns 3 --fresh-cache            # real variance envelope
```

Training data is a plain JSON list of `{text, per_lens, domain?, source?}`
(items with text under 20 chars or an empty `per_lens` are skipped); holdout
files may be the same shape or `{"_metadata": ..., "examples": [...]}`.
Lens labels are validated at load time: any `per_lens` key outside the 10
canonical lenses is a **hard error**, not a warning — a typo'd label can
never match gate output, so the example would silently score 0 and poison
metrics without a trace (and a warning would scroll past in a
4,000-rollout compile). Fixing a label forces a new run designation.
Eval writes self-describing results JSON to **versioned filenames**
(`eval-<date>-runK.json`) — results are never overwritten. `--reruns N`
additionally banks a variance envelope: per-metric min/max/mean/pstdev plus
a direction-aware worst-of-N floor (accuracy/catch floor = MIN observed,
fp_rate floor = MAX observed). Checkpoints are saved with a `.sha256`
sibling and a full `.provenance.json` sidecar (model + provider note,
dspy/gepa/litellm versions, dataset block, `not_a_claim_of` list — see the
provenance-sidecar section below). Exit codes: 3 = costing-gate hard stop,
4 = pace kill.

## Calibrating the gate (C3)

```bash
# Generate planted-flaw fixtures from your profile's calibration vocabulary
# (no LLM calls, deterministic — same seed + profile = byte-identical output):
lens-kit calibrate generate --profile my-profile.yaml --out battery/ --per-class 3

# Run the battery through the gate and score each verdict against pre-agreed RULES:
lens-kit calibrate run battery/ --profile my-profile.yaml
lens-kit calibrate run battery/ --profile my-profile.yaml --reruns 3 --fresh-cache
```

**Why planted-flaw calibration comes before trusting any gate.** A gate's
PASS is only meaningful once you have proven it can FAIL the right things and
clear clean content. A gate that blocks everything is not safe, it is broken —
it teaches you nothing about where your tuning helped and makes the product
read as a rigged scare tool. So before relying on a verdict you run a battery of
fixtures with KNOWN planted flaws (and known-clean controls), each carrying a
pre-agreed acceptance rule: clean content must not be blocked, an internal
contradiction must HOLD, a dark pattern must HOLD, an uncited statistic must at
least demand sources. This is mutation control aimed at your own gate: re-run it
after every tuning change, and a clean fixture that starts blocking — or a
planted flaw that starts passing — is a regression you see before a customer
does. The battery is **template-generated and deterministic**, parameterized by
the `calibration:` vocabulary in your profile (company / product / audience /
price / statistic slots); the planted flaws themselves are structural and
identical regardless of slot values. Each fixture ships with a sidecar
`*.gold.json` recording its class, expected per-lens labels, acceptable/target
verdicts, and a sha256 of the fixture bytes.

**Verdict mapping (a design decision you should know about).** The gate returns
a structured `passed` / `per_lens` / `violations` / `halted` result, not a
SHIP/HOLD/NEEDS_SOURCES ladder. The battery's RULES are written against a
3-verdict ladder, so the runner maps the gate output deterministically:

| Gate signal | Calibration verdict |
|---|---|
| `halted` (Rights HALT) | `HOLD` |
| any critical/high-severity violation | `HOLD` |
| `passed=False`, sole non-relevance blocker is `truth`, and a failing-truth violation names a missing-source pattern | `NEEDS_SOURCES` |
| `passed=False`, any other blocking lens | `HOLD` |
| `passed=True` | `SHIP` |

Relevance is warning-only (never blocks `passed`), and `consciousScan` only
annotates — neither can move a verdict on its own. `NEEDS_SOURCES` is the
"needs a citation, not a block" verdict: it fires only when Truth is the lone
blocker and its violation reads as an uncited-claim complaint, not a
fabricated-fact one. Known fragility, by design: that "reads as" test is a
substring match against the LLM's own violation wording ("source", "cite",
"uncited", ...), so a truth-only block phrased without a source word maps to
`HOLD` rather than `NEEDS_SOURCES`. The mapping degrades SAFE: for the
unsourced class both verdicts are accepted catches, so a marker miss can only
lower `on_target_rate`, never flip a fixture from pass to fail or let an
uncited claim ship. Per-class RULES: clean accepts `{SHIP, NEEDS_SOURCES}`
(target `SHIP`); unsourced accepts `{NEEDS_SOURCES, HOLD}` (target
`NEEDS_SOURCES`); contradiction and darkpattern require `HOLD`. Exit codes:
`0` = every fixture acceptable on every rerun, `2` = config/usage error
(missing dir, missing gold sidecar — fail closed), `5` = battery failures
present (an unacceptable verdict or an UNKNOWN — `5`, not `3`, so it never
collides with the compile costing-gate stop). Results land in versioned
`calib-<date>-runK.json` files (never overwritten); `--reruns N` additionally
banks a variance envelope with the same direction-aware worst-of-N floors as
eval (`acceptable_rate`/`on_target_rate` floor at MIN, `fp_rate`/`fn_rate`/
`unknown_count` floor at MAX), and a `RUNS.md` row is appended. `fn_rate`
counts a planted-flaw fixture as a miss when its blocking verdict did not fire
— for unsourced, both HOLD and NEEDS_SOURCES are catches, so only a silent
SHIP is the miss.

## Mutation control — is my gate actually alive? (C4)

```bash
# Seed deterministic mutants from your real holdout, run them through the gate;
# the gate MUST flag every one or the command exits 5 naming the misses:
lens-kit mutate holdout.json --profile my-profile.yaml
lens-kit mutate holdout.json --profile my-profile.yaml --per-type 2 --output-dir out/

# Or fold it into an eval as a post-check (a missed mutant overrides a green eval):
lens-kit eval holdout.json --profile my-profile.yaml --checkpoint ckpt.json --mutation-control
```

**A gate that cannot fail a planted flaw proves nothing.** A green eval and a
PASS verdict tell you the gate *can* clear content; they do not tell you the
gate *can still fail* a real defect after a compile or a model change. The only
way to know is to plant a flaw you KNOW is there and confirm the gate fails it.
That is the seeded-broken-variant pattern: take a real (clean) holdout example,
deterministically inject ONE known defect — a fabricated specific statistic, a
self-contradicting number, a stripped source attribution — and require the gate
to block it. If any planted mutant sails through, the gate is rubber-stamping on
that defect class, `lens-kit mutate` exits **5**, and it NAMES the missed
mutants so the failure is attributable to a specific lens family, not a vague
"the gate feels weak". Mutant construction is pure Python and seeded — same
`(holdout, seed)` yields byte-identical mutants, no LLM in the loop — so this is
a cheap check you run after every compile, before you trust the receipts. The
formalism is borrowed from a verification-pipeline mutation control where a
deliberately mutated artifact's receipts MUST FAIL or the verifier is shown to
be a rubber stamp; here the "verifier" is your own gate.

## Label audit — audit the gold before you blame the model (C4)

```bash
# Build a Confirmed / Borderline / False-Alarms audit scaffold for the examples
# the gate disagreed with, with impact math (how many FNs/FPs could be phantom):
lens-kit label-audit holdout.json eval-run1.json

# Before you EDIT any label, make the sanctioned dated backup:
lens-kit label-audit holdout.json eval-run1.json --backup
```

**When the gate disagrees with a gold label, the LABEL might be wrong.** A
mislabeled example penalizes a correct gate and rewards a broken one; if you
tune against it you are optimizing toward a data bug. So before you accept an
eval's FNs and FPs as the gate's fault, audit the examples where gate and gold
disagree and decide, per example, whether the label or the model is at fault.
`lens-kit label-audit` reads your gold labels plus a single-run eval's
per-example confusion and writes the audit SCAFFOLD: every disagreement seeded
into *Borderline*, both relabel directions (FN: model passed, gold says FAIL;
FP: model failed, gold says PASS) spelled out, and an impact section quantifying
the UPPER BOUND of label-attributable error — how many FNs/FPs would flip if
every suspicious label turned out wrong. (It is an upper bound, not a verdict:
the human decides each row, because relabeling a genuine model error would hide
a real gate weakness.) The discipline is enforced mechanically: the workflow
never edits your dataset, a relabel requires an explicit `--backup` that writes
a dated copy (never overwriting a prior one), and every run prints the
new-run-designation rule — a label edit forces a NEW run id, and you must never
compare metrics across a relabel. A `LABEL_AUDIT` row is appended to `RUNS.md`.

## Provenance sidecar — the number refuses to travel past its evidence (C4)

```bash
# Every compile writes a full sidecar automatically; generate one for any
# existing checkpoint, optionally enriched with the dataset + an eval receipt:
lens-kit sidecar ckpt.json --profile my-profile.yaml \
    --dataset holdout.json --eval-results eval-envelope.json \
    --not-a-claim-of "our specific edge"
```

**A bare accuracy number travels; a sidecar makes it refuse to.** A checkpoint
is a pile of optimized prompts — on its own it cannot tell you which model
produced it, which labeled data it was tuned and scored on, what it scored, or
what it is NOT a claim of. `lens-kit sidecar` writes a `<ckpt>.provenance.json`
recording the checkpoint's sha256 + size, the model string and a provider note,
the dspy/gepa/litellm versions, the dataset file with counts and labels version,
the eval metrics (with the worst-of-N floor when an envelope is supplied), an
empty `re_measurements: []` for future drift records — and, load-bearing, an
explicit `not_a_claim_of` list. That list ships the honest boundary WITH the
artifact: by default it always includes "no transfer to another domain or
dataset", "no accuracy floor on customer data the gate was never tuned on", "no
concurrency/scale claim", and "no per-call bit-reproducibility", plus a
single-run caveat whenever the eval had no variance envelope. The defaults can
be added to but never removed — the whole point is that an
N%-on-our-holdout figure can't be quietly re-dressed as a transfer promise. The generator upgrades
C2's minimal provenance stub and is **backward compatible**: every field the
stub guaranteed is still present with the same meaning; `provenance_stub` simply
flips to `false`.

## Review surface — put a human in front of the verdicts (C5)

```bash
# Serve one results file (or a directory of them) as a review page.
# Default port 0 = an OS-chosen ephemeral port (the URL is printed):
lens-kit review eval-2026-06-12-run1.json
lens-kit review results-dir/                       # eval + calib + mutation together
lens-kit review results-dir/ --port 3117 --open    # fixed port, open a browser

# Write a standalone HTML file instead of serving (shareable, works offline):
lens-kit review results-dir/ --static review.html

# Show the previous iteration's notes as diff context next to each record:
lens-kit review results-dir/ --previous-feedback ../old/feedback.json
```

**A trainable gate needs a human verdict on the gate's verdicts.** Eval,
calibration, and mutation results are JSON — fine for receipts, useless for the
judgement call of "is the gate actually right here, or is the label wrong?".
`lens-kit review` renders those artifacts as a single self-contained page so a
human can page through every record and leave a note that auto-saves to
`feedback.json` — the input the label-audit and re-tuning loops are built to
consume. The page embeds all data inline (no build step, no CDN), and the only
network surface is an optional localhost feedback server.

It detects and renders three artifact types:

| Artifact | Per record | Summary header |
|---|---|---|
| **eval** results | per-lens verdict chips (tp/tn/fp/fn), gold-vs-got score, mismatch strings | accuracy / catch / FP rate, example + error count, variance envelope if present |
| **calibration** battery | verdict vs acceptable / on-target, per-lens confusion vs gold | acceptable / on-target / FP / FN rate, unknown count, envelope if present |
| **mutation** control | planted-defect marker, gate verdict, caught / MISSED, expected-catch set | total mutants, missed count, seed |

Type is detected from each file's `*_schema_version` key (with a structural
fallback), so you can point it at one file or a whole results directory and it
sorts the records out. Feedback is keyed by a stable per-record id
(`eval:<file>#<index>`, `calib:<file>#<fixture>`, `mut:<file>#<id>`); "submit
all" writes a row for EVERY record so "no note" (looked fine) is distinguishable
from "not reviewed". An unparseable or unrecognised input is a hard usage error
(exit 2), never a silent empty page.

Mutation cards render the full **mutated text body** alongside the
planted-defect marker (the C5 increment also extended the mutation writer to
persist `text` in each results row — a backward-compatible artifact addition).
For artifacts written by older kit versions that lack `text`, the card shows
the marker plus an explicit "(mutant text not stored in results)" note rather
than inventing the text.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | success / validation passed / all mutants caught |
| `1` | validation failed or halted |
| `2` | usage or config error (fail closed) |
| `3` | costing-gate hard stop — **compile only**, "this run costs too much" |
| `4` | pace kill — live pace blew the approved projection |
| `5` | the gate failed to catch planted flaws: calibration battery failures (`calibrate run`) OR a missed/rubber-stamped mutant (`mutate`, `eval --mutation-control`) |
| `6` | consistency violation (`consistency`) — a dropped marker, a forbidden-string leak, or an orphan summary number. Deterministic; no score overrides it |

`3` is reserved for the compile costing-gate stop alone so a wrapper script can
tell "too expensive to run" apart from "ran, but the gate is weak" (`5`).

### The costing gate (why compile can refuse to run)

A GEPA run in our own use once projected a spend two orders of magnitude
above what it actually cost, and only stopped there because a human happened to
be watching. (Internal observation, one run, our endpoint and prices — not a
benchmark and not a claim about yours.) The projection is therefore CODE,
not vigilance: before compiling, the harness computes projected rollouts
from GEPA's own budget accounting (`--max-evals N` -> N x (train+val);
`--auto` -> GEPA's auto_budget), measures live seconds-per-rollout on a
small probe chunk, and prints projected wall-clock — plus a dollar figure
only if your profile sets `compile.cost_per_rollout_usd` (default: UNKNOWN,
and the gate says so rather than inventing a number). A projection over the
approval threshold (default 2h; `compile.approval_threshold_hours`) is a
hard stop without an explicit `--approve-cost`. During the run a pace
monitor aborts with a structured kill record (`<ckpt>.kill.json`: % done,
elapsed, projected remaining, reason) if measured pace exceeds the approved
projection by `compile.pace_kill_factor` (default 1.5x).

**The probe is measured against a fresh cache by default.** A warm dspy
disk cache (left by a prior run, an eval, or an earlier compile of the same
data) replays sub-second pace for every probe call; the monitor would then
arm at 1.5x of a fictional sub-second pace and the first real, uncached
rollout would trip a **false pace kill**. (This happened once: probe
0.11s/rollout, real call 37s/rollout, killed at exit 4.) A probe that's
accidentally cached lies about money, so `compile` runs the probe against an
isolated fresh cache dir and restores the live cache afterwards. Pass
`--warm-probe-cache` to opt out (price the cached pace deliberately).
`--fresh-cache` additionally points the *whole GEPA compile run* at a fresh
cache (mirrors `eval --fresh-cache`) so no rollout is served from a stale
cache and the monitor sees real pace throughout.

### Rerun variance and the cache trap

dspy's disk cache replays identical responses for identical prompts. A
variance battery we ran was voided by exactly this: the "reruns" were
cache replays and the measured variance was zero by construction. `lens-kit
eval --reruns N` therefore prints a cache warning, records a
`cache_caveat` in the envelope when reruns shared a cache, and supports
`--fresh-cache` to point the dspy cache at a fresh temp dir per rerun
(memory cache disabled) so the envelope reflects real model variance.

## Consistency checks (deterministic, no LLM)

The LLM gate reads one document at a time. Some defects only show up
*between* artifacts — a marker present in a source that vanishes in a
rendered summary, an internal phrase that leaks into customer-facing copy,
a number a summary asserts that its body never states. `lens-kit
consistency` mechanizes three such cross-artifact checks in pure Python: no
model call, no network, fully deterministic.

```bash
# Marker parity — evidence markers in a source must survive into every render:
lens-kit consistency markers source.md rendered-a.md rendered-b.md
lens-kit consistency markers source.md out.md --markers "[UNVERIFIED]" "[ESTIMATE]"

# Forbidden-string leak scan — deny-list from the profile (+ optional --deny):
lens-kit consistency leaks copy.md email.md --profile my-profile.yaml
lens-kit consistency leaks copy.md --deny "internal cost" "gross margin"

# Number parity — every number in a summary must appear (normalized) in the body:
lens-kit consistency numbers summary.md body.md

# Run several from one config (only the listed checks run):
lens-kit consistency all --config checks.yaml --profile my-profile.yaml
```

**What each check catches**

- **markers** — evidence markers (`[UNVERIFIED]` / `[ESTIMATE]` /
  `[PROJECTION]`, configurable via `consistency.markers` or `--markers`)
  dropped between a source and its rendered outputs. A marker counted in the
  source but missing from a render is how an unverified number gets
  laundered into clean-looking copy (markers dropped in tables and
  platform-specific variants).
- **leaks** — internal vocabulary leaking into customer-facing copy:
  internal lens/scoring names and counts, cost or margin data, any phrase
  you list in `consistency.deny` (case-insensitive, literal — no regex).
  **One hit is a violation no score overrides** — a leak is a defect by
  itself, however good the copy reads.
- **numbers** — numbers a summary asserts that are absent from the body it
  summarizes (the exec-summary-vs-detail mismatch).

**Tripwire, not oracle.** Every check is literal and deterministic — none of
them understands meaning. They fire to send a human or a validator agent to
look, they do not adjudicate. Two false-positive boundaries are by design:

- *markers*: a rendered file that legitimately covers a SUBSET of the source
  will carry fewer markers and be flagged. The check can't know what a render
  "includes", so it flags the under-count for review rather than guessing.
- *numbers*: literal matching after normalization only (thousands
  separators, `%`, currency symbols, `k`/`M`/`bn` suffixes — so `$1.2M`
  matches `$1,200,000`). There is **no semantic math and no derived-value
  resolution**: a summary's "13 of 17" will NOT be reconciled with a body's
  "76%" and will false-positive. Keep summary numbers literal, or expect the
  tripwire to fire.

A clean run means "no tripwire fired", not "the artifacts are proven
consistent". Any check that fires exits `6` and appends a `RUNS.md` row.

## Catches — the institutional-memory loop (no LLM)

A validator's value compounds only if its catches are remembered. `lens-kit
catches` is the file-based memory: a `catches.jsonl` in your working directory
that records every defect a validator finds, surfaces the recurring traps
before the next run, and flags a pattern for promotion to a deterministic
check once it has recurred enough.

**The flywheel.** An agent catches a defect; the same pattern recurs across
runs; at a threshold `stats` says *promote this to a deterministic check*; the
check then runs in pure Python forever after — cheaper and earlier than any
LLM pass. The shipped `consistency` checks (marker parity, leak scan, number
parity) were built **exactly this way** from a real validator's catch history:
they are the first three patterns to complete that journey, and
`consistency.py` is the extension point the promotion suggestion points at.

**Exclusion discipline (why passes are not recorded).** The memory holds
defects, not "looked fine" notes. `catches add` rejects an empty or pass-like
`catch` by design — a memory full of routine passes is a memory no one reads.
A catch is required to NAME what was wrong; a catch without a `pattern`/`rule`
warns, because a catch without a generalization is half a catch (you re-learn
it the next time it shows up in a new shape).

```bash
# Record a catch (a NAMED DEFECT). Routine passes are rejected; a catch with
# no pattern/rule still records but warns.
lens-kit catches add \
  --catch "summary table dropped the [ESTIMATE] marker present in the draft" \
  --pattern "evidence markers get lost when prose is summarized into a table" \
  --rule "diff marker counts source vs every rendered output" \
  --artifact-type content-pack --domain marketing

# Agents append a full record from JSON (object or list of objects):
lens-kit catches add --from-json new-catch.json

# Seed a fresh memory with the shipped, fully-genericized example catches:
lens-kit catches add --seed

# Before a run: surface prior catches for an artifact type (or --all),
# most-recurrent patterns first — a block designed to paste into a validator
# agent's context:
lens-kit catches relevant content-pack
lens-kit catches relevant --all --domain marketing
lens-kit catches relevant content-pack --format json   # raw records for tools

# Per-pattern recurrence; any pattern at the threshold (default 3) gets a
# promote-to-deterministic-check suggestion:
lens-kit catches stats --threshold 3
```

The `relevant` paste block is a **stable, documented format** — a header with
the scope and count, then any promote-ready patterns marked `[PROMOTE]` at the
top, then one `CATCH / PATTERN / RULE` block per record in most-recurrent-first
order (a `(self-catch)` line marks validator-discipline failures). It is meant
to be read into an agent's context verbatim before it validates.

`catches` is a memory tool, not a gate: it exits `0` on success and `2` on a
usage error (including a rejected routine-pass `catch`). It writes no `RUNS.md`
row and assigns no score. The shipped seed (`catches add --seed`) is example
content — replace it with your own real catches as you record them.

## The validator agent — the cross-relationship tier

The deterministic kit reads one document at a time and fires literal tripwires.
Some defects are RELATIONSHIPS no single-document check can see: a contradiction
between two files, a number a summary invents, a marker dropped between platform
variants, a policy phrase carried verbatim from an internal brief into customer
copy. The **validator agent** is the tier above the gate that reasons across those
relationships — and it carries the institutional memory (`catches`) so its value
compounds run over run.

It is model-agnostic. [`docs/VALIDATOR-AGENT.md`](docs/VALIDATOR-AGENT.md) is the
protocol you hand to ANY agent; [`agent/validator-agent.md`](agent/validator-agent.md)
is a drop-in system prompt that implements it, with the verdict receipt formats in
[`agent/receipt-templates.md`](agent/receipt-templates.md). The kit's CLI is the
agent's entire toolbelt — no graph, no service. The 7-step loop, on real commands:

1. **classify** the artifact (domain, artifact_type, customer-facing?)
2. **read prior catches** — `lens-kit catches relevant <type> --domain <d>`
3. **run the gate** — `lens-kit validate <file> --profile <yaml> --json`
4. **run cross-checks** — `lens-kit consistency markers|leaks|numbers`
5. **name the downstream consequence** (advisory — what it costs if the
   load-bearing claim is wrong; changes attention, never the verdict arithmetic)
6. **render the verdict receipt** (per-lens table / CONDITIONAL / paired v1+v2)
7. **append new catches** — `lens-kit catches add ...` (routine passes excluded)

**Honest framing of what the agent is.** The agent layer's value traces to a real
validator's catch history — the recurring misses that are all CROSS-something, which
is why the layer is *relationship validation + institutional memory ABOVE* the gate.
It makes **no autonomy claim**: the agent RECOMMENDS, the gate's verdict is the
score of record, and a human owns any irreversible action (money, publishing,
sending, deleting). The boundary rules are load-bearing, not advisory: never validate
output you generated (a generator grading itself is circular); your reasoning
supplements the gate and never overrides a FAIL into a PASS; a claim about a file or
number needs a direct check, not prose plausibility; on a gate error the verdict is
UNKNOWN, never PASS; one bounded fix round per cycle, then escalate to a human. A
clean acceptance walkthrough of the full loop is in
[`docs/VALIDATOR-AGENT-walkthrough.md`](docs/VALIDATOR-AGENT-walkthrough.md).

### Run ledger (RUNS.md)

`compile`, `eval`, and `consistency` append a row to `RUNS.md` in the working
directory (created with header on first use): run id, date, command,
checkpoint sha, data file, key metrics, status
(KEEP/DISCARD/STOPPED_COST/KILLED/EVAL/CONSISTENCY_OK/CONSISTENCY_FAIL). Rows
are append-only — negative results are evidence. Rule (printed in the
header): any edit to labels, fixtures, or data files forces a NEW run
designation; never compare metrics across a label edit under the same run id.

**Sensitive arguments are not echoed in the clear.** The ledger records the
command for provenance, but a `consistency leaks --deny <term>` value is an
internal phrase you are scanning FOR — it must not be written into `RUNS.md`.
Deny-list values are recorded as `--deny <redacted xN>` by default; the global
`--no-echo-args` flag drops all arguments and records only the subcommand
(`lens-kit consistency leaks copy.md [args redacted]`) for anyone who wants
nothing but the command name in the ledger.

**Verdict semantics:** the gate fails on any critical/high-severity
violation, AND independently whenever any lens other than relevance
reports violations — even warning-severity ones. Relevance is warning-only;
consciousScan only annotates. On a Rights HALT, `per_lens` contains only
`rights`; absent lenses were not evaluated. Per-lens wall times are
recorded under `result.timings` when called with `include_timings=True`.

### Credentials and the .env footgun

litellm (pulled in by dspy) can autoload a `.env` via python-dotenv at
import time. If you run inside a repo tree with an ancestor `.env`, API-key
variables may be silently repopulated — an endpoint you believed
unconfigured can quietly become live, defeating fail-closed checks that
test for a missing key. lens_kit itself never loads dotenv: the only
credential path is the env var NAMED by `api_key_env`. Audit your
environment (`env | grep -i key`) and working directory before trusting a
fail-closed result. The kit's own test suite strips all `*_API_KEY`-style
variables and runs from a neutral temp cwd for every test, so it is immune
to ambient keys.

## The 10 canonical lenses

rights (HALT gate), truth, causality, definitionalIntegrity (undefined
load-bearing term / equivocation — added 2026-06-14, runs after causality;
causality keeps derivation), contradiction, extrapolation, structure,
consistency, relevance (warning-only, needs context),
consciousScan (broad judgment-detection — qualia + aesthetic/emotional/moral
judgment; non-blocking flags). Plus two support layers:
ViolationCrossCheck (safety net) and AutoFix (error correction), and a
claim-extraction pre-layer.

**Detect freely, filter deterministically:** the LLM signatures detect
generally; all deterministic suppression lives in `filters.py` and is
driven entirely by profile vocabulary. An empty profile means no
suppression at all (pure LLM detection — more false positives, never fewer
true positives). The shipped `profiles/agency-example.yaml` is the
production agency vocabulary as a worked example; copy it and replace the
`[domain]`-tagged lists with your terms.

## Extracted vs deferred (honesty table)

Source: extracted from the author's internal validation gate and claim
extractor. Those originals are not published; this package is the extraction,
and the tests here are the contract.

| Item | Status in C1 |
|---|---|
| 10 lens signatures (verbatim docstrings/prompts) | Extracted |
| ViolationCrossCheck safety net + AutoFix pass | Extracted |
| Claim-extraction pre-layer (truth/extrapolation routing) | Extracted |
| BestOfN(N=3) on Truth/Extrapolation + `fast_mode` (N=1) | Extracted |
| Full deterministic filter layer (13 filters) | Extracted, vocab moved to profile |
| TRUTH_WHITELIST, SCENARIO_LABELS, internal-source, first-party, temporal, structure-trigger lists | Externalized to profile YAML |
| DOMAIN_LENS_RULES strictness preambles | Externalized to profile (`domain_rules`) |
| Domain auto-detection keywords | Externalized to profile (`domain_detection`) |
| Verdict structure (LensResult/LensViolation, per_lens, halt, timings) | Extracted |
| Sequential pipeline (`_forward_sync`) | Extracted as `forward()` |
| Async parallel pipeline (`aforward`, asyncio.gather fan-out, `LENS_GATE_ASYNC`) | **Deferred** — the *asyncio* path was not ported. Note this is not the same as "no concurrency": thread-based per-stage parallel execution **did** ship and is the **default** (`LensGate(parallel=True)`, see CHANGELOG). `LensGate(parallel=False)` selects the sequential reference. |
| GEPA checkpoint key remapping (`gate.*` prefix strip, BestOfN `.module` remap) | Extracted (C2, `checkpoint.load_checkpoint`); the `LENS_CHECKPOINT_DIR`/`LENS_GATE_NO_CHECKPOINT` env-var auto-load is **deliberately not ported** — checkpoints are explicit via `--checkpoint` |
| `configure_lm` provider fallback chain (a hardcoded vendor order plus env switches) | **Deliberately not ported** — replaced by the explicit, fail-closed profile `llm` block. You name one endpoint; if its key is missing the kit stops instead of falling back to a vendor you did not choose. |
| `dotenv` autoload of a project-local `.env` file | **Deliberately not ported** — credentials are explicit via `api_key_env` |
| `cli_theme` rich console output | **Deliberately not ported** — plain text + `--json` |
| GEPA compile harness (F2 metric, scored-lens selector, patience early-stop) + costing gate + pace monitor | Extracted (C2, `compile_harness.py` / `costing.py`) |
| Holdout eval harness (per-lens confusion, mismatch strings, variance envelope, versioned receipts) | Extracted (C2, `eval_harness.py`) |
| Run ledger (RUNS.md) | Added in C2 (`ledger.py`); calibrate appends CALIB rows in C3; mutate/label-audit append MUTATE/LABEL_AUDIT rows in C4 |
| Checkpoint sha256 + provenance sidecar | Added in C2 (`checkpoint.py`, minimal stub); upgraded to the FULL sidecar in C4 (`sidecar.py`), backward-compatible with every stub field |
| Planted-flaw calibration battery (generator + runner, verdict mapping, gold labels, variance envelope) | Added in C3 (`calibration.py`, `lens-kit calibrate generate`/`run`) |
| Mutation control (seeded mutants from real holdout, rubber-stamp detection) | Added in C4 (`mutation.py`, `lens-kit mutate`, `eval --mutation-control`) |
| Gold-label audit (Confirmed/Borderline/False-Alarms scaffold, impact math, backup enforcement, run-redesignation rule) | Added in C4 (`label_audit.py`, `lens-kit label-audit`) |
| Full provenance sidecar (`not_a_claim_of` defaults, dataset/eval blocks, single-run caveat, re_measurements) | Added in C4 (`sidecar.py`, `lens-kit sidecar`) |
| Review surface (eval-viewer HTML + feedback.json) | Added in C5 (`review/`, `lens-kit review`); adapted from Anthropic's eval-viewer (Apache-2.0) — kept the self-contained HTML + stdlib feedback server + previous-feedback diff; replaced the run-directory model with lens-kit's flat JSON artifacts. See the attribution block below |
| Consistency-lens claim context (`consistency_claim_context`) | **Not ported** — dead code in the source: computed but never passed to the Consistency lens (its signature has no domain field) |

## Tests

No-network suite (stubbed predictors, no live API):

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

**Apache License, Version 2.0** — Copyright (c) 2026 Soulfield.

The grant is in [`LICENSE`](LICENSE) at the root of this package; [`NOTICE`](NOTICE)
carries the copyright notice and the third-party attribution. You may use, modify,
redistribute and train this kit on your own data under those terms.

One subcomponent — the review surface — is a modified derivative of third-party
Apache-2.0 code and keeps its own notice; see below.

## Attribution (review surface)

The `lens-kit review` surface (`src/lens_kit/review/`) is adapted from
Anthropic's skill-creator **eval-viewer** (`generate_review.py` + `viewer.html`),
distributed under the **Apache License, Version 2.0**. The full license text
ships at `src/lens_kit/review/LICENSE.Apache-2.0` and the attribution + changes
list at `src/lens_kit/review/NOTICE`.

**Kept** from the upstream: the self-contained embedded-data HTML (single
`/*__EMBEDDED_DATA__*/` injection point), the tiny stdlib-only HTTP server, the
GET/POST `/api/feedback` round-trip that auto-saves to `feedback.json`, the
`--previous-feedback` diff context (and stale-prefill suppression), `--port`,
`--static` export, keyboard navigation, and the submit-all-records contract.

**Changed / added**: replaced the skill-creator run-directory model (a "run" =
a directory with an `outputs/` subdir, plus `transcript.md` / `user_notes.md` /
`grading.json` conventions and a recursive workspace scan) with lens-kit's flat
JSON artifact model and per-record rendering; added artifact-type detection and
kind-specific chips/summaries (eval / calibration / mutation); changed the
feedback key from `run_id` to `record_id` (the loader reads either); defaulted
the port to 0 (ephemeral) and dropped the `lsof`-based fixed-port kill; stripped
skill-creator-only notions (skill name, benchmark with/without-skill tabs,
image/pdf/xlsx output embedding).

## Claims discipline

**This package makes no accuracy claims.** Not "low" ones — none.

Any figure you see in these docs is there to demonstrate the *method*, never to
sell the tool. Every one of them is an internal measurement on our own frozen
agency-domain holdout, taken on a stated date against a served model that has
since changed. They are **not independently verifiable from outside this repo**,
they carry **no transfer promise**, **no accuracy floor on your data**, and no
concurrency or scale claim. Treat them as a worked example of the discipline in
`docs/WORKED-EXAMPLE.md` — which exists precisely because one of our own numbers
moved by eleven points with no change to the code, the data, or the local stack.

If you want a number you can trust for your own use, the honest route is the one
this kit is built for: run it on your own material, against your own holdout,
and read the envelope it gives you. That figure is yours and it is the only one
that means anything about your domain.
