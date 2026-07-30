# The lens-kit Methodology Manual

This is the training loop the kit is built around. The optimizer it wraps is
free; what you are buying — and what this manual teaches — is the discipline
*around* the optimizer: how to label, baseline, calibrate, cost-gate, evaluate
with real variance, do forensics on the misses, fix one thing at a time,
prove the gate can still fail a planted flaw, and ship a number that carries
its own evidence boundary.

The manual teaches a **method**. It does not promise a number. Every accuracy
figure depends on your data, your model, and the day you measured it — see
[CLAIMS.md](CLAIMS.md) for why numbers do not transfer, and
[WORKED-EXAMPLE.md](WORKED-EXAMPLE.md) for a real run where a single headline
number changed by eleven points without anyone touching the gate.

Each step below gives you: **why** the discipline exists (and the failure it
prevents), the **exact command**, and **what the receipt looks like**. Run the
steps in order. The order is load-bearing.

---

## The loop at a glance

```
label data
  → baseline eval            (measure the untuned gate first)
  → calibrate                (prove the gate can FAIL the right things)
  → compile  [costing gate]  (tune — but only after you've seen the price)
  → eval + variance          (measure with a real envelope, not a cache replay)
  → FN forensics             (read every miss before you change anything)
  → ONE bounded fix round     (change one thing, re-measure)
  → re-eval
  → mutation control         (can the tuned gate still fail a planted flaw?)
  → sidecar                  (bolt the evidence boundary onto the number)
  → ship the claim           (only what the receipts support)
```

You will go around the FN-forensics → fix → re-eval inner loop more than once.
You should go around it **one change at a time**. The reasons follow.

---

## Step 1 — Label your data

**Why.** The gate is only as honest as the labels you score it against. A
mislabeled example penalizes a correct gate and rewards a broken one; if you
tune against it, you are optimizing toward a data bug. The kit treats labels as
the foundation, not an afterthought — and it refuses to let a typo'd label slip
through silently, because a label outside the ten canonical lenses can never
match gate output and would score a quiet zero that poisons your metrics with no
trace. (Source for the data-bug risk: recovered practice 1.3, gold-label audit.)

**What you produce.** Two plain-JSON files of `{text, per_lens, domain?,
source?}` records:

- `training.json` — the examples you will tune on.
- `holdout.json` — a frozen set you will **never** train on. This is the only
  honest number you have. Keep it separate and keep it frozen.

**Receipt.** None yet — but note the kit's load-time guard: any `per_lens` key
outside the ten canonical lenses is a hard error, not a warning. A warning
would scroll past in a multi-thousand-rollout compile; a hard error stops you
before you measure garbage.

---

## Step 2 — Baseline eval (measure the untuned gate first)

**Why.** Before you change anything, you need a frozen *before* number. Without
a baseline you cannot tell whether a later change helped, did nothing, or hurt —
you only have an *after* with nothing to compare it to. The recovered discipline
here is the hallucination/quality baseline census: freeze the starting point as
an anchor, then every later number is a delta against the same denominator.
(Source: recovered practice 1.4, baseline census; 1.6, self-describing receipts.)

**Command.**

```bash
lens-kit eval holdout.json --profile my-profile.yaml --baseline
```

**Receipt.** A versioned `eval-<date>-runK.json` file (results are *never*
overwritten — see Step 6 on why) plus a row appended to `RUNS.md`. The results
file is self-describing: per-lens TP/TN/FP/FN counts and per-example mismatch
strings (`"causality: FN (expected FAIL, got PASS)"`). Those mismatch strings
are what make Step 7 (FN forensics) possible by grep alone, weeks later, with
zero re-runs. If your receipts do not carry per-example mismatches, you cannot
do forensics — you can only re-run and hope.

---

## Step 3 — Calibrate (prove the gate can FAIL the right things)

**Why.** A gate's PASS is only meaningful once you have proven it can FAIL the
right things and clear clean content. A gate that blocks everything is not safe,
it is broken: it teaches you nothing about where your tuning helped, and it reads
to a customer as a rigged scare tool. So before you trust any verdict, you run a
battery of fixtures with **known** planted flaws and **known-clean** controls,
each carrying a pre-agreed acceptance rule — clean content must not be blocked, a
contradiction must HOLD, a dark pattern must HOLD, an uncited statistic must at
least demand sources. (Source: recovered practice 1.3 / the planted-flaw
calibration suite; this is mutation control aimed at your own gate.)

**Why outside-in, grounded in the research.** The reason the scorer of record is
an *external* gate — not the model grading its own output — is that
preference-tuned models optimize for *evaluator approval*, which diverges from
*accuracy*. The "Machine Bullshit" study (arXiv 2507.07484, Princeton) measures
this directly: a model's *bullshit index* — its indifference to truth — sits near
zero before RLHF and *rises* after it, while user satisfaction climbs ~48%. The
model learns to be more persuasive and more approved-of, not more correct; that
is Goodhart's law on a reward signal. Sycophancy is the same failure measured
another way: SycEval (arXiv 2502.08177) finds sycophantic behaviour 58.19% of the
time across three major models on math and medical tasks. The fix-direction the
literature points at is *outcome-conditioned* feedback — reward the model for the
downstream result, not the in-the-moment approval (RLHS, arXiv 2501.08617, same
lab: conditioning on hindsight beats foresight, even when the outcome is only
*simulated*). The kit's mechanisms are built to that shape: the gate is an
external scorer rewarded against **defect-labeled** fixtures (this calibration
battery — it cannot earn its score by being agreeable, only by catching the
planted flaw); **mutation control** (Step 8) is the rubber-stamp detector that
fails a gate which has learned to approve; and the **catches loop** (Step 7) is
outcome-grounded (hindsight) feedback — real defects found downstream, fed back
as the next run's memory. (A study widely cited for downstream harms of
sycophancy, arXiv 2510.01395, is noted by title only here — its popular figures
are not verified against the primary, so no number from it is quoted.)

**Command.**

```bash
# Generate fixtures from your profile's calibration vocabulary (no LLM,
# deterministic — same seed + profile = byte-identical output):
lens-kit calibrate generate --profile my-profile.yaml --out battery/ --per-class 3

# Run the battery and score each verdict against its pre-agreed rule:
lens-kit calibrate run battery/ --profile my-profile.yaml
```

**Receipt.** Each fixture ships with a `*.gold.json` sidecar recording its
class, expected per-lens labels, acceptable/target verdicts, and a sha256 of the
fixture bytes. The run writes a versioned `calib-<date>-runK.json` and appends a
`CALIB` row to `RUNS.md`. Exit code `0` means every fixture was acceptable on
every rerun; exit `5` means the battery has failures — an unacceptable verdict or
an UNKNOWN. `5` is deliberately distinct from the costing-gate `3` so a wrapper
can tell "the gate is weak" apart from "too expensive to run".

**Failure this prevents.** Shipping a gate you have never proven can fail
anything. Re-run this battery after *every* tuning change: a clean fixture that
starts getting blocked, or a planted flaw that starts passing, is a regression
you see before a customer does.

**Experimental classes (read this before quoting their numbers).** The battery
generates four original, gold-audited flaw classes (clean, contradiction,
darkpattern, unsourced) and four **experimental** ones from the Machine-Bullshit
taxonomy (weasel, paltering, emptyrhetoric, sycophantic; arXiv 2507.07484). The
experimental fixtures carry `"experimental": true` in their gold sidecar and are
**EXPERIMENTAL until their gold labels are audited on your real gate outputs** —
their per-lens labels are design assertions, not yet confirmed against what your
gate actually does. The run report keeps them out of the headline `fn_rate` and
reports them under a separate `experimental_fn_rate` block; treat that block as a
direction to investigate, not a measured catch-rate, until you have run a
label-audit (Step 7) over them.

---

## Step 4 — Compile, behind the costing gate

**Why (the discipline).** Tuning runs an optimizer, and optimizers cost money
and wall-clock. The recovered rule is blunt: project the spend *before* you
launch, present the number, and hard-stop anything over the approval threshold.
This exists because a real GEPA run once projected roughly $360 of API spend and
survived to a small fraction of that **only because a human happened to be
watching**. (Internal observation, one run on our own endpoint and prices — not
a benchmark, and not a prediction about your costs.) The lesson:
projection must be **code, not vigilance**. (Source: recovered practice 1.9,
pre-run costing + kill-at-projection.)

**What the gate does.** Before compiling, the harness computes projected
rollouts from the optimizer's own budget accounting (`--max-evals N` →
N × (train+val); `--auto` → the auto budget), measures live seconds-per-rollout
on a small probe chunk, and prints projected wall-clock — plus a dollar figure
*only* if your profile sets `compile.cost_per_rollout_usd` (otherwise it prints
UNKNOWN rather than inventing a number — see [CLAIMS.md](CLAIMS.md) on never
inventing figures). A projection over the threshold (default 2h;
`compile.approval_threshold_hours`) is a **hard stop** unless you pass
`--approve-cost`. During the run a pace monitor aborts with a structured kill
record if measured pace exceeds the approved projection by
`compile.pace_kill_factor` (default 1.5×).

**Command.**

```bash
lens-kit compile training.json --profile my-profile.yaml --output ckpt.json \
    --auto light --threads 4            # or --max-evals N (mutually exclusive)
# Over the threshold? The harness stops with exit 3 until you re-run with:
lens-kit compile training.json --profile my-profile.yaml --output ckpt.json \
    --auto light --approve-cost
```

**Receipt.** A checkpoint `ckpt.json`, its `ckpt.json.sha256` sibling, an
automatic `ckpt.json.provenance.json` sidecar (Step 9 generates a richer one),
and a `RUNS.md` row with status `KEEP`, `STOPPED_COST`, or `KILLED`. If the
costing gate stops the run, exit code `3`; if the pace monitor kills it, exit
`4` and a `ckpt.kill.json` record (% done, elapsed, projected remaining,
reason). **Exit `3` is reserved for the costing stop alone**, so a wrapper never
confuses "too expensive" with "gate too weak" (`5`).

**Failure this prevents.** An overnight run that completes on momentum and bills
you for a result you would never have approved. The projection number lands
*before* the spend, not after.

---

## Step 5 — Eval with a real variance envelope

**Why.** A single eval number is a coin you flipped once. The same checkpoint on
the same holdout will not return bit-identical metrics across runs, because the
model is non-deterministic. The recovered discipline publishes **two** numbers:
a canonical claim and a worst-of-N floor from N reruns. (Source: recovered
practice 1.7, variance bounding.)

**The cache trap — read this before you trust an envelope.** The optimizer's
disk cache replays identical responses for identical prompts. A variance battery
run inside the project was once **voided by exactly this**: the "reruns" were
cache replays and the measured variance was zero by construction — three runs
that looked rock-stable were one run played three times. The kit therefore prints
a cache warning, records a `cache_caveat` in the envelope when reruns shared a
cache, and gives you `--fresh-cache` to point the cache at a fresh temp dir per
rerun so the envelope reflects **real** model variance.

**Command.**

```bash
# Evaluate the compiled checkpoint with a REAL variance envelope:
lens-kit eval holdout.json --profile my-profile.yaml --checkpoint ckpt.json \
    --reruns 3 --fresh-cache
```

**Receipt.** A versioned `eval-<date>-runK.json` carrying per-metric
min/max/mean/pstdev and a **direction-aware worst-of-N floor** (accuracy and
catch floor at the MIN observed; FP rate floor at the MAX observed — the floor
is always the conservative direction). A `RUNS.md` row is appended. If you omit
`--fresh-cache` and the reruns shared a cache, the envelope says so in
`cache_caveat` — an envelope with that caveat is not a variance measurement, it
is the same run printed N times.

**Failure this prevents.** Citing a single lucky run as a stable property. A
worst-of-N floor is the number you can stand behind; the canonical mean is the
number you measured.

---

## Step 6 — FN forensics (read every miss before you change anything)

**Why.** When the gate misses, the next move is **not** to edit a prompt and
re-run. It is to read every false negative, tabulate it, cluster the misses into
named patterns, and pick the cheapest fix class per cluster — with a
pre-committed forecast of how much it could recover and what false-positive risk
it carries. The recovered arc that did this took a gate from N% to N%
accuracy by reading the misses first; one one-clause prompt edit, chosen from a
named cluster, moved a single lens's catch rate from 28.6% to 42.9%. "Tune and
hope" produced none of that. (Source: recovered practice 1.1, FN forensics →
pattern clustering → fix-class with a recovery ceiling.)

**How.** This is a forensic reading step you do over the receipts, not a single
command. The eval results file from Step 5 carries the per-example mismatch
strings (`"truth: FN (expected FAIL, got PASS)"`) — that is what makes the table
buildable by grep. You tabulate each FN as:

| source_id | domain | expected | actual | excerpt | likely miss pattern | proposed fix class |

…then cluster the rows into named patterns (e.g. "vague-source attribution",
"financial projection treated as data not plan"), choose ONE fix per cluster,
and write a forecast: a floor/central/ceiling projection of the recovery, tagged
`[PROJECTION]`, plus the FP risk the fix introduces, plus an explicit list of
what the fix does **not** touch. (Source for the projection-and-not-authorised
format: recovered practice 1.2, single-lane residual triage.)

**Receipt.** A short markdown forensics doc you keep next to the run: the FN
table, the clustered patterns, and the one bounded fix you will try next with
its `[PROJECTION]` forecast. This doc is the input to Step 7.

**Failure this prevents.** Changing five things at once, getting a different
number, and never knowing which change did it — or whether you traded a catch
improvement for a false-positive regression you did not look for.

---

## Step 7 — One bounded fix round

**Why.** Change **one thing**, then re-measure the whole holdout. The recovered
fix-then-revalidate discipline is explicit: one lens per round, one located fix,
a delta-only re-validation, and both the before and after receipts preserved.
The opposite — a batch of simultaneous edits — destroys attribution: you cannot
say which edit helped and you cannot catch the one that quietly regressed another
lens. (Source: recovered practice 1.5, fix-then-revalidate, one lens per round.)

**Before you blame the model: audit the gold.** When the gate disagrees with a
gold label, the **label** might be wrong. Before you accept an eval's FNs and FPs
as the gate's fault, audit the disagreements and decide, per example, whether the
label or the model is at fault. A real audit of an 83-example holdout found two
of nine consistency "misses" were **phantom** — the gate was right and the gold
label was wrong; tuning against them would have optimized toward the data bug.
(Source: recovered practice 1.3.)

**Command.**

```bash
# Build the Confirmed / Borderline / False-Alarms audit scaffold for every
# example where gate and gold disagree, with impact math:
lens-kit label-audit holdout.json eval-<date>-run1.json

# Before you EDIT any label, make the sanctioned dated backup (relabeling
# without it is refused):
lens-kit label-audit holdout.json eval-<date>-run1.json --backup
```

**Receipt.** A label-audit scaffold: every disagreement seeded into *Borderline*,
both relabel directions spelled out, and an impact section quantifying the
**upper bound** of label-attributable error (how many FNs/FPs would flip if every
suspicious label turned out wrong — an upper bound, not a verdict; the human
decides each row). A `LABEL_AUDIT` row is appended to `RUNS.md`. The workflow
never edits your dataset; a relabel requires the explicit `--backup`, which writes
a dated copy and never overwrites a prior one; and every run prints the
new-run-designation rule.

**The run-redesignation rule (do not skip it).** A label edit forces a **new run
id**. You must never compare metrics across a relabel under the same run id —
otherwise a label correction reads as a gate improvement, and the two become
indistinguishable in your record. (Source: recovered practice 1.3, separate run
designation for label changes.)

Once the label question is settled, apply your *one* prompt/profile fix from Step
6 and re-run the eval (back to Step 5). Keep both the before and after results
files. The `RUNS.md` ledger is append-only — a negative result is evidence, not
something to overwrite. Seed the "What was tried and didn't work" expectation
early: a fix that increased a false negative instead of cutting it belongs in the
record so nobody quietly retries it. (Source: recovered practice 1.6, run ledger
with DISCARD rows.)

---

## Step 8 — Mutation control (can the tuned gate still fail a planted flaw?)

**Why.** A green eval and a PASS verdict tell you the gate *can* clear content.
They do **not** tell you the gate can still *fail* a real defect after a compile
or a model change. The only way to know is to plant a flaw you KNOW is there and
confirm the gate fails it. If a planted mutant sails through, the gate is
rubber-stamping that defect class — and a green eval was hiding it. (Source:
recovered practice 1.1's sibling discipline; the formalism is borrowed from a
verification pipeline where a deliberately mutated artifact's receipts MUST FAIL
or the verifier is shown to be a rubber stamp — here the "verifier" is your gate.)

**Command.**

```bash
# Seed deterministic mutants from your real holdout; the gate MUST flag every
# one or the command exits 5 and NAMES the misses:
lens-kit mutate holdout.json --profile my-profile.yaml

# Or fold it into the eval as a post-check — a missed mutant overrides a
# green eval:
lens-kit eval holdout.json --profile my-profile.yaml --checkpoint ckpt.json \
    --mutation-control
```

**Receipt.** A mutation results file (when `--output-dir` is given) listing each
planted-defect marker, the gate verdict, and caught/MISSED. Mutant construction
is pure Python and seeded — same `(holdout, seed)` yields byte-identical mutants,
no LLM in the loop — so this is a cheap check you run after every compile. Exit
`5` if any mutant was missed, with the misses named so the failure is
attributable to a specific lens family, not a vague "the gate feels weak".

**Failure this prevents.** Trusting a green scorecard from a gate that has
quietly stopped catching a whole defect class after a tuning change or a model
swap. (For why a model swap can do this *under an unchanged config*, see the
worked example.)

---

## Step 9 — Sidecar (bolt the evidence boundary onto the number)

**Why.** A bare accuracy number travels — it gets quoted out of context, re-dressed
as a transfer promise, cited months after the model behind it changed. A
checkpoint on its own cannot tell you which model produced it, which data it was
tuned and scored on, what it scored, or what it is **not** a claim of. The sidecar
makes the number refuse to travel past its evidence. (Source: recovered practice
the `not_a_claim_of` provenance contract; see [CLAIMS.md](CLAIMS.md).)

**Command.**

```bash
lens-kit sidecar ckpt.json --profile my-profile.yaml \
    --dataset holdout.json --eval-results eval-envelope.json \
    --not-a-claim-of "our specific edge"
```

**Receipt.** A `<ckpt>.provenance.json` recording the checkpoint sha256 + size,
the model string and a provider note, the dspy/gepa/litellm versions, the dataset
file with counts and labels version, the eval metrics (with the worst-of-N floor
when you pass an envelope), an empty `re_measurements: []` for future drift
records, and — load-bearing — an explicit `not_a_claim_of` list. That list always
includes, and you can add to but never remove: no transfer to another domain or
dataset, no accuracy floor on customer data the gate was never tuned on, no
concurrency/scale claim, no per-call bit-reproducibility, plus a single-run caveat
whenever the eval had no variance envelope. The whole point is that a
"N%-on-our-holdout" figure cannot be quietly re-dressed into a promise about your
customer's data.

**The drift event you must plan for.** The served model behind a provider alias
can change underneath you, with no code change and no warning. When that happens,
the old number is stale — even though nothing in your repo moved. Treat provider
drift as a first-class event: re-run Step 5 against a **pinned, listed** model id
where the provider offers one, record the new measurement, and **era-stamp** the
old number rather than silently carrying it forward. The `re_measurements` array
in the sidecar exists to hold those dated re-baselines. The worked example is a
real instance of this exact failure. (Source: Stage-A report, served-model
substitution.)

---

## Step 10 — Ship the claim (only what the receipts support)

**Why.** The last discipline is the hardest: say only what you measured. The
promotion gate is a five-field check — what was measured, on what data, with what
model, as-of when, and explicitly what it is **not** a claim of. A number that
cannot pass that gate does not go in customer-facing copy. (Source: recovered
practice the claim-promotion gate; full doctrine in [CLAIMS.md](CLAIMS.md).)

**There is no `ship` command.** This step is a human gate, by design. The
artifacts you built — the versioned eval with its envelope, the `RUNS.md` ledger,
the calibration receipts, the mutation-control pass, and the sidecar with its
`not_a_claim_of` list — are the evidence. Your claim is whatever those receipts
support, era-stamped, and nothing more. The kit makes **no** accuracy claim about
your data, and neither should you until you have measured yours.

---

## What the kit refuses to do for you

These are deliberate. They are part of the discipline, not gaps.

- **It never invents a cost figure.** No `cost_per_rollout_usd` in your profile
  means the costing gate prints UNKNOWN, not a guess.
- **It never silently switches providers.** A missing key or model is a hard
  error. There is no fallback chain that could quietly send your data to a model
  you did not choose.
- **It never edits your labels for you.** Relabeling requires an explicit dated
  backup, and forces a new run designation.
- **It never overwrites a result.** Eval, calibration, and mutation results land
  in versioned filenames; `RUNS.md` is append-only. Negative results are evidence.
- **It never loads a `.env` for you.** The only credential path is the env var
  named by `api_key_env` in your profile — so an ambient `.env` in a parent
  directory cannot quietly make an endpoint live and defeat a fail-closed check.
  Audit your environment (`env | grep -i key`) before trusting a fail-closed run.

---

## Commands used in this manual

Every command and flag below is verified against the kit's CLI. No fictional
flags.

| Step | Command |
|---|---|
| 2 — baseline eval | `lens-kit eval holdout.json --profile P --baseline` |
| 3 — calibrate generate | `lens-kit calibrate generate --profile P --out battery/ --per-class 3` |
| 3 — calibrate run | `lens-kit calibrate run battery/ --profile P` |
| 4 — compile (costing gate) | `lens-kit compile training.json --profile P --output ckpt.json --auto light --threads 4` |
| 4 — compile, cost approved | `lens-kit compile training.json --profile P --output ckpt.json --auto light --approve-cost` |
| 5 — eval + variance | `lens-kit eval holdout.json --profile P --checkpoint ckpt.json --reruns 3 --fresh-cache` |
| 7 — label audit | `lens-kit label-audit holdout.json eval-<date>-run1.json` |
| 7 — label audit + backup | `lens-kit label-audit holdout.json eval-<date>-run1.json --backup` |
| 8 — mutation control | `lens-kit mutate holdout.json --profile P` |
| 8 — eval with mutation post-check | `lens-kit eval holdout.json --profile P --checkpoint ckpt.json --mutation-control` |
| 9 — sidecar | `lens-kit sidecar ckpt.json --profile P --dataset holdout.json --eval-results eval-envelope.json --not-a-claim-of "..."` |
| (any) — review surface | `lens-kit review results-dir/ --static review.html` |

Exit codes: `0` success/pass · `1` validation failed/halted · `2` usage/config
error (fail closed) · `3` costing-gate hard stop (compile only) · `4` pace kill ·
`5` gate failed to catch planted flaws (calibration failure or missed mutant).
