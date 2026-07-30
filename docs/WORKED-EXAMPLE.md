# Worked Example — when the upgrade didn't break it

This is a real run, retold as a tutorial. We took a gate that had a promoted
accuracy number, bumped its dependencies, recompiled it, and re-measured it on
the same frozen holdout. The headline number moved by eleven points. The
interesting part is not the drop — it is the chain of receipts that told us
**why**, and the two wrong conclusions the discipline stopped us from shipping.

Every number in this document is ours, measured on our own data and model. None
of it transfers to your deployment — see [CLAIMS.md](CLAIMS.md). It is here to
show the *method*, not to set an expectation.

The environment is described generically on purpose: a fresh virtual
environment, a dependency-version bump, the same frozen 83-example holdout, the
same eval harness, the same checkpoint that had earned a promoted number months
earlier.

---

## What we set out to do

A dependency bump — the optimizer library and the framework underneath it — went
from old pinned versions to new ones. The plan was the standard one: recompile
the gate against the frozen holdout and re-run the full discipline end to end, so
that every receipt becomes a worked example a customer can reproduce on their own
data. The expectation going in was mundane: a version bump should be roughly
inference-neutral, and a recompile might nudge the number a little.

Here is the result matrix we ended up with. The "as-of" run is the historical
promoted number; everything else is "tonight".

> **Read the fence before the numbers.** Every figure in this table is an
> internal measurement on our own frozen 83-example agency-domain holdout, taken
> on the dates named below, against a hosted mid-size open-weight model whose
> served version changed between the two runs. They are **not** a claim about
> this package's accuracy, **not** transferable to your domain or dataset, and
> **not** independently verifiable from outside our tree. They are here because
> the *shape* of the change is the lesson. Your own numbers are the only ones
> that mean anything about your data — see `docs/CLAIMS.md`.

| Run | Accuracy | Catch | FP | Note |
|---|---|---|---|---|
| Canonical, as-of 2026-04-07 | **88.0%** | 72.4% | 8.7% | the historical promoted lane |
| Same checkpoint, re-measured tonight | **77.2%** | 71.4% | **21.5%** | real calls, new stack |
| Uncompiled baseline, tonight | 73.4% | 79.3% | 27.8% | the untuned gate, for reference |
| Light recompile, tonight | — | — | — | returned the seed: no improvement found |
| "3-run variance battery", tonight | 77.2% ×3 | — | — | **VOID — see below** |

The headline: the identical checkpoint, on the identical frozen holdout, scored
**88.0% then and 77.2% now**, with the false-positive rate going from 8.7% to
21.5% — more than doubled. Eleven points below the historical variance floor.
Nothing in the checkpoint, the holdout, or the harness file had changed.

---

## Wrong conclusion #1: "the upgrade broke it"

The obvious story is that the dependency bump broke the gate. It is the wrong
story, and the receipts say so.

**What the discipline did.** We did not trust the new number on its own — we
checked whether the two stacks were even sending the same requests. The
optimizer's disk cache keys on the exact request. When we ran the *old*-version
control, it **replayed the new-version run's cache** — at a fraction of the time
per example (cache hits, not fresh calls). That cache collision is itself the
evidence: if the old and new stacks produced cache-key-identical requests, they
were sending byte-identical requests. The checkpoint also loaded under the new
version without any key remapping and produced cache-key-identical requests. A
format or behavior break would have produced a load error or cache *misses*. It
produced neither.

So the dependency bump was inference-neutral for this checkpoint. The requests
going out were the same. If the requests were the same and the answers were
different, the change was on the **response** side — the model — not the local
stack.

**The probe that confirmed it.** We then queried the provider's model list
directly. The model id our gate named was **absent** from the list. A live
completion requested against that id came back labelling itself as a *different,
newer model* — created about a week after the original promoted measurement. The
provider had silently redirected a now-deprecated alias to a newer model. A
frozen served model would have echoed the requested id; this one did not.

**The real conclusion:** the upgrade did not break it. The *served model behind
the alias changed*, with no code change and no warning. The "88% → 77.2%" drop is
served-model drift, and it is the exact failure that
[CLAIMS.md §4](CLAIMS.md) (era-stamping and re-baseline) exists to handle. The
number was not wrong when it was measured; it became stale when the model behind
it changed.

This is why the kit's sidecar `not_a_claim_of` list and as-of stamping are
load-bearing. A bare "88% accuracy" line, copied forward without its date, would
now be a false claim — through no fault of the gate.

---

## Wrong conclusion #2: "3 identical runs, so the number is stable"

After re-measuring, we ran the eval three times to get a variance envelope. All
three returned **77.2%**. Three identical runs looks like rock-solid stability.

It was nothing of the sort. The three "runs" completed in about twenty seconds
total — far too fast for real model calls. They were **cache replays**: the
optimizer's disk cache served the first run's responses to the second and third.
The measured variance was zero **by construction**, not because the gate is
deterministic. There was no envelope; there were not even three runs. There was
one run, printed three times.

**What the discipline requires.** A real variance envelope needs the cache
pointed at a fresh directory per rerun, so each rerun makes real calls:

```bash
lens-kit eval holdout.json --profile my-profile.yaml \
    --checkpoint ckpt.json --reruns 3 --fresh-cache
```

Without `--fresh-cache`, the kit warns and records a `cache_caveat` in the
envelope — and an envelope with that caveat is not a stability measurement. We
filed every tonight number as **single-run** with the caveat travelling with it,
because that is what the receipts actually supported. The real envelope was an
explicit, costed, separately-approved next step — not something to fake by
re-printing a cached number three times.

**The real conclusion:** "three identical runs" was a cache artifact, not
evidence of stability. The honest claim was a single-run number with the
single-run caveat attached, full stop.

---

## What the recompile told us (and what it didn't)

We also did a light recompile of the gate against the current model. It found
**nothing better than the existing checkpoint**: every proposed mutation scored
worse than the seed on its subsamples, the best-program index never moved off the
seed, and the output prompts came back byte-identical to what we started with.

The discipline here is to caveat the number precisely, not over-read it:

- It **does** show that a light-budget recompile did not improve the gate against
  the current model.
- It does **not** show that the gate cannot be improved for the current model, or
  that the method is broken — the original improvement arc on the earlier model
  remains a valid receipt of the method. It was a light budget on a deterministic
  subset, seeded from an already-optimized program. A larger budget might find
  improvements; that is untested, and we tagged it `[PROJECTION]` rather than
  claiming it either way.
- And it carries a confound we disclosed rather than buried: the recompile seeded
  from the deployed checkpoint, so "found nothing better than the seed" means
  "nothing better than the *already-optimized* program as scored by the current
  model" — it does not measure from-scratch compile capability.

That is the residual-triage habit: one change per attribution, every projection
tagged, every "does NOT prove" written down next to every "does prove".

---

## The lessons the kit ships because of this run

1. **Scorecards are model-snapshot-stamped.** A promoted number needs an as-of
   date and a re-baseline plan, and provider drift is a first-class event — not an
   edge case. The kit treats it as one.
2. **Recompile-on-drift is not automatic recovery.** A light recompile did not
   restore the earlier performance. When the served model changes, re-tuning is
   work with an uncertain budget, not a one-command fix.
3. **Pin a listed model id, not a deprecated alias.** The alias was the thing that
   got swapped underneath us. Where the provider offers dated, listed ids, pin one
   — and treat de-listing of your id as a re-baseline trigger.
4. **A green "stable" reading can be a cache replay.** Always demand
   `--fresh-cache` before you call a variance envelope real.

The whole point of the kit is that none of these were judgement calls made under
pressure. They were forced by the receipts: the cache-collision evidence, the
model-list probe, the versioned eval files, the cache caveat in the envelope, the
`[PROJECTION]` tags. The discipline produced the right conclusion and blocked two
plausible wrong ones.

---

## Commands used in this example

Every command and flag is verified against the kit's CLI.

| Stage of the story | Command |
|---|---|
| Re-measure the checkpoint on the frozen holdout | `lens-kit eval holdout.json --profile P --checkpoint ckpt.json` |
| Measure the untuned baseline for reference | `lens-kit eval holdout.json --profile P --baseline` |
| Light recompile against the current model | `lens-kit compile training.json --profile P --output ckpt.json --auto light` |
| A REAL variance envelope (what the cached battery wasn't) | `lens-kit eval holdout.json --profile P --checkpoint ckpt.json --reruns 3 --fresh-cache` |
| Re-attach the evidence boundary after re-baselining | `lens-kit sidecar ckpt.json --profile P --dataset holdout.json --eval-results eval-envelope.json` |

The drop from 88.0% to 77.2%, the doubled false-positive rate, and the
served-model substitution are all ours, on our agency-domain holdout, against a
specific model, on specific dates. They do not transfer to your data — they
demonstrate the method. Run [the loop](MANUAL.md) on your own holdout to get
your own scorecard.
