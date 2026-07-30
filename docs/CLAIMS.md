# Claims Discipline — say only what you measured

This document is the honest-claims doctrine the kit is built to enforce. It
governs the claims the kit makes about itself, and it is the template for the
claims **you** make about your gate. The rule underneath all of it: a number is
evidence for exactly the thing it measured, on the data and model it measured,
on the day it measured it — and nothing else.

This doctrine applies to this manual too. Nothing here promises an outcome on
your data. The one figure quoted below is *ours*, on *our* holdout, and it is
shown precisely to demonstrate how a number is supposed to be fenced — not as a
target you should expect to hit.

---

## 1. Evidence lanes — a number is promoted per-lane, never across

An **evidence lane** is one measured thing on one dataset under one model as-of
one date. Evidence proven on Lane A does not license a claim about Lane B. A
holdout accuracy measurement is a claim about *that holdout under that model* —
it is **not** a claim about customer-fitness, about concurrency, about
sustained-load behaviour, or about any other dataset. Each lane carries its own
promotion gate; per-lane evidence stays per-lane.

Concretely, the kit ships with exactly one promoted methodology lane and a
second runtime-fidelity lane, and it is explicit that the runtime-fidelity lane
is **NOT a claim of** accuracy, catch-rate, FP, customer-fitness, concurrency,
sustained-load, public-launch-readiness, or per-call bit-reproducibility. That
"NOT a claim of" list is not boilerplate — it is the fence that stops a fidelity
measurement from being re-quoted as an accuracy promise.

**Your version:** every measurement you take is a lane. Name it: *what* you
measured, *on what data*, *with what model*, *as-of when*. Write the lane down
with that header. A number without a lane header is a number that will get
quoted out of context.

---

## 2. The claim-promotion gate — five fields, or it doesn't ship

Before any number goes into customer-facing copy, it passes a five-field gate:

| Field | The question it forces |
|---|---|
| **What** | What metric, exactly? (accuracy / catch / FP / fidelity / …) |
| **On what data** | Which dataset, how many examples, frozen or not? |
| **With what model** | Which model id — and was it pinned or an alias? |
| **As-of when** | What date was this measured? |
| **Not a claim of** | What does this number explicitly NOT support? |

A number that cannot fill all five fields does not get promoted. "It scored N%"
is not a claim — it is a fragment. "It scored N% on our frozen 83-example
agency-domain holdout, against the served model on 2026-04-07, and is not a claim
of transfer to any other domain or dataset" is a claim. The difference is the
fence.

**Forbidden promotions** (these never pass the gate, no matter the receipt):

- Promoting a holdout number into a promise about a customer's data.
- Promoting an accuracy number into a concurrency or scale claim.
- Promoting a single run into a stability claim (no variance envelope = no
  stability claim — see §5).
- Promoting a number measured against one model into a claim about another.

---

## 3. `not_a_claim_of` — the boundary ships *with* the number

The kit makes the fence mechanical. Every checkpoint gets a provenance sidecar
whose `not_a_claim_of` list ships *attached to the artifact*. By default it
always includes, and you can add to but **never remove**:

- no transfer to another domain or dataset;
- no accuracy floor on customer data the gate was never tuned on;
- no concurrency / scale claim;
- no per-call bit-reproducibility;
- **no immunity from sycophancy / rubber-stamping** — the kit is *externally
  measured* against it (calibration false-negative rate, mutation control,
  variance envelopes), which is not the same as being free of it. The gate
  itself runs on an LLM under a compiled reward, so it is subject to the exact
  approval-vs-accuracy drift the measurements exist to catch; the claim is "we
  measure for it and surface it", never "the gate cannot do it";
- plus a single-run caveat whenever the eval had no variance envelope.

The point is that the number cannot be physically separated from its boundary. A
checkpoint is a pile of optimized prompts; on its own it cannot say what it is
NOT a claim of. The sidecar makes it say so. You generate it with:

```bash
lens-kit sidecar ckpt.json --profile my-profile.yaml \
    --dataset holdout.json --eval-results eval-envelope.json \
    --not-a-claim-of "your domain-specific edge"
```

The defaults being un-removable is the whole design: it means a
"N%-on-our-holdout" figure cannot be quietly re-dressed as a transfer promise,
even by someone who copies the number without the context.

---

## 4. Era-stamping and re-baseline on provider drift

A measured number has an as-of date because **the model behind it can change**.
This is not theoretical. In our own gate, a checkpoint that measured one accuracy
figure on one date measured a materially different figure later — eleven points
lower, with the false-positive rate more than doubled — **with no change to the
checkpoint, the data, or the local stack**. The cause was the served model: a
provider alias that had pointed at one model silently began serving a different,
newer model. A frozen served model would have echoed the requested id; this one
did not.

The discipline that follows:

1. **Era-stamp every number.** Carry the as-of date *with* the figure, always.
   Our own promoted lane reads, in shape: "N% accuracy / N% catch / N% FP on the
   83-example holdout, as-of 2026-04-07 — the identical checkpoint re-measured
   against the current model scored a lower figure; drop attributed to
   served-model drift, not the local stack." The old number is not deleted; it is
   *dated*, and the new measurement sits beside it.

2. **Pin a listed model id where the provider offers one.** An alias can be
   re-pointed under you. A dated, listed model id is the only thing you can lock.

3. **Watch for de-listing as a re-baseline trigger.** When your model id stops
   appearing in the provider's model list, that is your signal that the alias may
   now redirect — re-baseline before you cite the old number again.

4. **Re-baseline = re-run, not a special command.** There is no magic "rebaseline"
   button. You re-run the eval against a pinned model with a fresh variance
   envelope and generate a fresh sidecar; the sidecar's `re_measurements` array
   exists to hold those dated re-baselines:

   ```bash
   lens-kit eval holdout.json --profile my-profile.yaml \
       --checkpoint ckpt.json --reruns 3 --fresh-cache
   lens-kit sidecar ckpt.json --profile my-profile.yaml \
       --dataset holdout.json --eval-results eval-envelope.json
   ```

The lesson the kit was designed around: **version locks on your local stack are
useful, but the volatile dependency is the served model, and no local pin can
lock it.** Only the provider's dated model ids can — and only until they are
de-listed.

---

## 5. Variance before stability — one run is not a property

A single eval number is one coin flip. The model is non-deterministic, so the
same checkpoint on the same holdout will not return identical metrics twice. You
cannot call a number a *stable property* of the gate until you have measured its
spread.

The kit publishes two numbers when you ask for variance: a canonical mean and a
**direction-aware worst-of-N floor** (accuracy/catch floor at the MIN observed,
FP rate floor at the MAX observed — always the conservative direction). The floor
is the number you can stand behind; the mean is the number you measured.

```bash
lens-kit eval holdout.json --profile my-profile.yaml \
    --checkpoint ckpt.json --reruns 3 --fresh-cache
```

**The cache caveat is part of the claim.** If the reruns shared a disk cache,
they are cache replays, not real reruns — the "variance" is zero by construction.
A real variance battery was once voided by exactly this. The kit records a
`cache_caveat` in the envelope when this happens, and `--fresh-cache` is what
makes the reruns real. An envelope carrying a `cache_caveat` is **not** a
stability claim — it is the same run printed N times, and the sidecar will add a
single-run caveat accordingly.

---

## 6. The score-transfer ban — numbers do not transfer; measure yours

This is the rule that matters most for a customer, and the kit is blunt about it:

> **Numbers do not transfer between domains or between models. Measure yours.**

The kit's promoted figure is an *agency-domain* number on an *agency-domain*
holdout against a *specific model* on a *specific date*. It is not a prediction of
what your gate will score on your data with your model. It cannot be. Different
data, different model, different day — different number. That is why the kit sells
a **method with receipts**, not a pre-tuned checkpoint with a guaranteed score:
the only honest accuracy number for your deployment is the one you measure on your
own frozen holdout, behind your own variance envelope, with your own sidecar.

The kit makes **no accuracy claim about your data**. Anyone who tells you a
validation gate will hit a specific accuracy on data it has never seen is selling
you a number that does not exist yet. Run the loop in
[MANUAL.md](MANUAL.md), read the [worked example](WORKED-EXAMPLE.md), and
produce your own scorecard.

---

## Commands used in this doctrine

Every command and flag is verified against the kit's CLI.

| Use | Command |
|---|---|
| Attach the evidence boundary to a number | `lens-kit sidecar ckpt.json --profile P --dataset holdout.json --eval-results eval-envelope.json --not-a-claim-of "..."` |
| Measure a real variance envelope (re-baseline) | `lens-kit eval holdout.json --profile P --checkpoint ckpt.json --reruns 3 --fresh-cache` |
| Prove the gate can still fail planted flaws before re-claiming | `lens-kit mutate holdout.json --profile P` |

There is no `re-baseline` command and no `ship` command: re-baselining is an
`eval --reruns --fresh-cache` plus a fresh `sidecar`, and shipping a claim is a
human gate over the receipts — both by design.
