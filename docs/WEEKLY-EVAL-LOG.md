# Weekly holdout eval — the drift log

A published catch rate is a claim about a day. This is the log that checks whether it is
still true, every week, against a holdout that does not move.

It runs on a timer. It re-scores the same 113 hand-labelled rows against a fixed anchor
and writes one line: `OK` inside the band, `DRIFT` outside. Nobody decides afterwards
whether the week went well.

## Read the tier first. The two number pairs are not interchangeable.

We publish two pairs, measured on **the same 113 rows**, through **two different paths**.
They are not the same number and must never be averaged, swapped, or quoted for each
other.

| | Path | Catch | False positive | Where it appears |
|---|---|---|---|---|
| **API tier** | `POST /v1/validate` — the exact door a customer calls, halt semantics and rate limits included | **0.8966** | **0.1952** | The homepage pair, **89.7% / 19.5%** |
| **Harness tier** | the in-process eval rig that ships in this kit | **0.9333** | **0.2017** | The anchor in this log |

**The rig scores better than the product, and the product's number is the one on our
homepage.** Through the API the gate is quieter: it catches 3.7 points fewer violations,
and flags 0.65 points fewer clean rows, because the serving layer converts some
violations to warnings. A number measured on a rig describes the rig — so the rig's
number stays here, in a log about drift, and never migrates into a sentence about what
you would get.

If you ever see 93.3% quoted as what the product does, that is this file being misread.

## The anchor and the band

- **Anchor:** `eval-v4gold-20260810` — catch `0.9333`, fp `0.2017`, harness tier.
- **Band:** ±0.03 on either metric. Observed run-to-run noise is about 0.011, so the band
  carries roughly 3× headroom. Inside it, a wobble is noise; outside it, something moved.
- **Anchor discipline:** the anchor changes only when a pre-registered change ships and
  its result is ratified. It is never nudged to match a week we liked. Labels were last
  re-adjudicated 2026-08-10 under a written rule, which is why pre-v4 numbers are void
  rather than comparable.
- A run also fails on an error rate above 0.05, so a broken week reads as broken rather
  than as a good score on the rows that happened to complete.

## The log

| Date | Verdict | Catch (Δ) | False positive (Δ) | Errors | Anchor |
|---|---|---|---|---|---|
| 2026-08-10 | OK | 0.9255 (−0.0107) | 0.1948 (−0.0022) | 0.0 | eval-c3-holdout-20260809 |
| 2026-08-20 | OK | 0.9333 (+0.0000) | 0.2082 (+0.0065) | 0.0 | eval-v4gold-20260810 |
| 2026-08-24 | OK | 0.9444 (+0.0111) | 0.2060 (+0.0043) | 0.0 | eval-v4gold-20260810 |

Three rows. The 2026-08-10 row was scored against the previous anchor, immediately before
the label re-adjudication; it is kept because deleting a row that used an older anchor is
how a log becomes a highlight reel.

## What this proves, and what it does not

**Does:** that the pair we publish is re-measured on a schedule against a frozen holdout
and a band written down in advance, and that three consecutive weeks landed inside it.

**Does not:** anything about your data. These are our rows, hand-labelled by us, and a
detector's numbers do not transfer across models, datasets or runtimes. Three rows is a
short log — it says the gate has not drifted since the anchor was set, not that it never
will. And a holdout this size cannot resolve small per-lens differences; the band exists
because we measured the noise, not because 0.03 is meaningful.

It also proves nothing about any week that has not run yet. That is the point of leaving
the log where you can check it.

## Run the same thing on your own bank

The harness is in this kit. Label your own rows, set your own anchor from a run you
trust, pick a band from your own observed noise rather than from ours, and put it on a
timer. The number you get will not be ours and should not be — the denominator that
matters is yours.

The discipline transfers even where the number does not: score against rows you labelled
before you saw the results, fix the band in advance, keep the rows that went against you,
and never let a rig's number stand in for what your users actually call.
