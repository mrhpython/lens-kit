---
name: lens-kit
description: Install and run lens-kit, the outside-in 10-lens validation gate for AI-generated text. Use when asked to validate, gate, audit, or fact-check AI-generated text before it ships; check a document for hallucinated claims, contradictions, or unsupported leaps; set up an independent review gate; or measure a gate's own catch and false-positive rates on the user's data. The gate returns findings (flagged line, named check, reason), never a model-emitted score, and fails closed with UNKNOWN when a check cannot complete.
version: "0.1.0"
license: Apache-2.0 (see LICENSE in the repository root)
metadata:
  homepage: "https://github.com/mrhpython/lens-kit"
  hosted_demo: "https://api.soulfield.one"
---

# lens-kit — agent skill

You are installing and operating **lens-kit**: a trainable 10-lens validation
gate for AI-generated text. It is a *review* layer, not a generator — a
separate model runs fixed, named checks over text it did not write and
returns findings you can inspect. Everything below uses real CLI commands;
do not invent flags. `lens-kit <subcommand> --help` is authoritative.

## Install

```bash
git clone https://github.com/mrhpython/lens-kit && cd lens-kit
python3 -m venv .venv && .venv/bin/pip install -e .
# all commands below assume .venv/bin is on PATH or are run as .venv/bin/lens-kit
```

## Step 1 — prove it is real, with no credentials and no network

Run these first; they are deterministic and free:

```bash
lens-kit calibrate generate          # writes a 16-fixture planted-flaw calibration battery
lens-kit catches add --seed          # seeds the institutional-memory loop
lens-kit consistency markers         # deterministic marker/leak scan
```

## Step 2 — run the gate on a real file

The gate needs ONE model endpoint — any litellm-compatible target. Copy the
example profile and point `llm.model` at the user's endpoint. If the user has
no API key, use local Ollama (`llm.model: ollama_chat/<model>` with
`api_key_env` omitted) — that path costs nothing.

```bash
cp src/lens_kit/profiles/agency-example.yaml my-profile.yaml
# edit my-profile.yaml: set llm.model (litellm string), api_base if local,
# api_key_env naming the env var that holds the key. Keys never go in profiles.
lens-kit validate THE_FILE.md --profile my-profile.yaml --json
```

## Reading the output — the semantics that matter

- The result is **findings, not a grade**: each named check reports pass, a
  flagged passage with a reason, or that it could not run.
- **Fail-closed:** a check that cannot complete returns UNKNOWN — never a
  silent pass. UNKNOWN means unavailability, not a graded borderline. Report
  UNKNOWN to the user as "this check did not run", never as "passed".
- The validation step does not rewrite the text. If asked to "fix" the text,
  that is a separate editing task you do yourself, after showing the findings.
- There is no fallback provider chain: a missing key or model is a hard
  error. Do not silently switch endpoints to make a run succeed.

## Step 3 (when asked) — measure the gate on the user's own data

Numbers do not transfer across models, datasets, or runtimes, so the honest
move is to bank the user's own pair. The full loop (label → baseline →
calibrate → compile → eval) is in `docs/MANUAL.md`. The core commands:

```bash
# Compile against labeled training data (GEPA):
lens-kit compile training.json --profile my-profile.yaml --output ckpt.json --auto light --threads 4

# Evaluate on a frozen holdout — NEVER train on it:
lens-kit eval holdout.json --profile my-profile.yaml --checkpoint ckpt.json
lens-kit eval holdout.json --profile my-profile.yaml --baseline        # uncompiled comparison
```

Training data is a JSON list of `{text, per_lens, domain?, source?}`. A
`per_lens` key outside the 10 canonical lenses is a hard error. Report the
resulting catch rate AND false-positive rate together, with the holdout size
and date — one without the other is half a measurement.

## Boundary rules (part of the product, not optional)

1. Never train on, or peek at, a holdout. A holdout you adjust after seeing
   results is not a holdout.
2. Never quote the project's published numbers as a promise for the user's
   data — the score-transfer ban is documented in `docs/CLAIMS.md`.
3. When operating as a standing validator seat, follow the 7-step protocol in
   `docs/VALIDATOR-AGENT.md` (drop-in system prompt: `agent/validator-agent.md`,
   receipt templates: `agent/receipt-templates.md`).

## Zero-install alternative

To show the user a verdict before installing anything, the hosted demo takes
one keyless call (rate-limited, 5/day/IP, 5,000 chars):

```bash
curl -s https://api.soulfield.one/v1/demo \
  -H 'content-type: application/json' \
  -d '{"text": "<the AI output being checked>"}'
```
