# Lens Stop-Hook Gate (Claude Code)

Runs the Qwen **outside-in** 10-lens gate on Claude Code's finished answer. On a
non-Rights HOLD it blocks the turn and feeds the violations back for one bounded
rework. A Rights HOLD remains blocked until cleared. The validator is a separate
model (Qwen) — Claude never grades itself.

## Install

Release install, pinned to the same tag as the core:

```bash
python -m pip install \
  "lens-kit @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0"
python -m pip install \
  "lens-stop-hook @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0#subdirectory=integrations/claude-code"
```

For a development checkout, replace those commands with
`pip install -e .` and `pip install -e integrations/claude-code` from the
repository root.

This installs a `lens-stop-hook` console script, which is what the settings
snippet calls.

## Configure (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `LENS_HOOK_PROFILE` | the profile shipped in `lens_kit` | Absolute path to your own profile. Unset is fine — it resolves to the packaged example, which you should replace with your domain vocabulary and your endpoint. |
| `LENS_HOOK_FAIL` | `open` | `open` = deliver + "not validated" note on lens failure; `closed` = block |
| `LENS_HOOK_RISK` | `internal-low` | Only `internal-low` permits delivery with warnings after one correction. Set `public`, `customer-facing`, `high-stakes`, `security-sensitive`, or `irreversible` to block unresolved FAIL and UNKNOWN output with `ESCALATE`. Unknown values also escalate. |
| *your provider's key env* | — | Whatever `llm.api_key_env` in your profile names (e.g. `DEEPINFRA_API_KEY`, `OPENAI_API_KEY`). Missing it fails closed at load. |

Put these in your shell env or the `settings.json` hook `env`.

### Sizing the `timeout`

The snippet uses `120000` ms. **Size it to the tail, not the typical.** The gate
runs the lens set over your whole answer, so cost scales with answer length, and
long answers are exactly the ones worth gating. A hook `timeout` is a ceiling,
not a delay — an oversized one costs nothing when the gate is fast.

This matters more than it looks. For the default `internal-low` classification,
a timeout fails open with a visible "not validated" message. For public/high-risk
work it remains blocked with disposition `ESCALATE`. Measure the gate on your own
provider and text lengths before trusting a smaller number.

## Enable

Merge `settings-snippet.json` into `~/.claude/settings.json` (or a project
`.claude/settings.json`).

## Behavior

| Lens result | Action |
|-------------|--------|
| PASS | answer finishes |
| HOLD (first stop) | turn blocked; violations fed back; Claude reworks once |
| non-Rights HOLD after rework, `internal-low` | answer finishes with a "still flagged" note |
| non-Rights HOLD after rework, public/high-risk | remains blocked with disposition `ESCALATE` |
| Rights HOLD | remains blocked with disposition `HALT`; sensitive values are not repeated |
| lens unavailable, `internal-low` | fail-open: finishes + "not validated" note (or `closed`: blocks once) |
| lens unavailable, public/high-risk | remains blocked with verdict `UNKNOWN` and disposition `ESCALATE` |

## Use with the Claude Agent SDK

Customers building agents on `claude-agent-sdk` can add the same outside-in lens
gate to their agent's final answer:

```bash
python -m pip install "claude-agent-sdk>=0.2"
```

Install the tagged adapter above first. The GitHub release wheel exposes the
same `agent-sdk` optional extra.

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
from lens_stop_hook.agent_sdk import make_stop_hook

opts = ClaudeAgentOptions(
    hooks={"Stop": [HookMatcher(hooks=[make_stop_hook()])]},
)
```

`make_stop_hook()` lenses the agent's final assistant message (the SDK supplies it
directly as `last_assistant_message`; falls back to the transcript if absent). On a
lens HOLD it returns `{"decision": "block", "reason": ...}`,
forcing one rework. A second non-Rights HOLD downgrades to a warning only for
`internal-low` work. The blocking
Qwen gate runs off the event loop via `asyncio.to_thread`, so it does not stall the
agent runtime. Same env config as the Claude Code hook (`LENS_HOOK_FAIL`,
`LENS_HOOK_RISK`, `LENS_HOOK_PROFILE`, `LENS_HOOK_DAILY_CAP`). Rights HOLD always
remains blocked; public/high-risk unresolved output escalates instead of downgrading.

Options: `make_stop_hook(domain="finance", context="...", extract_text=fn)`. Pass
`extract_text` to feed text you captured from the message stream instead of the
transcript file.
