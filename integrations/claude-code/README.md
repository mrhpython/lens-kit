# Lens Stop-Hook Gate (Claude Code)

Runs the Qwen **outside-in** 10-lens gate on Claude Code's finished answer. On a
HOLD it blocks the turn and feeds the violations back as revision guidance,
forcing one rework. The validator is a separate model (Qwen) — Claude never
grades itself.

## Install

```bash
pip install -e .                      # the lens_kit engine, from the repo root
pip install -e integrations/claude-code
```

This installs a `lens-stop-hook` console script, which is what the settings
snippet calls.

## Configure (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `LENS_HOOK_PROFILE` | the profile shipped in `lens_kit` | Absolute path to your own profile. Unset is fine — it resolves to the packaged example, which you should replace with your domain vocabulary and your endpoint. |
| `LENS_HOOK_FAIL` | `open` | `open` = deliver + "not validated" note on lens failure; `closed` = block |
| `LENS_HOOK_REWORKS` | `1` | rework cap (N=1 today) |
| *your provider's key env* | — | Whatever `llm.api_key_env` in your profile names (e.g. `DEEPINFRA_API_KEY`, `OPENAI_API_KEY`). Missing it fails closed at load. |

Put these in your shell env or the `settings.json` hook `env`.

### Sizing the `timeout`

The snippet uses `120000` ms. **Size it to the tail, not the typical.** The gate
runs the lens set over your whole answer, so cost scales with answer length, and
long answers are exactly the ones worth gating. A hook `timeout` is a ceiling,
not a delay — an oversized one costs nothing when the gate is fast.

This matters more than it looks, because **a timeout fails open silently**: the
hook reports "not validated" and the turn proceeds. A gate that times out on
every substantial answer is indistinguishable from no gate at all, and it will
not tell you. Measure the gate on your own provider and text lengths before
trusting a smaller number.

## Enable

Merge `settings-snippet.json` into `~/.claude/settings.json` (or a project
`.claude/settings.json`).

## Behavior

| Lens result | Action |
|-------------|--------|
| PASS | answer finishes |
| HOLD (first stop) | turn blocked; violations fed back; Claude reworks once |
| HOLD (after rework) | answer finishes with a "still flagged" note |
| lens unavailable | fail-open: finishes + "not validated" note (or `closed`: blocks) |

## Use with the Claude Agent SDK

Customers building agents on `claude-agent-sdk` can add the same outside-in lens
gate to their agent's final answer:

```bash
pip install lens-stop-hook[agent-sdk]
```

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
forcing one rework; a second HOLD downgrades to a warning (N=1 cap). The blocking
Qwen gate runs off the event loop via `asyncio.to_thread`, so it does not stall the
agent runtime. Same env config as the Claude Code hook (`LENS_HOOK_FAIL`,
`LENS_HOOK_PROFILE`, `LENS_HOOK_DAILY_CAP`).

Options: `make_stop_hook(domain="finance", context="...", extract_text=fn)`. Pass
`extract_text` to feed text you captured from the message stream instead of the
transcript file.
