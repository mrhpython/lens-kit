# Lens Stop-Hook Gate for Grok

This integration runs lens-kit's outside-in gate on Grok's finished response.
Grok supplies the frozen candidate as `lastAssistantMessage`. The hook does
not read a session transcript. The validating model is separate from Grok, so
the generator does not grade itself.

## Install

Release install, pinned to the same tag as the core:

```bash
python -m pip install \
  "lens-kit @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0"
python -m pip install \
  "lens-grok-hook @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0#subdirectory=integrations/grok"
```

For a development checkout, replace those commands with
`pip install -e .` and `pip install -e integrations/grok` from the repository
root.

This installs `lens-grok-stop-hook`.

## Configure

Put the command in `~/.grok/hooks/lens-stop.json` (Grok timeout is seconds,
not milliseconds):

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "lens-grok-stop-hook",
            "timeout": 600
          }
        ]
      }
    ]
  }
}
```

Reload with `/hooks` in the Grok TUI. Global files under `~/.grok/hooks/` are
always trusted.

Environment variables:

- `LENS_GROK_PROFILE`: absolute profile YAML path. Default: lens-kit's
  packaged `agency-example` profile. That packaged profile uses a `REPLACE_ME`
  model and is not an operational endpoint.
- `LENS_GROK_FAIL`: `open` (default) allows an UNKNOWN with a visible
  "not validated" warning on stderr; `closed` blocks once, then allows the
  re-stop so the hook cannot wedge the turn.
- `LENS_GROK_RISK`: `internal-low` (default) permits unresolved output after
  one correction only with an explicit `DELIVER_WITH_WARNINGS` notice. Any
  other value fails toward `ESCALATE`.
- `LENS_GROK_SCOPE`: `substantive` (default), `all`, or `off`.
- `LENS_GROK_DOMAIN`: domain passed to lens-kit. Default: `general`.
- `LENS_GROK_CONTEXT`: optional audience/task context.
- The provider credential named by the selected profile, if any.

All adapter examples allow 600 seconds; Grok expresses the ceiling in seconds.
This is a conservative configuration ceiling, not a latency measurement or
completion guarantee. Set the hook timeout to the provider's measured cold
tail. A timeout means the external verdict is UNKNOWN, not PASS.

## Grok payload differences from Claude / Codex

- Input keys are camelCase: `lastAssistantMessage`, `stopHookActive`,
  `hookEventName`. Snake_case is accepted as a fallback.
- Filter `reason == "end_turn"`. Grok also fires Stop on session end
  (`channel_closed` / `shutdown`); those must not spend a gate call.
- Skip when `subagentType` is present. This hook gates the main agent only.
- On Stop, Grok `additionalContext` keeps the agent working. Allow-notes
  (PASS chatter, UNKNOWN, DELIVER_WITH_WARNINGS) go to stderr. Only a
  `{"decision":"block","reason":"..."}` object is written to stdout.

## Behavior

- PASS: disposition `SHIP`; finish normally.
- Non-Rights FAIL on the first stop: disposition `REVISE`; block with
  specific findings.
- FAIL with `stopHookActive=true` and `LENS_GROK_RISK=internal-low`: do not
  block a second time; disposition `DELIVER_WITH_WARNINGS` on stderr.
- FAIL with `stopHookActive=true` in a configured public/high-risk workflow:
  disposition `ESCALATE`; block.
- Provider/profile/runtime failure: UNKNOWN. Fail open by default, or block
  once with `LENS_GROK_FAIL=closed`.
- Rights HALT remains blocked after `stopHookActive=true`. Feedback is
  value-free and does not repeat detected secrets or paths.

The one-correction cap follows Grok's `stopHookActive` field. An allowed
re-stop is not a PASS.

## Test

```bash
PYTHONPATH=integrations/grok python -m pytest integrations/grok/tests
```

The tests stub the gate; they do not call a model or the network.
