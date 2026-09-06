# Lens Stop Observer for Grok

This integration runs lens-kit's outside-in gate when a Grok Stop event includes
the finished response as `lastAssistantMessage`. It does not read a session
transcript. The validating model is separate from Grok, so the generator does
not grade itself.

Grok documents Stop as a passive event: stdout is ignored and only
`PreToolUse` can block. This adapter therefore reports findings to stderr for
logs or a separate publication control to consume; it cannot stop Grok from
ending the turn. See the [official hook contract](https://docs.x.ai/build/features/hooks).

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

Reload with `/hooks` in the Grok TUI. The global hooks directory is one of the
documented discovery locations. Project hooks require `/hooks-trust` or the
`--trust` launch option; see the
[official hook contract](https://docs.x.ai/build/features/hooks).

Environment variables:

- `LENS_GROK_PROFILE`: absolute profile YAML path. Default: lens-kit's
  packaged `agency-example` profile. That packaged profile uses a `REPLACE_ME`
  model and is not an operational endpoint.
- `LENS_GROK_FAIL`: `open` (default) or `closed`. This changes the requested
  diagnostic disposition only; a passive Stop hook cannot enforce it.
- `LENS_GROK_RISK`: `internal-low` (default) or a stricter classification.
  This changes the reported disposition only; use a separate publication
  control for enforcement.
- `LENS_GROK_SCOPE`: `substantive` (default), `all`, or `off`.
- `LENS_GROK_DOMAIN`: domain passed to lens-kit. Default: `general`.
- `LENS_GROK_CONTEXT`: optional audience/task context.
- The provider credential named by the selected profile, if any.

All adapter examples allow 600 seconds; Grok expresses the ceiling in seconds.
This is a conservative configuration ceiling, not a latency measurement or
completion guarantee. Set the hook timeout to the provider's measured cold
tail. A timeout means the external verdict is UNKNOWN, not PASS.

## Grok payload handling

- Input keys are camelCase: `lastAssistantMessage`, `stopHookActive`,
  `hookEventName`. Snake_case is accepted as a fallback.
- If an event has a `reason` field, the adapter validates only `end_turn`.
- Skip when `subagentType` is present. This hook gates the main agent only.
- PASS is silent. FAIL, HALT and UNKNOWN diagnostics go to stderr. Stdout stays
  empty because Grok ignores output from passive Stop hooks.

## Behavior

- PASS: no diagnostic.
- Non-Rights FAIL: report `REVISE` or, after one correction, the configured
  warning/escalation disposition.
- Provider/profile/runtime failure: report UNKNOWN.
- Rights HALT: report value-free guidance without repeating detected secrets
  or paths.

These are observations, not enforced decisions. Put an independent release,
deployment, or send control after the hook if a FAIL, HALT, or UNKNOWN result
must prevent publication.

## Test

```bash
PYTHONPATH=integrations/grok python -m pytest integrations/grok/tests
```

The tests stub the gate; they do not call a model or the network.
