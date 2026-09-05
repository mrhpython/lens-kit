# Lens Stop-Hook Gate for Codex

This integration runs lens-kit's outside-in gate on Codex's finished response.
Codex supplies the frozen candidate as `last_assistant_message`; the hook does
not read or reconstruct a session transcript. The validating model is separate
from Codex, so the generator does not grade itself.

## Install

Release install, pinned to the same tag as the core:

```bash
python -m pip install \
  "lens-kit @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0"
python -m pip install \
  "lens-codex-hook @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0#subdirectory=integrations/codex"
```

For a development checkout, replace those commands with
`pip install -e .` and `pip install -e integrations/codex` from the repository
root.

This installs `lens-codex-stop-hook`.

## Configure

Add the command to the existing `Stop` hook list in `~/.codex/hooks.json`. Do
not replace other Stop commands:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/to/lens-codex-stop-hook",
            "timeout": 600
          }
        ]
      }
    ]
  }
}
```

Use Codex's `/hooks` command to inspect and trust the local hook after changing
the configuration. Codex records trust against the command and configuration
content, so a later edit can require trust again.

Environment variables:

- `LENS_CODEX_PROFILE`: absolute profile YAML path. Default: lens-kit's
  packaged `agency-example` profile. That packaged profile intentionally uses
  a `REPLACE_ME` model and is not an operational endpoint; set this variable to
  a configured profile before expecting a completed verdict.
- `LENS_CODEX_FAIL`: `open` (default) allows an UNKNOWN with a visible
  "not validated" warning; `closed` blocks once, then allows the re-stop so
  the hook cannot wedge the turn.
- `LENS_CODEX_RISK`: `internal-low` (default) permits unresolved output after
  one correction only with an explicit `DELIVER_WITH_WARNINGS` notice. Any
  other value (for example `public`, `customer-facing`, `high-stakes`,
  `security-sensitive`, or `irreversible`) fails toward `ESCALATE`: unresolved
  FAIL or UNKNOWN output is blocked and must not ship. The adapter does not
  guess risk from prose.
- `LENS_CODEX_SCOPE`: `substantive` (default), `all`, or `off`.
  `substantive` skips only an exact finite set of acknowledgement phrases such
  as `Thanks!` or `Understood.` after punctuation/space normalization. It is a
  cost/latency control, not a semantic classifier. Any other response,
  including a short command or factual status, reaches the gate. `all` also
  validates acknowledgements. Unknown values validate rather than skip.
- `LENS_CODEX_DOMAIN`: domain passed to lens-kit. Default: `general`.
- `LENS_CODEX_CONTEXT`: optional audience/task context passed to lens-kit.
- The provider credential named by the selected profile, if any.

All adapter examples allow 600 seconds; Codex expresses the ceiling in seconds.
This is a conservative configuration ceiling, not a latency measurement or
completion guarantee. Set it to the provider's measured cold tail. A timeout
means the external verdict is UNKNOWN, not PASS.

## Behavior

- PASS: disposition `SHIP`; finish normally.
- Non-Rights FAIL, or PASS with actionable violations, on the first stop:
  disposition `REVISE`; block and return specific findings as revision guidance.
- FAIL with `stop_hook_active=true` and `LENS_CODEX_RISK=internal-low`: do not
  block a second time; disposition `DELIVER_WITH_WARNINGS` and preserve the
  remaining findings.
- FAIL with `stop_hook_active=true` in a configured public/high-risk workflow:
  disposition `ESCALATE`; block the unresolved candidate.
- Provider/profile/runtime failure: label the result UNKNOWN. Fail open with a
  warning by default for internal low-risk work, or block once with
  `LENS_CODEX_FAIL=closed`. Public/high-risk UNKNOWN is `ESCALATE` and blocked.
- Rights HALT is the only unconditional Lens hard stop and remains blocked after
  `stop_hook_active=true`. Its feedback is deliberately value-free: it tells
  Codex to remove or redact sensitive content without repeating the detected value. Runtime
  failures expose only a generic error category, never raw exception text or
  local paths.
- Empty, malformed, or non-object hook input: exit zero without output so the
  integration cannot wedge Codex.

The one-correction cap follows Codex's own `stop_hook_active` field. The adapter
does not store conversation state and does not claim that an allowed re-stop
passed validation. Candidate verdict and workflow disposition remain separate.

## Test

```bash
PYTHONPATH=integrations/codex python -m pytest integrations/codex/tests
```

The tests stub the gate; they do not call a model or the network.
