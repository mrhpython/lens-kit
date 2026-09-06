# lens-kit v0.1.0

The first packaged GitHub release includes the core outside-in validation gate,
correction-request adapters for Codex and Claude Code / Claude Agent SDK, and
an observational Grok Stop adapter. Grok documents Stop as passive, so its
adapter reports findings but cannot block the turn.

Install the core once:

```bash
python -m pip install \
  "lens-kit @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0"
```

Then install the adapter for your coding agent:

```bash
# Codex
python -m pip install \
  "lens-codex-hook @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0#subdirectory=integrations/codex"

# Claude Code / Claude Agent SDK
python -m pip install \
  "lens-stop-hook @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0#subdirectory=integrations/claude-code"

# Grok (observational Stop validation)
python -m pip install \
  "lens-grok-hook @ git+https://github.com/mrhpython/lens-kit.git@v0.1.0#subdirectory=integrations/grok"
```

Agent-specific configuration:

- [Codex](https://github.com/mrhpython/lens-kit/tree/v0.1.0/integrations/codex)
- [Claude Code / Claude Agent SDK](https://github.com/mrhpython/lens-kit/tree/v0.1.0/integrations/claude-code)
- [Grok](https://github.com/mrhpython/lens-kit/tree/v0.1.0/integrations/grok)

The attached release assets contain a wheel and source archive for each of the
four packages, plus `SHA256SUMS` for verification.
