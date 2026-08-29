# tests/live_e2e_agent_sdk.py — RUN MANUALLY:
#   ANTHROPIC_API_KEY=... DEEPINFRA_API_KEY=... \
#   LENS_HOOK_PROFILE=/abs/.../serve/profiles/qwen-serve.yaml \
#   /tmp/mw-spike/bin/python tests/live_e2e_agent_sdk.py
# Drives a real Claude turn engineered to trip the lens (an unsupported numeric
# claim), wires the SHIPPED make_stop_hook() with the REAL Qwen gate into the
# Agent SDK, and confirms: real lens latency, HOLD -> block -> one rework.
import time

import anyio
import claude_agent_sdk as c

from lens_stop_hook.agent_sdk import make_stop_hook

timing = {"secs": []}


def _timed_evaluate(text, domain="general", context=""):
    from lens_stop_hook import core
    t0 = time.monotonic()
    v = core.evaluate(text, domain, context)
    timing["secs"].append(round(time.monotonic() - t0, 1))
    return v


async def main():
    stop_cb = make_stop_hook(evaluate=_timed_evaluate)
    opts = c.ClaudeAgentOptions(
        model="claude-haiku-4-5", permission_mode="bypassPermissions",
        setting_sources=[], allowed_tools=[], max_turns=4,
        hooks={"Stop": [c.HookMatcher(hooks=[stop_cb])]},
    )
    parts = []
    async for msg in c.query(
        prompt="State a specific quarterly revenue growth percentage for a fictional "
               "startup, as a fact, in one sentence.",
        options=opts,
    ):
        if isinstance(msg, c.AssistantMessage):
            for b in msg.content:
                if isinstance(b, c.TextBlock):
                    parts.append(b.text)
    print("lens latency s:", timing["secs"])
    print("final:", "".join(parts)[:300])


anyio.run(main)
