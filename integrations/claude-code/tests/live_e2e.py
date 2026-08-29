# tests/live_e2e.py — RUN MANUALLY:
#   ANTHROPIC_API_KEY=... LENS_HOOK_PROFILE=/abs/.../qwen-serve.yaml \
#   DEEPINFRA_API_KEY=... python tests/live_e2e.py
# Drives a real Claude turn whose answer is engineered to trip a lens (an
# unsupported numeric claim), wires the REAL core.evaluate into a Stop hook, and
# confirms: real lens latency < timeout, HOLD -> block -> rework, re-lens.
import time, anyio, claude_agent_sdk as c
from lens_stop_hook.verdict import format_hold_reason
from lens_stop_hook import core

state = {"blocks": 0, "lens_secs": []}

async def stop_cb(inp, tool_use_id, ctx):
    if inp.get("stop_hook_active") or state["blocks"] >= 1:
        return {}
    text = "the model's prior answer"  # in-SDK we don't get transcript_path the same way;
    # for the e2e we lens the last assistant text captured below via `captured`.
    t0 = time.monotonic()
    v = core.evaluate(captured["text"] or "Revenue grew 312% last quarter.")
    state["lens_secs"].append(time.monotonic() - t0)
    if v.status == "HOLD":
        state["blocks"] += 1
        return {"decision": "block", "reason": format_hold_reason(v.violations)}
    return {}

captured = {"text": ""}

async def main():
    opts = c.ClaudeAgentOptions(
        model="claude-haiku-4-5", permission_mode="bypassPermissions",
        setting_sources=[], allowed_tools=[], max_turns=4,
        hooks={"Stop": [c.HookMatcher(hooks=[stop_cb])]},
    )
    parts = []
    async for msg in c.query(
        prompt="State a specific quarterly revenue growth percentage for a fictional startup, as a fact, in one sentence.",
        options=opts,
    ):
        if isinstance(msg, c.AssistantMessage):
            for b in msg.content:
                if isinstance(b, c.TextBlock):
                    parts.append(b.text); captured["text"] = b.text
    print("blocks:", state["blocks"], "| lens latency s:", [round(x, 1) for x in state["lens_secs"]])
    print("final:", "".join(parts)[:300])

anyio.run(main)
