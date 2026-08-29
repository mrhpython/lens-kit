"""Agent-SDK Stop-hook helper. make_stop_hook() -> async Stop-hook callback.

Wraps build #1's gate loop for customers building agents on claude-agent-sdk:

    from lens_stop_hook.agent_sdk import make_stop_hook
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    opts = ClaudeAgentOptions(hooks={"Stop": [HookMatcher(hooks=[make_stop_hook()])]})

The blocking ~16s Qwen gate runs off the event loop via asyncio.to_thread so it
never stalls the agent's async runtime. All gate logic is reused from .hook.decide;
this module adds no new lens-touching code.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from .core import evaluate as _default_evaluate
from .hook import _fail_closed, decide
from .transcript import final_assistant_text
from .verdict import Verdict


def _default_extract(inp: dict) -> str:
    """Final assistant text from a StopHookInput.

    The Agent SDK supplies the finished answer directly as the string
    ``last_assistant_message`` (verified, claude_agent_sdk 0.2.104). The SDK's
    own transcript JSONL uses a different record shape than the Claude Code CLI
    transcript, so ``final_assistant_text`` does NOT parse it — we use it only as
    a fallback when ``last_assistant_message`` is absent.
    """
    lam = inp.get("last_assistant_message")
    if isinstance(lam, str) and lam.strip():
        return lam
    return final_assistant_text(inp.get("transcript_path", ""))


def make_stop_hook(
    *,
    evaluate: Callable[..., Verdict] = _default_evaluate,
    extract_text: Callable[[dict], str] | None = None,
    domain: str = "general",
    context: str = "",
) -> Callable[..., Awaitable[dict]]:
    """Return an async Stop-hook callback for ClaudeAgentOptions(hooks=...).

    evaluate     : the lens call (Qwen, outside-in). Injectable for tests.
    extract_text : (StopHookInput) -> str. Default reads transcript_path; override
                   to feed text captured from the message stream.
    domain/context : forwarded to evaluate.
    """
    extract = extract_text or _default_extract

    def _run_gate(inp: dict) -> dict:
        text = extract(inp)
        return decide(inp, lambda t: evaluate(t, domain, context), text)

    async def stop_cb(inp: dict, tool_use_id=None, ctx=None) -> dict:
        try:
            return await asyncio.to_thread(_run_gate, inp)
        except Exception as e:  # noqa: BLE001 — fail per policy, never wedge
            if _fail_closed() and not inp.get("stop_hook_active"):
                return {"decision": "block",
                        "reason": f"Lens hook error — not validated ({e})."}
            return {"systemMessage": f"⚠ Lens hook error — not validated ({e})"}

    return stop_cb
