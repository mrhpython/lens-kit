# lens_stop_hook/hook.py
"""Claude Code Stop-hook driver: stdin StopHookInput -> gate loop -> SyncHookJSONOutput.

Gate loop (N=1 falls out of stop_hook_active):
  PASS                          -> allow
  HOLD, first stop              -> block + revision guidance
  HOLD, re-stop (rework done)   -> allow + "still flagged" note (no second block)
  UNAVAILABLE, fail-open        -> allow + "not validated" note
  UNAVAILABLE, fail-closed,1st  -> block "not validated"
  UNAVAILABLE, fail-closed,re   -> allow + note (never wedge past N=1)
  empty/no answer text          -> allow
Any uncaught error -> fail per policy (default open = allow).
"""
from __future__ import annotations

import json
import os
import sys

from .core import evaluate as _default_evaluate
from .transcript import final_assistant_text
from .verdict import format_hold_reason


def _fail_closed() -> bool:
    return os.environ.get("LENS_HOOK_FAIL", "open").strip().lower() == "closed"


def decide(event: dict, evaluate, text: str) -> dict:
    """Return the SyncHookJSONOutput dict for one Stop event."""
    if not text or not text.strip():
        return {}  # nothing to lens

    active = bool(event.get("stop_hook_active"))
    v = evaluate(text)

    if v.status == "PASS":
        return {}

    if v.status == "HOLD":
        if not active:
            return {"decision": "block", "reason": format_hold_reason(v.violations)}
        return {"systemMessage": "⚠ Lens: still flagged after 1 rework — "
                                 + format_hold_reason(v.violations)}

    # UNAVAILABLE
    if _fail_closed() and not active:
        return {"decision": "block",
                "reason": f"Output not validated — lens unavailable ({v.note}). "
                          "Re-finish; if it persists the gate will let it through."}
    return {"systemMessage": f"⚠ Lens unavailable — answer not validated ({v.note})"}


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return 0  # can't even read the event — never wedge
    try:
        text = final_assistant_text(event.get("transcript_path", ""))
        out = decide(event, _default_evaluate, text)
    except Exception as e:  # noqa: BLE001 — fail per policy
        if _fail_closed() and not event.get("stop_hook_active"):
            out = {"decision": "block", "reason": f"Lens hook error — not validated ({e})."}
        else:
            out = {"systemMessage": f"⚠ Lens hook error — not validated ({e})"}
    if out:
        sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
