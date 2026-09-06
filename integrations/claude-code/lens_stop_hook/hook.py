# lens_stop_hook/hook.py
"""Claude Code Stop-hook driver: stdin StopHookInput -> gate loop -> SyncHookJSONOutput.

Gate loop (N=1 falls out of stop_hook_active):
  PASS                          -> allow
  HOLD, first stop              -> block + revision guidance
  HOLD, re-stop, internal-low   -> allow + "still flagged" note
  HOLD, re-stop, public/high    -> block + escalate
  Rights HOLD                   -> block every time
  UNAVAILABLE, fail-open        -> allow + "not validated" note
  UNAVAILABLE, fail-closed,1st  -> block "not validated"
  UNAVAILABLE, public/high      -> block + escalate
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


_SAFE_RIGHTS_REASON = (
    "Lens verdict HALT; disposition HALT. Sensitive or restricted content was "
    "detected. Remove or redact it before finishing."
)


def _fail_closed() -> bool:
    return os.environ.get("LENS_HOOK_FAIL", "open").strip().lower() == "closed"


def _requires_escalation() -> bool:
    """Only explicitly classified internal-low work may finish with warnings."""
    return os.environ.get("LENS_HOOK_RISK", "internal-low").strip().lower() != "internal-low"


def _has_rights_hold(violations: list) -> bool:
    return any(str(item.get("lens", "")).casefold() == "rights" for item in violations)


def _event_text(event: dict) -> str:
    """Read the current Stop response, falling back for older payloads."""
    message = event.get("last_assistant_message")
    if isinstance(message, str) and message.strip():
        return message
    return final_assistant_text(event.get("transcript_path", ""))


def decide(event: dict, evaluate, text: str) -> dict:
    """Return the SyncHookJSONOutput dict for one Stop event."""
    if not text or not text.strip():
        return {}  # nothing to lens

    active = bool(event.get("stop_hook_active"))
    v = evaluate(text)

    if v.status == "PASS":
        return {}

    if v.status == "HOLD":
        if _has_rights_hold(v.violations):
            return {"decision": "block", "reason": _SAFE_RIGHTS_REASON}
        if not active:
            return {"decision": "block", "reason": format_hold_reason(v.violations)}
        if _requires_escalation():
            return {
                "decision": "block",
                "reason": "Lens verdict FAIL; disposition ESCALATE. This workflow is "
                          "configured as public/high-risk, so the unresolved candidate "
                          "must not ship.",
            }
        return {"systemMessage": "⚠ Lens: still flagged after 1 rework — "
                                 + format_hold_reason(v.violations)}

    # UNAVAILABLE
    if _requires_escalation():
        return {
            "decision": "block",
            "reason": "Lens verdict UNKNOWN; disposition ESCALATE. This workflow is "
                      "configured as public/high-risk, so unvalidated output must not ship.",
        }
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
        text = _event_text(event)
        out = decide(event, _default_evaluate, text)
    except Exception:  # noqa: BLE001 — fail per policy
        if _requires_escalation():
            out = {"decision": "block", "reason":
                   "Lens verdict UNKNOWN; disposition ESCALATE. This workflow is "
                   "configured as public/high-risk, so unvalidated output must not ship."}
        elif _fail_closed() and not event.get("stop_hook_active"):
            out = {"decision": "block", "reason": "Lens hook error — not validated."}
        else:
            out = {"systemMessage": "⚠ Lens hook error — not validated"}
    if out:
        sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
