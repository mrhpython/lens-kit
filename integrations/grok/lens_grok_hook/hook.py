"""Grok Stop observer: finished assistant text -> outside-in lens verdict.

Grok supplies the finished response as ``lastAssistantMessage`` (camelCase).
The hook validates that frozen text with lens-kit and maps the outcomes to
an operator-visible diagnostic:

* PASS: emit nothing.
* FAIL/HALT/UNKNOWN: emit the requested disposition and safe guidance on
  stderr for logs or a separate publication control to consume.

Grok Stop hooks are passive: their stdout and exit status cannot block the
turn. This adapter therefore makes no containment claim.

Grok-specific wiring this adapter must get right:

* Skip when ``subagentType`` is present (main-agent Stop only).
* ``main()`` writes diagnostics to stderr and always leaves stdout empty.
* Accept snake_case Codex/Claude keys as a compatibility fallback.

Malformed input exits successfully without output because a broken hook must
not wedge Grok before it can report the integration problem.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable


_SAFE_RIGHTS_ISSUE = (
    "Sensitive or restricted content was detected. Remove or redact it before finishing."
)
_SAFE_UNKNOWN_NOTES = {
    "hook runtime error",
    "invalid validation result",
    "profile configuration error",
    "provider or validation I/O error",
    "validation runtime error",
    "validation timeout",
    "validation unavailable",
}


@dataclass(frozen=True)
class Verdict:
    """One outside-in gate result.

    ``UNKNOWN`` means the external gate did not complete. It is intentionally
    distinct from ``PASS``.
    """

    status: str  # "PASS" | "FAIL" | "HALT" | "UNKNOWN"
    violations: list[dict] = field(default_factory=list)
    note: str = ""

    @classmethod
    def passed(cls) -> "Verdict":
        return cls("PASS")

    @classmethod
    def failed(cls, violations: list[dict]) -> "Verdict":
        return cls("FAIL", violations=list(violations))

    @classmethod
    def halted(cls, violations: list[dict]) -> "Verdict":
        return cls("HALT", violations=list(violations))

    @classmethod
    def unknown(cls, note: str) -> "Verdict":
        return cls("UNKNOWN", note=note)


@lru_cache(maxsize=1)
def _runtime():
    """Load the selected profile and gate lazily, once per hook process."""

    from lens_kit import LensGate, Profile, builtin_profile_path

    profile_path = os.environ.get("LENS_GROK_PROFILE") or str(builtin_profile_path())
    profile = Profile.load(profile_path)
    return profile, LensGate(profile=profile, parallel=True)


def lm_context(profile):
    """Imported lazily so decide() can run without lens_kit installed."""

    from lens_kit import lm_context as _lm_context

    return _lm_context(profile)


def evaluate(text: str, domain: str = "general", context: str = "") -> Verdict:
    """Run lens-kit; convert every load/provider/runtime error to UNKNOWN."""

    try:
        profile, gate = _runtime()
        with lm_context(profile):
            result = gate(text=text, domain=domain, context=context)
        record = result.to_dict()
    except Exception as exc:  # noqa: BLE001 - gate failure is UNKNOWN, never PASS
        return Verdict.unknown(_safe_validation_exception_note(exc))

    violations = record.get("violations") or []
    if record.get("passed") and not record.get("halted") and not violations:
        return Verdict.passed()

    if record.get("halted"):
        return Verdict.halted([{
            "lens": "rights",
            "severity": "critical",
            "issue": _SAFE_RIGHTS_ISSUE,
        }])
    return Verdict.failed([_safe_violation(item) for item in violations])


def _safe_validation_exception_note(exc: Exception) -> str:
    """Classify an exception without exposing its message, paths, or values."""

    # Name-match ConfigError so this path works when lens_kit is not installed.
    if type(exc).__name__ == "ConfigError":
        return "profile configuration error"
    if isinstance(exc, TimeoutError):
        return "validation timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return "provider or validation I/O error"
    return "validation runtime error"


def _safe_violation(item: dict) -> dict:
    """Remove raw Rights content while retaining non-Rights lens guidance."""

    if str(item.get("lens", "")).casefold() == "rights":
        return {
            "lens": "rights",
            "severity": item.get("severity", "critical"),
            "issue": _SAFE_RIGHTS_ISSUE,
        }
    return dict(item)


_SIMPLE_ACKS = {
    "acknowledged",
    "done",
    "got it",
    "i understand",
    "no",
    "noted",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "understood",
    "yes",
    "you are welcome",
    "you're welcome",
}


def _normalized_ack(text: str) -> str:
    folded = re.sub(r"[^\w']+", " ", text.casefold(), flags=re.UNICODE)
    return " ".join(folded.split())


def should_validate(text: str) -> bool:
    """Apply the deterministic cost/latency scope control.

    ``LENS_GROK_SCOPE`` values:

    * ``substantive`` (default): skip only exact phrases in ``_SIMPLE_ACKS``;
    * ``all``: validate every non-empty response;
    * ``off``: disable gate calls while leaving the hook installed.

    Unknown values fail safely toward validation.
    """

    if not text or not text.strip():
        return False
    scope = os.environ.get("LENS_GROK_SCOPE", "substantive").strip().casefold()
    if scope == "off":
        return False
    if scope == "substantive" and _normalized_ack(text) in _SIMPLE_ACKS:
        return False
    return True


def _fail_closed() -> bool:
    return os.environ.get("LENS_GROK_FAIL", "open").strip().casefold() == "closed"


def _requires_escalation() -> bool:
    """Return whether unresolved/UNKNOWN output must not ship.

    The hook does not guess risk from prose. Operators classify the workflow
    with ``LENS_GROK_RISK``. Unknown values fail toward escalation; only
    ``internal-low`` enables delivery with warnings.
    """

    risk = os.environ.get("LENS_GROK_RISK", "internal-low").strip().casefold()
    return risk != "internal-low"


def _candidate_text(event: dict) -> str | None:
    for key in ("lastAssistantMessage", "last_assistant_message"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return None


def _stop_active(event: dict) -> bool:
    return bool(event.get("stopHookActive") or event.get("stop_hook_active"))


def _is_end_turn(event: dict) -> bool:
    """When a reason is supplied, gate only explicit turn completions.

    A missing ``reason`` is treated as end-turn so Codex/Claude-shaped test
    payloads and older envelopes still reach the gate.
    """

    reason = event.get("reason")
    if reason is None:
        return True
    return reason == "end_turn"


def _format_fail_reason(violations: list[dict]) -> str:
    if not violations:
        return (
            "Lens verdict FAIL; disposition REVISE. Revise the response before "
            "finishing because it did not pass validation."
        )
    safe_violations = [_safe_violation(item) for item in violations]
    clauses = "; ".join(
        f"{item.get('lens', '?')}: {str(item.get('issue', '')).strip()}"
        for item in safe_violations
    )
    return (
        "Lens verdict FAIL; disposition REVISE. Fix these outside-in findings, "
        f"then finish the response: {clauses}"
    )


def _halt_reason() -> str:
    return "Lens verdict HALT; disposition HALT. " + _SAFE_RIGHTS_ISSUE


def _escalation_reason(verdict: str, detail: str) -> str:
    return (
        f"Lens verdict {verdict}; disposition ESCALATE. This workflow is configured "
        f"as public/high-risk, so the unresolved candidate must not ship. {detail} "
        "Replace it with a concise escalation notice or produce a materially corrected candidate."
    )


def _unknown_message(note: str) -> str:
    safe_note = note if note in _SAFE_UNKNOWN_NOTES else "validation unavailable"
    return (
        "Lens UNKNOWN - response was not validated because the gate was unavailable "
        f"({safe_note})"
    )


def _event_name(event: dict) -> str:
    return str(event.get("hookEventName") or event.get("hook_event_name") or "").casefold()


def _is_subagent_stop(event: dict) -> bool:
    name = _event_name(event)
    return name in {"subagent_stop", "subagentstop"}


def decide(
    event: dict,
    evaluate_fn: Callable[[str, str, str], Verdict] = evaluate,
) -> dict:
    """Return Grok Stop-hook JSON for one decoded event."""

    # Main-agent Stop must not lens a nested agent. SubagentStop is the
    # event that carries the nested agent's finished text.
    if (event.get("subagentType") or event.get("subagent_type")) and not _is_subagent_stop(event):
        return {}
    if not _is_subagent_stop(event) and not _is_end_turn(event):
        return {}

    text = _candidate_text(event)
    if not isinstance(text, str) or not should_validate(text):
        return {}

    active = _stop_active(event)
    domain = os.environ.get("LENS_GROK_DOMAIN", "general")
    context = os.environ.get("LENS_GROK_CONTEXT", "")
    verdict = evaluate_fn(text, domain, context)

    if verdict.status == "PASS":
        return {}

    if verdict.status == "HALT":
        return {"decision": "block", "reason": _halt_reason()}

    if verdict.status in {"FAIL", "HOLD"}:
        if active:
            detail = _format_fail_reason(verdict.violations)
            if _requires_escalation():
                return {
                    "decision": "block",
                    "reason": _escalation_reason("FAIL", detail),
                }
            return {
                "systemMessage": (
                    "Lens verdict FAIL remains after one correction; disposition "
                    "DELIVER_WITH_WARNINGS. " + detail
                )
            }
        return {"decision": "block", "reason": _format_fail_reason(verdict.violations)}

    note = verdict.note if verdict.status == "UNKNOWN" else "invalid validation result"
    message = _unknown_message(note)
    if _requires_escalation():
        return {
            "decision": "block",
            "reason": _escalation_reason("UNKNOWN", message),
        }
    if _fail_closed() and not active:
        return {"decision": "block", "reason": message + ". Re-finish once to retry."}
    return {"systemMessage": message}


def emit(output: dict) -> None:
    """Write a passive Stop diagnostic to stderr; stdout stays empty."""

    if not output:
        return
    note = output.get("systemMessage") or output.get("reason") or ""
    if note:
        sys.stderr.write(note + "\n")


def main() -> int:
    """Read one Grok Stop event and emit a passive diagnostic, if any."""

    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            return 0
    except Exception:  # noqa: BLE001 - malformed hook input must not wedge Grok
        return 0

    try:
        output = decide(event)
    except Exception:  # noqa: BLE001 - apply configured failure policy
        active = _stop_active(event)
        message = _unknown_message("hook runtime error")
        if _requires_escalation():
            output = {
                "decision": "block",
                "reason": _escalation_reason("UNKNOWN", message),
            }
        elif _fail_closed() and not active:
            output = {"decision": "block", "reason": message}
        else:
            output = {"systemMessage": message}

    emit(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
