"""Codex Stop hook: finished assistant text -> outside-in lens verdict.

Codex supplies the finished response directly as ``last_assistant_message``.
The hook validates that frozen text with lens-kit and maps the three possible
outcomes to the Codex Stop-hook contract:

* PASS: allow the turn to finish (SHIP).
* FAIL: block once with concrete revision guidance (REVISE). After that one
  correction, internal low-risk work may finish with explicit warnings;
  configured public/high-risk work is escalated and does not ship.
* HALT: a Rights failure remains blocked until the sensitive content is removed.
* UNKNOWN: never call it PASS. Fail open with a visible warning by default, or
  block once when ``LENS_CODEX_FAIL=closed``. Configured public/high-risk work
  is escalated and does not ship while validation is unavailable.

Malformed input exits successfully without output because a broken hook must
not wedge Codex before it can report the integration problem.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from lens_kit import ConfigError, LensGate, Profile, builtin_profile_path, lm_context


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
def _runtime() -> tuple[Profile, LensGate]:
    """Load the selected profile and gate lazily, once per hook process."""

    profile_path = os.environ.get("LENS_CODEX_PROFILE") or str(builtin_profile_path())
    profile = Profile.load(profile_path)
    return profile, LensGate(profile=profile, parallel=True)


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
        # Rights findings can contain the exact credential/PII value. A halted
        # result therefore discards both halt_reason and every raw violation;
        # only value-free revision guidance may enter Codex's prompt/UI.
        return Verdict.halted([{
            "lens": "rights",
            "severity": "critical",
            "issue": _SAFE_RIGHTS_ISSUE,
        }])
    return Verdict.failed([_safe_violation(item) for item in violations])


def _safe_validation_exception_note(exc: Exception) -> str:
    """Classify an exception without exposing its message, paths, or values."""

    if isinstance(exc, ConfigError):
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


# Conservative finite allowlist. This is not a semantic classifier: it only
# recognizes whole-message acknowledgements after punctuation/space folding.
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

    ``LENS_CODEX_SCOPE`` values:

    * ``substantive`` (default): skip only exact phrases in ``_SIMPLE_ACKS``;
    * ``all``: validate every non-empty response;
    * ``off``: disable gate calls while leaving the hook installed.

    Unknown values fail safely toward validation.
    """

    if not text or not text.strip():
        return False
    scope = os.environ.get("LENS_CODEX_SCOPE", "substantive").strip().casefold()
    if scope == "off":
        return False
    if scope == "substantive" and _normalized_ack(text) in _SIMPLE_ACKS:
        return False
    return True


def _fail_closed() -> bool:
    return os.environ.get("LENS_CODEX_FAIL", "open").strip().casefold() == "closed"


def _requires_escalation() -> bool:
    """Return whether unresolved/UNKNOWN output must not ship.

    The hook deliberately does not guess risk from prose. Operators explicitly
    classify the workflow with ``LENS_CODEX_RISK``. Unknown values fail toward
    escalation; only ``internal-low`` enables delivery with warnings.
    """

    risk = os.environ.get("LENS_CODEX_RISK", "internal-low").strip().casefold()
    return risk != "internal-low"


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
    return (
        "Lens verdict HALT; disposition HALT. " + _SAFE_RIGHTS_ISSUE
    )


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


def decide(
    event: dict,
    evaluate_fn: Callable[[str, str, str], Verdict] = evaluate,
) -> dict:
    """Return Codex Stop-hook JSON for one decoded event."""

    text = event.get("last_assistant_message")
    if not isinstance(text, str) or not should_validate(text):
        return {}

    active = bool(event.get("stop_hook_active"))
    domain = os.environ.get("LENS_CODEX_DOMAIN", "general")
    context = os.environ.get("LENS_CODEX_CONTEXT", "")
    verdict = evaluate_fn(text, domain, context)

    if verdict.status == "PASS":
        return {}

    if verdict.status == "HALT":
        # Rights is the only unconditional Lens hard stop. Re-blocking is
        # intentional until the candidate no longer contains the detected data.
        return {"decision": "block", "reason": _halt_reason()}

    if verdict.status in {"FAIL", "HOLD"}:  # HOLD accepts pre-0.2 adapter callers.
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

    # Any non-PASS/non-HOLD result is treated as UNKNOWN. This also keeps an
    # unexpected integration status from accidentally becoming an allow/PASS.
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


def main() -> int:
    """Read one Codex Stop event from stdin and write one JSON object, if any."""

    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            return 0
    except Exception:  # noqa: BLE001 - malformed hook input must not wedge Codex
        return 0

    try:
        output = decide(event)
    except Exception:  # noqa: BLE001 - apply configured failure policy
        active = bool(event.get("stop_hook_active"))
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

    if output:
        sys.stdout.write(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
