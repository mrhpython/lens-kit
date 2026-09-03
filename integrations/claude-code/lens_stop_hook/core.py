# lens_stop_hook/core.py
"""The only lens-touching code. evaluate(text) -> Verdict.

Builds the LensGate once at import (no network until called), reusing it across
calls — exactly like serve/lens_serve/gate_tool.py. Any failure (error, cap)
returns UNAVAILABLE; the hook driver applies the fail-open/closed policy.
"""
from __future__ import annotations

import os
from datetime import date

from lens_kit import LensGate, Profile, builtin_profile_path, lm_context

from .verdict import Verdict

_SAFE_RIGHTS_ISSUE = (
    "Sensitive or restricted content was detected. Remove or redact it before finishing."
)

# LENS_HOOK_PROFILE wins; otherwise fall back to the profile shipped inside
# lens_kit, resolved ABSOLUTELY. The old default was the relative string
# "profiles/qwen-serve.yaml", which only resolved when the process happened
# to start in the kit's serve/ directory — a hook never does, so the default
# could not work where it mattered most.
_profile = Profile.load(os.environ.get("LENS_HOOK_PROFILE") or str(builtin_profile_path()))
_gate = LensGate(profile=_profile, parallel=True)
_lm_ctx = lm_context  # indirection so tests can stub it

DAILY_CAP = int(os.environ.get("LENS_HOOK_DAILY_CAP", "200"))
_calls = {"day": None, "count": 0}


def _cap_ok() -> bool:
    today = str(date.today())
    if _calls["day"] != today:
        _calls["day"], _calls["count"] = today, 0
    if _calls["count"] >= DAILY_CAP:
        return False
    _calls["count"] += 1
    return True


def evaluate(text: str, domain: str = "general", context: str = "") -> Verdict:
    if not _cap_ok():
        return Verdict.unavailable(f"daily cap {DAILY_CAP} reached")
    try:
        with _lm_ctx(_profile):
            result = _gate(text=text, domain=domain, context=context)
        r = result.to_dict()
    except Exception as e:  # noqa: BLE001 — any failure is UNAVAILABLE, never a silent pass
        if isinstance(e, TimeoutError):
            note = "validation timeout"
        elif isinstance(e, (ConnectionError, OSError)):
            note = "provider or validation I/O error"
        else:
            note = "validation runtime error"
        return Verdict.unavailable(note)

    if r.get("passed") and not r.get("halted"):
        return Verdict.passed()
    violations = list(r.get("violations", []))
    if r.get("halted"):
        violations = [{"lens": "rights", "severity": "critical",
                       "issue": _SAFE_RIGHTS_ISSUE}]
    return Verdict.hold(violations)
