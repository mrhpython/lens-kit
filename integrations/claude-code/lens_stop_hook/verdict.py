# lens_stop_hook/verdict.py
"""The gate's classification of one piece of text, plus HOLD-reason formatting.

A Stop hook can only block (force rework) or allow — so the driver maps these
three statuses onto that contract. UNAVAILABLE means the lens could not judge
(error/timeout/cap); it is NOT a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Verdict:
    status: str  # "PASS" | "HOLD" | "UNAVAILABLE"
    violations: list = field(default_factory=list)
    note: str = ""

    @classmethod
    def passed(cls) -> "Verdict":
        return cls("PASS")

    @classmethod
    def hold(cls, violations: list) -> "Verdict":
        return cls("HOLD", violations=list(violations))

    @classmethod
    def unavailable(cls, note: str) -> "Verdict":
        return cls("UNAVAILABLE", note=note)


def format_hold_reason(violations: list) -> str:
    """Render lens violations as legitimate, specific revision guidance.

    Phrasing matters: arbitrary/opaque demands get refused as prompt-injection
    (spike run 2). Ground every clause in the lens's own {lens, issue}.
    """
    if not violations:
        return "Lens HOLD — revise your answer before finishing; it did not pass validation."
    clauses = "; ".join(f"{v.get('lens', '?')}: {v.get('issue', '').strip()}" for v in violations)
    return (
        "Lens HOLD — revise before finishing. The following issues were found in your "
        f"answer; fix them, then finish: {clauses}"
    )
