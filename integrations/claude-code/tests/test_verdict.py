# tests/test_verdict.py
from lens_stop_hook.verdict import Verdict, format_hold_reason


def test_verdict_factories():
    assert Verdict.passed().status == "PASS"
    assert Verdict.unavailable("down").status == "UNAVAILABLE"
    h = Verdict.hold([{"lens": "truth", "severity": "high", "issue": "no source"}])
    assert h.status == "HOLD"
    assert h.violations[0]["lens"] == "truth"


def test_format_hold_reason_is_revision_guidance():
    reason = format_hold_reason([
        {"lens": "truth", "severity": "high", "issue": 'claim "X" has no named source'},
        {"lens": "contradiction", "severity": "high", "issue": "A and B conflict"},
    ])
    assert reason.startswith("Lens HOLD")
    assert "revise" in reason.lower()
    assert "truth:" in reason and "contradiction:" in reason
    assert "no named source" in reason


def test_format_hold_reason_empty():
    assert "revise" in format_hold_reason([]).lower()
