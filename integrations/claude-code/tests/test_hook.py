# tests/test_hook.py
from lens_stop_hook import hook
from lens_stop_hook.hook import decide
from lens_stop_hook.verdict import Verdict


def _ev(active=False):
    return {"hook_event_name": "Stop", "stop_hook_active": active, "transcript_path": "x"}


def test_pass_allows():
    out = decide(_ev(), lambda t: Verdict.passed(), "answer")
    assert out == {}


def test_empty_text_allows():
    # core not even called when there's nothing to lens
    out = decide(_ev(), lambda t: (_ for _ in ()).throw(AssertionError("called")), "   ")
    assert out == {}


def test_event_text_prefers_current_stop_message():
    event = {
        "last_assistant_message": "current answer",
        "transcript_path": "/not/read",
    }
    assert hook._event_text(event) == "current answer"


def test_event_text_falls_back_for_older_payload(monkeypatch):
    monkeypatch.setattr(hook, "final_assistant_text", lambda path: f"from {path}")
    assert hook._event_text({"transcript_path": "/tmp/session.jsonl"}) == (
        "from /tmp/session.jsonl"
    )


def test_hold_first_blocks_with_guidance():
    v = Verdict.hold([{"lens": "truth", "issue": "no source"}])
    out = decide(_ev(active=False), lambda t: v, "answer")
    assert out["decision"] == "block"
    assert "truth:" in out["reason"] and "revise" in out["reason"].lower()


def test_hold_after_rework_allows_with_warning():
    v = Verdict.hold([{"lens": "truth", "issue": "no source"}])
    out = decide(_ev(active=True), lambda t: v, "answer")
    assert "decision" not in out
    assert "still flagged" in out["systemMessage"]


def test_hold_after_rework_escalates_public(monkeypatch):
    monkeypatch.setenv("LENS_HOOK_RISK", "public")
    v = Verdict.hold([{"lens": "truth", "issue": "no source"}])
    out = decide(_ev(active=True), lambda t: v, "answer")
    assert out["decision"] == "block"
    assert "ESCALATE" in out["reason"]


def test_rights_hold_reblocks_after_rework_without_echoing_value():
    secret = "sensitive-test-value"
    v = Verdict.hold([{"lens": "rights", "issue": secret}])
    out = decide(_ev(active=True), lambda t: v, "answer")
    assert out["decision"] == "block"
    assert "HALT" in out["reason"]
    assert secret not in out["reason"]


def test_unavailable_failopen_allows_with_note(monkeypatch):
    monkeypatch.setenv("LENS_HOOK_FAIL", "open")
    out = decide(_ev(), lambda t: Verdict.unavailable("down"), "answer")
    assert "decision" not in out
    assert "not validated" in out["systemMessage"]


def test_unavailable_failclosed_blocks(monkeypatch):
    monkeypatch.setenv("LENS_HOOK_FAIL", "closed")
    out = decide(_ev(active=False), lambda t: Verdict.unavailable("down"), "answer")
    assert out["decision"] == "block"
    assert "not validated" in out["reason"]


def test_unavailable_failclosed_does_not_wedge_after_rework(monkeypatch):
    monkeypatch.setenv("LENS_HOOK_FAIL", "closed")
    out = decide(_ev(active=True), lambda t: Verdict.unavailable("down"), "answer")
    assert "decision" not in out  # N=1 cap: don't block twice


def test_unavailable_public_escalates_even_after_rework(monkeypatch):
    monkeypatch.setenv("LENS_HOOK_RISK", "public")
    out = decide(_ev(active=True), lambda t: Verdict.unavailable("down"), "answer")
    assert out["decision"] == "block"
    assert "UNKNOWN" in out["reason"]
    assert "ESCALATE" in out["reason"]
