# tests/test_hook.py
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
