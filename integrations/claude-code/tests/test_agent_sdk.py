import asyncio
import os
import threading

import pytest

from lens_stop_hook import agent_sdk
from lens_stop_hook.verdict import Verdict


def _run(cb, inp):
    return asyncio.run(cb(inp, None, None))


def _fake(verdict, recorder=None):
    def _ev(text, domain="general", context=""):
        if recorder is not None:
            recorder.update(text=text, domain=domain, context=context)
        return verdict
    return _ev


def test_factory_returns_coroutine_function():
    cb = agent_sdk.make_stop_hook(evaluate=_fake(Verdict.passed()),
                                  extract_text=lambda inp: "hi")
    assert asyncio.iscoroutinefunction(cb)


def test_pass_allows():
    cb = agent_sdk.make_stop_hook(evaluate=_fake(Verdict.passed()),
                                  extract_text=lambda inp: "grounded answer")
    assert _run(cb, {"stop_hook_active": False}) == {}


def test_hold_first_stop_blocks_with_reason():
    v = Verdict.hold([{"lens": "truth", "issue": "unsupported claim"}])
    cb = agent_sdk.make_stop_hook(evaluate=_fake(v), extract_text=lambda inp: "x")
    out = _run(cb, {"stop_hook_active": False})
    assert out["decision"] == "block"
    assert "unsupported claim" in out["reason"]


def test_hold_restop_downgrades_to_warning():
    v = Verdict.hold([{"lens": "truth", "issue": "still off"}])
    cb = agent_sdk.make_stop_hook(evaluate=_fake(v), extract_text=lambda inp: "x")
    out = _run(cb, {"stop_hook_active": True})
    assert "decision" not in out
    assert "systemMessage" in out


def test_rights_hold_restop_remains_blocked():
    v = Verdict.hold([{"lens": "rights", "issue": "do not echo this"}])
    cb = agent_sdk.make_stop_hook(evaluate=_fake(v), extract_text=lambda inp: "x")
    out = _run(cb, {"stop_hook_active": True})
    assert out["decision"] == "block"
    assert "HALT" in out["reason"]
    assert "do not echo this" not in out["reason"]


def test_hold_restop_escalates_public(monkeypatch):
    monkeypatch.setenv("LENS_HOOK_RISK", "public")
    v = Verdict.hold([{"lens": "truth", "issue": "still off"}])
    cb = agent_sdk.make_stop_hook(evaluate=_fake(v), extract_text=lambda inp: "x")
    out = _run(cb, {"stop_hook_active": True})
    assert out["decision"] == "block"
    assert "ESCALATE" in out["reason"]


def test_unavailable_fail_open_default(monkeypatch):
    monkeypatch.delenv("LENS_HOOK_FAIL", raising=False)
    cb = agent_sdk.make_stop_hook(evaluate=_fake(Verdict.unavailable("boom")),
                                  extract_text=lambda inp: "x")
    out = _run(cb, {"stop_hook_active": False})
    assert "decision" not in out
    assert "not validated" in out["systemMessage"]


def test_unavailable_fail_closed_blocks(monkeypatch):
    monkeypatch.setenv("LENS_HOOK_FAIL", "closed")
    cb = agent_sdk.make_stop_hook(evaluate=_fake(Verdict.unavailable("boom")),
                                  extract_text=lambda inp: "x")
    out = _run(cb, {"stop_hook_active": False})
    assert out["decision"] == "block"
    assert "not validated" in out["reason"]


def test_empty_text_allows():
    cb = agent_sdk.make_stop_hook(evaluate=_fake(Verdict.hold([{"lens": "x", "issue": "y"}])),
                                  extract_text=lambda inp: "")
    assert _run(cb, {"stop_hook_active": False}) == {}


def test_domain_and_context_forwarded():
    rec = {}
    cb = agent_sdk.make_stop_hook(evaluate=_fake(Verdict.passed(), rec),
                                  extract_text=lambda inp: "ans",
                                  domain="finance", context="quarterly report")
    _run(cb, {"stop_hook_active": False})
    assert rec["domain"] == "finance"
    assert rec["context"] == "quarterly report"
    assert rec["text"] == "ans"


def test_gate_runs_off_event_loop():
    # DISCRIMINATING: in the worker thread there is no running loop. If evaluate
    # were awaited inline on the coroutine, get_running_loop() would succeed.
    seen = {}

    def _ev(text, domain="general", context=""):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        seen["thread_is_main"] = (threading.current_thread() is threading.main_thread())
        return Verdict.passed()

    cb = agent_sdk.make_stop_hook(evaluate=_ev, extract_text=lambda inp: "x")
    _run(cb, {"stop_hook_active": False})
    assert seen["on_loop"] is False
    assert seen["thread_is_main"] is False


def test_default_extract_prefers_last_assistant_message():
    # The Agent SDK supplies the final answer directly as last_assistant_message.
    out = agent_sdk._default_extract(
        {"last_assistant_message": "the final answer", "transcript_path": "/nope"}
    )
    assert out == "the final answer"


def test_default_extract_falls_back_to_transcript(monkeypatch):
    # No last_assistant_message -> use the transcript parser with the given path.
    calls = {}

    def _ft(p):
        calls["path"] = p
        return "from transcript"

    monkeypatch.setattr(agent_sdk, "final_assistant_text", _ft)
    out = agent_sdk._default_extract({"transcript_path": "/tmp/x.jsonl"})
    assert out == "from transcript"
    assert calls["path"] == "/tmp/x.jsonl"


def test_default_extract_blank_lam_falls_back(monkeypatch):
    monkeypatch.setattr(agent_sdk, "final_assistant_text", lambda p: "fallback")
    out = agent_sdk._default_extract({"last_assistant_message": "   ",
                                      "transcript_path": "/tmp/x.jsonl"})
    assert out == "fallback"


def test_default_extract_used_end_to_end():
    # Wired through make_stop_hook with the REAL default extractor: a HOLD on the
    # SDK-supplied last_assistant_message must reach the gate (not no-op to {}).
    v = Verdict.hold([{"lens": "truth", "issue": "fabricated number"}])
    cb = agent_sdk.make_stop_hook(evaluate=_fake(v))  # default extract_text
    out = _run(cb, {"stop_hook_active": False,
                    "last_assistant_message": "Revenue grew 312% last quarter."})
    assert out["decision"] == "block"
    assert "fabricated number" in out["reason"]


def test_uncaught_error_fails_per_policy(monkeypatch):
    def _boom(inp):
        raise ValueError("extract exploded")

    monkeypatch.delenv("LENS_HOOK_FAIL", raising=False)
    cb = agent_sdk.make_stop_hook(evaluate=_fake(Verdict.passed()), extract_text=_boom)
    out = _run(cb, {"stop_hook_active": False})
    assert "decision" not in out
    assert "not validated" in out["systemMessage"]

    monkeypatch.setenv("LENS_HOOK_FAIL", "closed")
    cb = agent_sdk.make_stop_hook(evaluate=_fake(Verdict.passed()), extract_text=_boom)
    out = _run(cb, {"stop_hook_active": False})
    assert out["decision"] == "block"
    assert "not validated" in out["reason"]
