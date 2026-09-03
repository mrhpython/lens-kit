# tests/test_core.py
import lens_stop_hook.core as core
from lens_stop_hook.verdict import Verdict


class _Result:
    def __init__(self, passed, halted=False, halt_reason="", violations=None):
        self._d = {"passed": passed, "halted": halted, "halt_reason": halt_reason,
                   "violations": violations or []}

    def to_dict(self):
        return self._d


class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _stub_gate(result):
    def _call(text, domain="general", context=""):
        return result
    return _call


def test_pass(monkeypatch):
    monkeypatch.setattr(core, "_gate", _stub_gate(_Result(True)))
    monkeypatch.setattr(core, "_lm_ctx", lambda *_: _Ctx())
    monkeypatch.setattr(core, "_cap_ok", lambda: True)
    assert core.evaluate("hello").status == "PASS"


def test_hold(monkeypatch):
    r = _Result(False, violations=[{"lens": "truth", "severity": "high", "issue": "no source"}])
    monkeypatch.setattr(core, "_gate", _stub_gate(r))
    monkeypatch.setattr(core, "_lm_ctx", lambda *_: _Ctx())
    monkeypatch.setattr(core, "_cap_ok", lambda: True)
    v = core.evaluate("hello")
    assert v.status == "HOLD" and v.violations[0]["lens"] == "truth"


def test_rights_halt_is_hold(monkeypatch):
    secret = "PII present: sensitive-test-value"
    r = _Result(False, halted=True, halt_reason=secret)
    monkeypatch.setattr(core, "_gate", _stub_gate(r))
    monkeypatch.setattr(core, "_lm_ctx", lambda *_: _Ctx())
    monkeypatch.setattr(core, "_cap_ok", lambda: True)
    v = core.evaluate("hello")
    assert v.status == "HOLD" and v.violations[0]["lens"] == "rights"
    assert secret not in str(v.violations)


def test_error_is_unavailable(monkeypatch):
    def _boom(*a, **k): raise RuntimeError("model down")
    monkeypatch.setattr(core, "_gate", _boom)
    monkeypatch.setattr(core, "_lm_ctx", lambda *_: _Ctx())
    monkeypatch.setattr(core, "_cap_ok", lambda: True)
    v = core.evaluate("hello")
    assert v.status == "UNAVAILABLE"
    assert v.note == "validation runtime error"
    assert "model down" not in v.note


def test_cap_is_unavailable(monkeypatch):
    monkeypatch.setattr(core, "_cap_ok", lambda: False)
    assert core.evaluate("hello").status == "UNAVAILABLE"
