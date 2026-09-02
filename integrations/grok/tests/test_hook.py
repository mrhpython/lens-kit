import io
import json

import lens_grok_hook.hook as hook
from lens_grok_hook.hook import Verdict


def _event(text="A substantive final answer.", active=False, reason="end_turn"):
    payload = {
        "hookEventName": "stop",
        "lastAssistantMessage": text,
        "stopHookActive": active,
        "promptId": "turn-1",
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def _result(verdict, seen=None):
    def run(text, domain, context):
        if seen is not None:
            seen.update(text=text, domain=domain, context=context)
        return verdict

    return run


def test_pass_allows():
    assert hook.decide(_event(), _result(Verdict.passed())) == {}


def test_uses_camelcase_last_assistant_message_and_forwards_context(monkeypatch):
    monkeypatch.setenv("LENS_GROK_DOMAIN", "finance")
    monkeypatch.setenv("LENS_GROK_CONTEXT", "board report")
    seen = {}
    hook.decide(_event("Frozen candidate text."), _result(Verdict.passed(), seen))
    assert seen == {
        "text": "Frozen candidate text.",
        "domain": "finance",
        "context": "board report",
    }


def test_accepts_snake_case_codex_payload_as_compat():
    seen = {}
    event = {
        "hook_event_name": "Stop",
        "last_assistant_message": "Compat candidate.",
        "stop_hook_active": False,
    }
    hook.decide(event, _result(Verdict.passed(), seen))
    assert seen["text"] == "Compat candidate."


def test_empty_or_non_string_text_does_not_call_gate():
    never = lambda *_: (_ for _ in ()).throw(AssertionError("gate called"))
    assert hook.decide(_event("   "), never) == {}
    assert hook.decide(_event(None), never) == {}


def test_session_end_and_non_end_turn_reasons_do_not_call_gate():
    never = lambda *_: (_ for _ in ()).throw(AssertionError("gate called"))
    for reason in ("channel_closed", "shutdown", "compact"):
        assert hook.decide(_event(reason=reason), never) == {}


def test_missing_reason_still_validates_for_compat_payloads():
    seen = {}
    hook.decide(_event(reason=None), _result(Verdict.passed(), seen))
    assert seen["text"] == "A substantive final answer."


def test_subagent_stop_does_not_call_gate():
    never = lambda *_: (_ for _ in ()).throw(AssertionError("gate called"))
    event = _event()
    event["subagentType"] = "explore"
    assert hook.decide(event, never) == {}


def test_fail_first_stop_blocks_with_specific_guidance():
    verdict = Verdict.failed([{"lens": "truth", "issue": "claim lacks a source"}])
    output = hook.decide(_event(active=False), _result(verdict))
    assert output["decision"] == "block"
    assert "disposition REVISE" in output["reason"]
    assert "truth: claim lacks a source" in output["reason"]
    assert "additionalContext" not in output
    assert "hookSpecificOutput" not in output


def test_fail_after_rework_does_not_block_again():
    verdict = Verdict.failed([{"lens": "truth", "issue": "still unsupported"}])
    output = hook.decide(_event(active=True), _result(verdict))
    assert "decision" not in output
    assert "DELIVER_WITH_WARNINGS" in output["systemMessage"]
    assert "still unsupported" in output["systemMessage"]
    assert "additionalContext" not in json.dumps(output)


def test_fail_after_rework_escalates_public_or_high_risk(monkeypatch):
    monkeypatch.setenv("LENS_GROK_RISK", "public")
    verdict = Verdict.failed([{"lens": "truth", "issue": "still unsupported"}])
    output = hook.decide(_event(active=True), _result(verdict))
    assert output["decision"] == "block"
    assert "disposition ESCALATE" in output["reason"]
    assert "must not ship" in output["reason"]


def test_unknown_fail_open_is_visible_and_not_pass(monkeypatch):
    monkeypatch.setenv("LENS_GROK_FAIL", "open")
    raw_note = "provider timed out with secret-value at /private/user/path"
    output = hook.decide(_event(), _result(Verdict.unknown(raw_note)))
    assert "decision" not in output
    assert "UNKNOWN" in output["systemMessage"]
    assert "not validated" in output["systemMessage"]
    assert raw_note not in output["systemMessage"]


def test_unknown_high_risk_escalates(monkeypatch):
    monkeypatch.setenv("LENS_GROK_RISK", "high-stakes")
    output = hook.decide(_event(), _result(Verdict.unknown("validation timeout")))
    assert output["decision"] == "block"
    assert "verdict UNKNOWN" in output["reason"]
    assert "disposition ESCALATE" in output["reason"]


def test_unknown_fail_closed_blocks_once(monkeypatch):
    monkeypatch.setenv("LENS_GROK_FAIL", "closed")
    evaluator = _result(Verdict.unknown("provider timed out"))
    first = hook.decide(_event(active=False), evaluator)
    second = hook.decide(_event(active=True), evaluator)
    assert first["decision"] == "block"
    assert "UNKNOWN" in first["reason"]
    assert "decision" not in second
    assert "UNKNOWN" in second["systemMessage"]


def test_unexpected_status_is_unknown_not_pass():
    output = hook.decide(_event(), _result(Verdict("BROKEN")))
    assert "UNKNOWN" in output["systemMessage"]
    assert "invalid validation result" in output["systemMessage"]


def test_default_scope_skips_only_exact_simple_ack(monkeypatch):
    monkeypatch.delenv("LENS_GROK_SCOPE", raising=False)
    never = lambda *_: (_ for _ in ()).throw(AssertionError("gate called"))
    assert hook.decide(_event("Thanks!"), never) == {}
    assert hook.should_validate("Thanks - tests pass.") is True
    assert hook.should_validate("Delete the account now.") is True


def test_scope_all_validates_ack_and_off_skips_everything(monkeypatch):
    monkeypatch.setenv("LENS_GROK_SCOPE", "all")
    assert hook.should_validate("Thanks!") is True
    monkeypatch.setenv("LENS_GROK_SCOPE", "off")
    assert hook.should_validate("A substantive answer with claims.") is False


def test_unknown_scope_fails_toward_validation(monkeypatch):
    monkeypatch.setenv("LENS_GROK_SCOPE", "typo")
    assert hook.should_validate("Thanks!") is True


def test_stop_hook_active_camelcase_is_the_rework_flag():
    verdict = Verdict.failed([{"lens": "truth", "issue": "still unsupported"}])
    output = hook.decide(_event(active=True), _result(verdict))
    assert "decision" not in output
    assert "DELIVER_WITH_WARNINGS" in output["systemMessage"]


class _GateResult:
    def __init__(self, record):
        self.record = record

    def to_dict(self):
        return self.record


class _Gate:
    def __init__(self, record=None, error=None):
        self.record = record
        self.error = error

    def __call__(self, **_kwargs):
        if self.error:
            raise self.error
        return _GateResult(self.record)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_evaluate_pass_hold_rights_and_error(monkeypatch):
    monkeypatch.setattr(hook, "lm_context", lambda _profile: _Context())
    monkeypatch.setattr(hook, "_runtime", lambda: (object(), _Gate({"passed": True})))
    assert hook.evaluate("candidate").status == "PASS"

    monkeypatch.setattr(
        hook,
        "_runtime",
        lambda: (object(), _Gate({
            "passed": False,
            "violations": [{"lens": "truth", "issue": "unsupported"}],
        })),
    )
    assert hook.evaluate("candidate").status == "FAIL"

    monkeypatch.setattr(
        hook,
        "_runtime",
        lambda: (object(), _Gate({
            "passed": True,
            "violations": [{"lens": "relevance", "issue": "actionable warning"}],
        })),
    )
    assert hook.evaluate("candidate").status == "FAIL"

    raw_credential = "sk-secret-value-must-not-appear"
    private_path = "/private/user/path/credentials.env"
    monkeypatch.setattr(
        hook,
        "_runtime",
        lambda: (object(), _Gate({
            "passed": False,
            "halted": True,
            "halt_reason": f"credential {raw_credential} found in {private_path}",
            "violations": [{
                "lens": "rights",
                "issue": f"exposed {raw_credential} at {private_path}",
            }],
        })),
    )
    rights = hook.evaluate("candidate")
    assert rights.status == "HALT"
    assert rights.violations[0]["lens"] == "rights"
    assert raw_credential not in json.dumps(rights.violations)
    assert private_path not in json.dumps(rights.violations)
    rights_output = hook.decide(_event(), _result(rights))
    assert rights_output["decision"] == "block"
    assert raw_credential not in rights_output["reason"]
    assert private_path not in rights_output["reason"]
    assert "Remove or redact" in rights_output["reason"]

    monkeypatch.setattr(
        hook,
        "_runtime",
        lambda: (object(), _Gate(error=RuntimeError(
            f"provider failed with {raw_credential} from {private_path}"
        ))),
    )
    unknown = hook.evaluate("candidate")
    assert unknown.status == "UNKNOWN"
    assert unknown.note == "validation runtime error"
    unknown_output = hook.decide(_event(), _result(unknown))
    assert raw_credential not in unknown_output["systemMessage"]
    assert private_path not in unknown_output["systemMessage"]


def test_direct_raw_rights_verdict_is_sanitized_before_hook_output():
    raw_credential = "token-secret-must-not-appear"
    private_path = "/private/user/path/report.txt"
    verdict = Verdict.halted([{
        "lens": "rights",
        "severity": "critical",
        "issue": f"remove {raw_credential} found at {private_path}",
    }])
    output = hook.decide(_event(), _result(verdict))
    assert output["decision"] == "block"
    assert raw_credential not in output["reason"]
    assert private_path not in output["reason"]
    assert "Sensitive or restricted content" in output["reason"]


def test_rights_halt_reblocks_after_revision():
    verdict = Verdict.halted([{
        "lens": "rights",
        "severity": "critical",
        "issue": "raw value must never be echoed",
    }])
    output = hook.decide(_event(active=True), _result(verdict))
    assert output["decision"] == "block"
    assert "verdict HALT" in output["reason"]
    assert "disposition HALT" in output["reason"]
    assert "raw value" not in output["reason"]


def test_main_emits_exact_json_for_block(monkeypatch, capsys):
    event = _event()
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(event)))
    monkeypatch.setattr(
        hook,
        "decide",
        lambda _event: {"decision": "block", "reason": "revise"},
    )
    assert hook.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "decision": "block",
        "reason": "revise",
    }


def test_main_allow_note_goes_to_stderr_not_stdout(monkeypatch, capsys):
    """Grok Stop additionalContext keeps the agent working. Allow-notes must not be JSON."""
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_event())))
    monkeypatch.setattr(
        hook,
        "decide",
        lambda _event: {"systemMessage": "Lens UNKNOWN - response was not validated"},
    )
    assert hook.main() == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not validated" in captured.err
    assert "additionalContext" not in captured.err


def test_main_malformed_or_non_object_input_does_not_wedge(monkeypatch, capsys):
    for payload in ("not-json", "[]", "null"):
        monkeypatch.setattr(hook.sys, "stdin", io.StringIO(payload))
        assert hook.main() == 0
        assert capsys.readouterr().out == ""


def test_main_uncaught_error_obeys_fail_policy(monkeypatch, capsys):
    raw_credential = "secret-main-value-must-not-appear"
    private_path = "/private/user/path/hook.py"
    monkeypatch.setenv("LENS_GROK_FAIL", "closed")
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_event())))
    monkeypatch.setattr(
        hook,
        "decide",
        lambda _event: (_ for _ in ()).throw(
            RuntimeError(f"failed with {raw_credential} at {private_path}")
        ),
    )
    assert hook.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "block"
    assert "UNKNOWN" in output["reason"]
    assert "hook runtime error" in output["reason"]
    assert raw_credential not in output["reason"]
    assert private_path not in output["reason"]
