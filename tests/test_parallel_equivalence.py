"""Parallel path is byte-identical to the sequential reference (stubbed, no network).

Primary load-bearing evidence for the parallel-lens change: deterministic,
isolates the scheduler from LLM nondeterminism. Each scenario builds two
gates with IDENTICAL stub outputs — one parallel=False, one parallel=True —
and asserts the verdicts match on every field except timings (wall-time
differs by design).
"""
import dspy
import pytest

from conftest import make_stub_gate, StubPredictor


def _pair(overrides=None, *, auto_fix=False, profile=None, fixer_fixed=None):
    """Two stub gates differing only in `parallel`. Fixer is stubbed when auto_fix."""
    seq = make_stub_gate(profile=profile, overrides=overrides, auto_fix=auto_fix, parallel=False)
    par = make_stub_gate(profile=profile, overrides=overrides, auto_fix=auto_fix, parallel=True)
    if auto_fix:
        for g in (seq, par):
            g.fixer = StubPredictor(fixed_text=fixer_fixed or "")
    return seq, par


def _key(result):
    """The fields the spec requires identical (timings excluded — wall-time)."""
    return {
        "passed": result.passed,
        "halted": result.halted,
        "halt_reason": result.halt_reason,
        "fixed_text": result.fixed_text,
        "consciousness_flags": result.consciousness_flags,
        "per_lens": result.per_lens,
        "violations": [(v.lens, v.severity, v.issue, v.auto_fixable) for v in result.violations],
    }


def _assert_equiv(overrides=None, *, text, context="", domain="general", **kw):
    seq, par = _pair(overrides, **kw)
    r_seq = seq(text=text, context=context, domain=domain)
    r_par = par(text=text, context=context, domain=domain)
    assert _key(r_par) == _key(r_seq)


def test_equiv_clean():
    _assert_equiv(text="Plain clean text with nothing to flag.")


def test_equiv_rights_halt():
    _assert_equiv(
        overrides={"rights": dict(halt="true", has_violations="true",
                   violations='[{"issue": "SSN exposed", "severity": "critical"}]')},
        text="SSN: 123-45-6789")


def test_equiv_rights_scrub():
    _assert_equiv(
        overrides={"rights": dict(has_violations="true", scrubbed_text="[REDACTED] remains",
                   violations='[{"issue": "email PII", "severity": "warning"}]')},
        text="Contact a@b.com remains")


def test_equiv_truth_mask_and_fix():
    _assert_equiv(
        overrides={"truth": dict(has_violations="true", fixed_text="The rate of [UNKNOWN] applies.",
                   violations='[{"issue": "7.5% has no source", "severity": "critical"}]')},
        text="The rate of 7.5% applies.")


def test_equiv_extrapolation_fix():
    _assert_equiv(
        overrides={"extrapolation": dict(has_violations="true", fixed_text="Revenue [PROJECTION] grows.",
                   violations='[{"issue": "unqualified projection", "severity": "warning"}]')},
        text="Revenue grows 50%.")


def test_equiv_auto_fix_critical():
    _assert_equiv(
        overrides={"truth": dict(has_violations="true", fixed_text="masked",
                   violations='[{"issue": "fabricated stat", "severity": "critical"}]')},
        text="Our revenue was $9B last quarter.",
        auto_fix=True, fixer_fixed="Our revenue was [UNKNOWN] last quarter.")


def test_equiv_relevance_with_context():
    _assert_equiv(
        overrides={"relevance": dict(has_violations="true",
                   violations='[{"issue": "off-audience", "severity": "warning"}]')},
        text="Some text.", context="CFO audience, needs cost numbers")


def test_equiv_no_context_skips_relevance():
    _assert_equiv(text="Some text.", context="")


def test_equiv_scenario_vocab_suppresses_contradiction():
    from lens_kit import Profile
    profile = Profile.empty()
    profile.vocabulary.scenario_labels = ["bull case", "bear case"]
    _assert_equiv(
        profile=profile,
        overrides={"contradiction": dict(has_violations="true",
                   violations='[{"claim_a": "GBP rallies", "claim_b": "GBP falls", "severity": "critical"}]')},
        text="Bull case: GBP rallies. Bear case: GBP falls.")


def test_equiv_consistency_raises_is_skipped_both_paths():
    # A lens that sequential wraps in try/except must be skipped identically in parallel.
    seq, par = _pair()
    seq.consistency = _raiser()
    par.consistency = _raiser()
    r_seq = seq(text="Section A says 10. Section B says 20.")
    r_par = par(text="Section A says 10. Section B says 20.")
    assert _key(r_par) == _key(r_seq)
    assert r_par.per_lens["consistency"] is True  # skipped, not blocking


def test_equiv_cross_check_injection():
    _assert_equiv(
        overrides={"cross_check": dict(
            missed_violations='[{"lens": "truth", "issue": "missed 42%", "location": "para 2"}]')},
        text="Some text with 42% somewhere.")


def test_equiv_claim_extractor_raises_propagates_both_paths():
    # claim_extractor.extract is NOT wrapped in try/except in sequential, so a
    # failure there must PROPAGATE out of the gate in BOTH modes — never be
    # swallowed to "no claims". (Regression: parallel originally swallowed it.)
    boom = lambda text: (_ for _ in ()).throw(RuntimeError("boom"))
    seq, par = _pair()
    seq.claim_extractor.extract = boom
    par.claim_extractor.extract = boom
    with pytest.raises(RuntimeError):
        seq(text="explodes inside claim extraction")
    with pytest.raises(RuntimeError):
        par(text="explodes inside claim extraction")


class _Raiser:
    calls = []
    def __call__(self, **kwargs):
        raise RuntimeError("boom")


def _raiser():
    return _Raiser()
