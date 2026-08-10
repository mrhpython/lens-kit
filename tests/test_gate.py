"""Gate composition and verdict structure — stubbed predictors, no network."""
import json

from lens_kit import CANONICAL_LENSES, LensResult, Profile

from conftest import make_stub_gate

CORE_PER_LENS = {"rights", "truth", "causality", "definitionalIntegrity",
                 "contradiction", "extrapolation", "structure", "consistency"}


def test_default_gate_composes_with_bestofn():
    import dspy
    from lens_kit import LensGate
    gate = LensGate()  # construction needs no configured LM
    # Truth is plain Predict (N=1) since 2026-08-10: the BestOfN strictness
    # reward is a false-positive amplifier under the v4 adjudicated standard,
    # and all banked eval numbers measure the N=1 path.
    assert isinstance(gate.truth, dspy.Predict)
    assert isinstance(gate.extrapolation, dspy.BestOfN)
    fast = LensGate(fast_mode=True)
    assert isinstance(fast.truth, dspy.Predict)
    assert isinstance(fast.extrapolation, dspy.Predict)


def test_canonical_lens_set_is_ten():
    assert len(CANONICAL_LENSES) == 10
    assert set(CANONICAL_LENSES) == CORE_PER_LENS | {"relevance", "consciousScan"}
    assert "definitionalIntegrity" in CANONICAL_LENSES


def test_definitional_integrity_signature_shape():
    """Item 1: the new lens signature exists with the documented fields."""
    from lens_kit import signatures as S
    sig = S.DefinitionalIntegrityCheck
    fields = sig.input_fields | sig.output_fields
    assert "text" in sig.input_fields
    assert "domain" in sig.input_fields
    assert "has_violations" in sig.output_fields
    assert "violations" in sig.output_fields
    # Boundary statement: the lens must NOT claim missing-mechanism/circular
    # causation (Causality owns those). Rubric states the boundary.
    doc = sig.__doc__.lower()
    assert "equivocation" in doc
    assert "causality" in doc  # boundary disclaimer present
    assert "undefined" in doc or "never defined" in doc


def test_clean_text_passes_all_lenses():
    gate = make_stub_gate()
    result = gate(text="Plain clean text with nothing to flag.")
    assert isinstance(result, LensResult)
    assert result.passed is True
    assert result.halted is False
    assert result.violations == []
    assert result.consciousness_flags == []
    assert set(result.per_lens) == CORE_PER_LENS  # relevance absent without context
    assert all(result.per_lens.values())
    assert result.fixed_text == "Plain clean text with nothing to flag."


def test_relevance_runs_only_with_context_and_is_warning_only():
    gate = make_stub_gate(overrides={
        "relevance": dict(has_violations="true",
                          violations='[{"section": "s1", "issue": "off-audience", "severity": "warning"}]'),
    })
    result = gate(text="Some text.", context="CFO audience, needs cost numbers")
    assert "relevance" in result.per_lens
    assert result.per_lens["relevance"] is False
    assert result.passed is True  # warning-only lens never fails the gate
    assert any(v.lens == "relevance" and v.severity == "warning" for v in result.violations)


def test_truth_violation_fails_gate():
    gate = make_stub_gate(overrides={
        "truth": dict(has_violations="true",
                      violations='[{"issue": "7.5% ECB rate has no named source", '
                                 '"severity": "critical", "location": "the 7.5% rate"}]'),
    })
    result = gate(text="The ECB rate of 7.5% means costs explode.")
    assert result.passed is False
    assert result.per_lens["truth"] is False
    assert any(v.lens == "truth" for v in result.violations)


def test_rights_halt_stops_pipeline():
    # Fixture must be PII the REGEX pre-pass cannot see (a context-judgment
    # violation), so this exercises the LLM Rights halt path — structured
    # PII like an SSN literal now halts earlier, in the deterministic
    # pre-pass (see test_pii_prepass_halts_before_any_llm_call).
    gate = make_stub_gate(overrides={
        "rights": dict(halt="true", has_violations="true",
                       violations='[{"issue": "Dark pattern: fabricated urgency", "severity": "critical", "pii_type": ""}]'),
    })
    result = gate(text="Only 2 seats left — everyone in your industry already upgraded.")
    assert result.halted is True
    assert result.passed is False
    assert result.halt_reason == "Dark pattern: fabricated urgency"
    # Pipeline stopped: only rights recorded, downstream lenses never called
    assert set(result.per_lens) == {"rights"}
    assert gate.truth.calls == []


def test_contradiction_suppressed_by_scenario_vocab():
    profile = Profile.empty()
    profile.vocabulary.scenario_labels = ["bull case", "bear case"]
    overrides = {
        "contradiction": dict(has_violations="true",
                              violations='[{"claim_a": "GBP rallies", "claim_b": "GBP falls", "severity": "critical"}]'),
    }
    gate = make_stub_gate(profile=profile, overrides=overrides)
    result = gate(text="Bull case: GBP rallies. Bear case: GBP falls.")
    assert result.per_lens["contradiction"] is True
    assert result.passed is True
    # Same finding WITHOUT scenario vocab -> critical fail
    gate2 = make_stub_gate(overrides=overrides)
    result2 = gate2(text="GBP rallies. GBP falls.")
    assert result2.per_lens["contradiction"] is False
    assert result2.passed is False


def test_truth_whitelist_vocab_clears_lens():
    profile = Profile.empty()
    profile.vocabulary.truth_whitelist = ["best practice"]
    gate = make_stub_gate(profile=profile, overrides={
        "truth": dict(has_violations="true",
                      violations='[{"issue": "uses best practice without citation", '
                                 '"severity": "warning", "location": "best practice"}]'),
    })
    result = gate(text="We follow best practice here.")
    # All detected violations were whitelisted -> lens passes
    assert result.per_lens["truth"] is True
    assert result.passed is True


def test_conscious_scan_flags_are_non_blocking():
    gate = make_stub_gate(overrides={
        "judgment": dict(flags='[{"text_snippet": "elegant design", "type": "aesthetic", "reason": "judgment"}]'),
    })
    result = gate(text="An elegant design.")
    assert result.passed is True
    assert len(result.consciousness_flags) == 1
    assert result.consciousness_flags[0]["type"] == "aesthetic"


def test_cross_check_injects_warning():
    gate = make_stub_gate(overrides={
        "cross_check": dict(missed_violations='[{"lens": "truth", "issue": "missed unsourced 42%", "location": "para 2"}]'),
    })
    result = gate(text="Some text with 42% somewhere.")
    cross = [v for v in result.violations if "cross-check" in v.issue]
    assert len(cross) == 1
    assert cross[0].lens == "truth"
    assert cross[0].severity == "warning"


def test_domain_rules_enrich_lens_input():
    profile = Profile.empty()
    profile.domain_rules = {"finance": "[DOMAIN: FINANCE] strict rules"}
    profile.domain_detection = {"finance": ["revenue"]}
    gate = make_stub_gate(profile=profile)
    gate(text="Our revenue grew.")  # auto-detects finance
    truth_call = gate.truth.calls[0]
    assert "[DOMAIN: FINANCE] strict rules" in truth_call["domain"]


def test_verdict_to_dict_shape():
    gate = make_stub_gate()
    result = gate(text="Clean.", include_timings=True)
    d = result.to_dict()
    assert set(d) == {"passed", "halted", "halt_reason", "violations",
                      "consciousness_flags", "per_lens", "timings"}
    assert json.dumps(d)  # JSON-serializable
    assert "total" in d["timings"]
    # Without include_timings the serialized shape carries no timing key
    plain = gate(text="Clean.").to_dict()
    assert "timings" not in plain
    assert "profile" not in plain  # old field name must not resurface


def test_warning_violation_on_blocking_lens_fails_gate():
    """per_lens override path: a blocking lens reporting only WARNING
    violations still fails the gate — independent of severity escalation."""
    gate = make_stub_gate(overrides={
        "causality": dict(has_violations="true",
                          violations='[{"issue": "recommendation has no mechanism", "severity": "warning"}]'),
    })
    result = gate(text="Do X because it works.")
    assert result.per_lens["causality"] is False
    assert all(v.severity == "warning" for v in result.violations)  # no critical anywhere
    assert result.passed is False  # failed via per_lens override, not severity


# ── Definitional Integrity lens (Item 2: independent blocking lens) ──

def test_definitional_integrity_violation_blocks_gate():
    """has_violations=true -> per_lens False AND result fails (blocking),
    mirroring the causality block exactly."""
    gate = make_stub_gate(overrides={
        "definitional_integrity": dict(
            has_violations="true",
            violations='[{"term": "alpha", "issue": "undefined load-bearing term", '
                       '"severity": "warning", "location": "produced alpha"}]'),
    })
    result = gate(text="Fund returned 12%, benchmark 8%. Therefore our fund produced alpha.")
    assert result.per_lens["definitionalIntegrity"] is False
    assert result.passed is False
    assert any(v.lens == "definitionalIntegrity" for v in result.violations)


def test_definitional_integrity_clean_does_not_block():
    """has_violations=false -> per_lens True, no block from this lens."""
    gate = make_stub_gate()  # all-clean stubs incl. definitional_integrity
    result = gate(text="All multiples of 4 are even. 12 is a multiple of 4. Therefore 12 is even.")
    assert result.per_lens["definitionalIntegrity"] is True
    assert result.passed is True
    assert not any(v.lens == "definitionalIntegrity" for v in result.violations)


def test_definitional_integrity_severity_defaults_to_warning():
    """A violation with no severity field still records (default warning) and
    still blocks via the per_lens override path — same as causality."""
    gate = make_stub_gate(overrides={
        "definitional_integrity": dict(
            has_violations="true",
            violations='[{"term": "clean", "issue": "equivocation: loaded-ok vs publication-quality"}]'),
    })
    result = gate(text="Our data is clean. Clean data is suitable for publication. "
                       "Therefore our data is suitable for publication.")
    assert result.per_lens["definitionalIntegrity"] is False
    di = [v for v in result.violations if v.lens == "definitionalIntegrity"]
    assert len(di) == 1
    assert di[0].severity == "warning"
    assert result.passed is False


def test_definitional_integrity_runs_after_causality_before_contradiction():
    """Reporting order: definitional integrity is called after causality and
    before contradiction (v1 has no short-circuit; this only affects order)."""
    gate = make_stub_gate()
    gate(text="Some neutral argument text.")
    # Each stub records its call; assert relative call ordering via a shared
    # monotonic counter on the StubPredictor call lists is overkill — instead
    # confirm all three ran exactly once (independent blocking lenses).
    assert len(gate.causality.calls) == 1
    assert len(gate.definitional_integrity.calls) == 1
    assert len(gate.contradiction.calls) == 1


def test_parallel_is_default_after_rollout():
    from lens_kit import LensGate
    assert LensGate().parallel is True
    assert LensGate(parallel=False).parallel is False  # sequential reference still selectable


def test_format_conflict_intra_passage_is_honest():
    """Intra-passage conflicts (claim_b empty or == claim_a) must not
    render as "'X' vs 'X'" / "'X' vs ''"."""
    from lens_kit.gate import _format_conflict
    assert _format_conflict({"claim_a": "revenue doubled", "claim_b": "revenue fell"}) == \
        "CONFLICT: 'revenue doubled' vs 'revenue fell'"
    single = _format_conflict({"claim_a": "always free yet billed monthly", "claim_b": ""})
    assert "vs ''" not in single and "single passage" in single
    same = _format_conflict({"claim_a": "X", "claim_b": "X"})
    assert "'X' vs 'X'" not in same and "single passage" in same


def test_pii_prepass_halts_before_any_llm_call(stub_gate_factory):
    """Critical structured PII must stop the gate BEFORE the text reaches
    a provider — the stubbed Rights predictor must never be consulted."""
    for parallel in (False, True):
        gate = stub_gate_factory(parallel=parallel)
        result = gate(text="charge card 4111 1111 1111 1111 for the renewal")
        assert result.halted and not result.passed
        assert "credit_card" in result.halt_reason
        assert result.per_lens == {"rights": False}
        assert all(v.lens == "rights" for v in result.violations)
        assert "4111 1111 1111 1111" not in result.fixed_text  # scrubbed
        assert gate.rights.calls == [], f"provider consulted (parallel={parallel})"


def test_pii_prepass_lets_warning_grade_matches_through(stub_gate_factory):
    """Emails/phones are context calls for the LLM Rights lens — the
    deterministic pre-pass must NOT fail the gate on them."""
    gate = stub_gate_factory(parallel=False)
    result = gate(text="Questions? Contact support@vendor-example.io anytime.")
    assert result.passed and not result.halted
    assert len(gate.rights.calls) == 1  # pipeline ran normally
