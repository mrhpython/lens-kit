"""Deterministic filter layer — profile vocabulary application, no LLM."""
from lens_kit import DeterministicFilters, LensVocabulary, detect_domain


def _flt(**vocab):
    return DeterministicFilters(LensVocabulary(**vocab))


def test_filters_take_defensive_vocab_copy():
    vocab = LensVocabulary(truth_whitelist=["best practice"])
    flt = DeterministicFilters(vocab)
    # Post-construction mutation of the caller's vocabulary must not
    # propagate into filter behavior (calibration-loop safety).
    vocab.truth_whitelist.append("roi claim")
    v = [{"issue": "roi claim uncited", "severity": "warning"}]
    assert flt.filter_truth_whitelist(v) == v          # new term NOT applied
    w = [{"issue": "uses best practice", "severity": "warning"}]
    assert flt.filter_truth_whitelist(w) == []         # original term still applied


def test_truth_whitelist_suppresses_matching_issue():
    flt = _flt(truth_whitelist=["best practice"])
    violations = [
        {"issue": "Uses 'best practice' without citation", "severity": "warning"},
        {"issue": "7.5% ECB rate has no named source", "severity": "critical"},
    ]
    kept = flt.filter_truth_whitelist(violations)
    assert len(kept) == 1
    assert "7.5%" in kept[0]["issue"]


def test_empty_vocab_suppresses_nothing():
    flt = _flt()
    violations = [{"issue": "best practice claim", "severity": "warning"}]
    assert flt.filter_truth_whitelist(violations) == violations
    assert flt.filter_truth_named_source(violations, "text") == violations
    assert flt.filter_truth_first_party(violations, "our revenue grew") == violations
    assert flt.filter_truth_forward_looking(violations, "text") == violations
    assert flt.filter_extrapolation_past_tense(violations, "text") == violations
    assert flt.filter_extrapolation_qualified(violations, "text") == violations
    assert flt.filter_structure_citation_overlap(violations) == violations
    assert flt.filter_structure_stated_basis(violations, "text") == violations
    assert not flt.is_scenario_text("bull case: rates fall. bear case: rates rise.")


def test_scenario_labels_detected():
    flt = _flt(scenario_labels=["bull case", "bear case"])
    assert flt.is_scenario_text("Bull case: GBP rallies. Bear case: GBP falls.")
    assert not flt.is_scenario_text("GBP will rally to 1.40.")


def test_first_party_suppression_with_external_override():
    flt = _flt(
        first_party_signals=["our "],
        first_party_context=["our churn rate"],
        external_claim_signals=["market share"],
    )
    text = "Our churn rate held at 2% this quarter."
    suppressed = [{"issue": "2% churn uncited", "location": "our churn rate held at 2%"}]
    assert flt.filter_truth_first_party(suppressed, text) == []
    # External-facing claims are never suppressed
    external = [{"issue": "claims 30% market share uncited", "location": "our churn rate held at 2%"}]
    assert flt.filter_truth_first_party(external, text) == external
    # Non-first-party text: no filtering at all
    assert flt.filter_truth_first_party(suppressed, "The churn rate held at 2%.") == suppressed


def test_forward_looking_routes_to_extrapolation():
    flt = _flt(
        future_signals=["will ", "by 202"],
        present_tense_override=["has already"],
    )
    violations = [
        {"issue": "claims revenue will reach $5M by 2028", "location": ""},
        {"issue": "claims AI has already replaced 40% of staff", "location": ""},
        {"issue": "7.5% rate has no source", "location": ""},
    ]
    kept = flt.filter_truth_forward_looking(violations, "irrelevant")
    issues = [v["issue"] for v in kept]
    assert "claims revenue will reach $5M by 2028" not in issues  # routed to Extrapolation
    assert "claims AI has already replaced 40% of staff" in issues  # present-tense stays
    assert "7.5% rate has no source" in issues  # no temporal signal -> Truth default


def test_past_tense_routes_to_truth_with_future_priority():
    flt = _flt(
        past_tense_signals=["raised "],
        future_counter_signals=["will "],
    )
    text_past = "The ECB raised rates to 7.5% last March."
    v = [{"issue": "7.5% unverified", "location": "raised rates to 7.5%"}]
    assert flt.filter_extrapolation_past_tense(v, text_past) == []
    text_mixed = "The ECB raised rates and will raise them again."
    v2 = [{"issue": "unqualified projection", "location": "raised rates and will raise"}]
    assert flt.filter_extrapolation_past_tense(v2, text_mixed) == v2


def test_extrapolation_qualified_needs_marker_and_confidence():
    flt = _flt(confidence_keywords=["high confidence"])
    qualified = "[PROJECTION] Revenue reaches $5M by 2028 (high confidence, based on pipeline)."
    v = [{"issue": "unqualified projection", "location": "Revenue reaches $5M"}]
    assert flt.filter_extrapolation_qualified(v, qualified) == []
    marker_only = "[PROJECTION] Revenue reaches $5M by 2028."
    assert flt.filter_extrapolation_qualified(v, marker_only) == v


def test_structure_complete_override():
    flt = _flt(precondition_keywords=["precondition"], rollback_keywords=["rollback"])
    text = "Preconditions: staging green. Rollback: redeploy previous tag."
    v = [{"issue": "missing rollback plan"}]
    assert flt.filter_structure_complete(v, text) == []
    assert flt.filter_structure_complete(v, "Just deploy it.") == v


def test_structure_trigger_empty_vocab_always_runs():
    assert _flt().text_has_recommendations("Plain analysis, no plans.") is True
    flt = _flt(recommendation_signals=["deploy "])
    assert flt.text_has_recommendations("We deploy on Friday.") is True
    assert flt.text_has_recommendations("Plain analysis.") is False


def test_structure_citation_overlap():
    flt = _flt(citation_overlap_signals=["lacks source", "citation"])
    v = [
        {"issue": "Plan lacks source for the $2M figure"},
        {"issue": "No rollback plan for the migration"},
    ]
    kept = flt.filter_structure_citation_overlap(v)
    assert len(kept) == 1
    assert "rollback" in kept[0]["issue"]


def test_tabular_filter_is_vocab_free():
    flt = _flt()
    text = "Header\n| Q1 | $2M | up |\nNot a table"
    v = [{"issue": "uncited $2M", "location": "line 1"}]
    assert flt.filter_truth_tabular(text, v) == []


def test_detect_domain_uses_profile_keywords():
    keywords = {"finance": ["revenue", "$"], "seo": ["keyword"]}
    assert detect_domain("Our revenue grew", keywords) == "finance"
    assert detect_domain("keyword rankings improved", keywords) == "seo"
    assert detect_domain("nothing special here", keywords) == "general"
    assert detect_domain("any text at all", {}) == "general"  # empty profile


def test_vocab_round_trip_dict():
    vocab = LensVocabulary(truth_whitelist=["a"], scenario_labels=["b"])
    assert LensVocabulary.from_dict(vocab.to_dict()) == vocab
