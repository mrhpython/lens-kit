"""Calibration battery: generator determinism, gold correctness, planted-flaw
presence, verdict mapping, and the runner exit codes — all no-network.

The runner is driven by conftest stub gates (LLM predictors stubbed); the
CLI/runner-level tests monkeypatch calibration.make_gate. No live API calls.
"""
import json

import pytest

from lens_kit.calibration import (CALIB_TEMPLATES_VERSION, CLASSES,
                                  EXPERIMENTAL_CLASSES, FLAW_MARKERS,
                                  GOLD_LENS_LABELS, ORIGINAL_CLASSES, RULES,
                                  _fn_rate, generate, load_battery, map_verdict,
                                  run_battery, score_fixture, summarize_run)
from lens_kit.config import Profile, builtin_profile_path
from lens_kit.gate import LensResult, LensViolation
from tests.conftest import make_stub_gate


def _profile():
    return Profile.load(builtin_profile_path("agency-example"))


# ── Generator determinism (same seed = identical bytes) ──────────

def test_generate_is_byte_identical_for_same_seed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(_profile(), a, seed=7)
    generate(_profile(), b, seed=7)
    for fa in sorted(a.glob("*")):
        fb = b / fa.name
        assert fb.exists()
        assert fa.read_bytes() == fb.read_bytes(), f"{fa.name} differs across runs"


def test_generate_emits_24_fixtures_with_gold_sidecars(tmp_path):
    written = generate(_profile(), tmp_path)
    assert len(written) == 24                       # 8 classes x 3
    assert len(list(tmp_path.glob("*.txt"))) == 24
    assert len(list(tmp_path.glob("*.gold.json"))) == 24
    for txt in tmp_path.glob("*.txt"):
        assert txt.with_name(txt.stem + ".gold.json").exists()


def test_generate_per_class_controls_count(tmp_path):
    generate(_profile(), tmp_path, per_class=2)
    assert len(list(tmp_path.glob("*.txt"))) == 16  # 8 classes x 2


def test_generate_per_class_above_template_count_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(_profile(), a, per_class=5, seed=3)
    generate(_profile(), b, per_class=5, seed=3)
    assert len(list(a.glob("*.txt"))) == 40         # 8 x 5, padded by seeded rng
    for fa in sorted(a.glob("*")):
        assert fa.read_bytes() == (b / fa.name).read_bytes()


# ── Per-class gold correctness ───────────────────────────────────

def test_gold_sidecar_matches_class_rules(tmp_path):
    generate(_profile(), tmp_path)
    for gold_path in tmp_path.glob("*.gold.json"):
        gold = json.loads(gold_path.read_text())
        cls = gold["class"]
        acceptable, target = RULES[cls]
        assert set(gold["acceptable_verdicts"]) == acceptable
        assert gold["target_verdict"] == target
        assert gold["gold_lens_labels"] == GOLD_LENS_LABELS[cls]
        assert gold["flaw_marker"] == FLAW_MARKERS[cls]
        # text_sha256 actually matches the fixture bytes
        body = (tmp_path / gold["fixture"]).read_bytes()
        import hashlib
        assert gold["text_sha256"] == hashlib.sha256(body).hexdigest()


def test_every_class_present(tmp_path):
    generate(_profile(), tmp_path)
    seen = {json.loads(g.read_text())["class"] for g in tmp_path.glob("*.gold.json")}
    assert seen == set(CLASSES)


# ── Template flaw presence (grep-able structural markers) ────────

def test_contradiction_fixtures_plant_two_incompatible_claims(tmp_path):
    generate(_profile(), tmp_path)
    texts = [p.read_text().lower() for p in tmp_path.glob("contradiction-*.txt")]
    assert texts
    # Each contradiction fixture pairs an absolute claim with its negation.
    joined = " ".join(texts)
    assert "fully automated" in joined and "manually review" in joined
    assert "free" in joined and ("billed" in joined or "charge" in joined)
    # timeline: "within 24 hours" (immediate) contradicted by a slow setup period
    assert "24 hours" in joined and "starts working immediately" in joined


def test_darkpattern_fixtures_plant_false_scarcity_or_forced_continuity(tmp_path):
    generate(_profile(), tmp_path)
    joined = " ".join(p.read_text().lower() for p in tmp_path.glob("darkpattern-*.txt"))
    assert "midnight tonight" in joined          # false-scarcity deadline
    assert "12-month minimum term" in joined     # forced continuity vs "cancel anytime"
    assert "cancel anytime" in joined


def test_unsourced_fixtures_plant_specific_uncited_statistics(tmp_path):
    generate(_profile(), tmp_path)
    joined = " ".join(p.read_text() for p in tmp_path.glob("unsourced-*.txt"))
    assert "%" in joined                          # uncited percentages
    assert "Studies show" in joined               # appeal to phantom evidence


def test_clean_fixtures_carry_no_manipulation_or_uncited_stats(tmp_path):
    generate(_profile(), tmp_path)
    for p in tmp_path.glob("clean-*.txt"):
        low = p.read_text().lower()
        assert "midnight tonight" not in low
        assert "studies show" not in low


# ── Verdict mapping edge cases ───────────────────────────────────

def _viol(lens, severity="warning", issue=""):
    return LensViolation(lens=lens, severity=severity, issue=issue)


def test_map_passed_is_ship():
    r = LensResult(passed=True, per_lens={"truth": True, "rights": True})
    assert map_verdict(r) == "SHIP"


def test_map_halt_is_hold():
    r = LensResult(passed=False, halted=True, per_lens={"rights": False},
                   violations=[_viol("rights", "critical", "PII leak")])
    assert map_verdict(r) == "HOLD"


def test_map_critical_violation_is_hold():
    r = LensResult(passed=False, per_lens={"contradiction": False},
                   violations=[_viol("contradiction", "critical", "CONFLICT a vs b")])
    assert map_verdict(r) == "HOLD"


def test_map_truth_only_with_source_marker_is_needs_sources():
    r = LensResult(passed=False, per_lens={"truth": False, "rights": True},
                   violations=[_viol("truth", "warning", "Claim lacks a source citation")])
    assert map_verdict(r) == "NEEDS_SOURCES"


def test_map_truth_only_fabrication_without_source_marker_is_hold():
    r = LensResult(passed=False, per_lens={"truth": False, "rights": True},
                   violations=[_viol("truth", "warning", "Fabricated invented figure")])
    assert map_verdict(r) == "HOLD"


def test_map_truth_plus_other_lens_is_hold_not_needs_sources():
    r = LensResult(passed=False, per_lens={"truth": False, "causality": False},
                   violations=[_viol("truth", "warning", "lacks source"),
                               _viol("causality", "warning", "broken chain")])
    assert map_verdict(r) == "HOLD"


def test_map_relevance_only_does_not_block_ship():
    # relevance is warning-only: passed stays True, so SHIP.
    r = LensResult(passed=True, per_lens={"relevance": False, "truth": True},
                   violations=[_viol("relevance", "warning", "audience mismatch")])
    assert map_verdict(r) == "SHIP"


def test_map_rights_warning_only_without_halt_is_hold():
    # Rights can fail at warning severity WITHOUT halting (per_lens override
    # path in the gate). Still a non-truth blocker -> HOLD, never
    # NEEDS_SOURCES, even if the issue text happens to mention a source word.
    r = LensResult(passed=False, halted=False,
                   per_lens={"rights": False, "truth": True},
                   violations=[_viol("rights", "warning",
                                     "manipulative urgency, no source cited")])
    assert map_verdict(r) == "HOLD"


# ── score_fixture confusion vs gold ──────────────────────────────

def test_score_fixture_tp_when_planted_flaw_caught():
    r = LensResult(passed=False, per_lens={"contradiction": False})
    row = score_fixture("contradiction", "HOLD", r, GOLD_LENS_LABELS["contradiction"])
    assert row["acceptable"] and row["on_target"]
    assert row["lens_confusion"]["contradiction"] == "tp"
    assert row["mismatches"] == []


def test_score_fixture_fn_when_planted_flaw_missed():
    r = LensResult(passed=True, per_lens={"contradiction": True})
    row = score_fixture("contradiction", "SHIP", r, GOLD_LENS_LABELS["contradiction"])
    assert not row["acceptable"]
    assert row["lens_confusion"]["contradiction"] == "fn"
    assert row["mismatches"] == ["contradiction: FN (expected FAIL, got PASS)"]


def test_score_fixture_fp_when_clean_wrongly_flagged():
    r = LensResult(passed=False, per_lens={lens: (lens != "truth")
                                           for lens in GOLD_LENS_LABELS["clean"]})
    row = score_fixture("clean", "HOLD", r, GOLD_LENS_LABELS["clean"])
    assert not row["acceptable"]
    assert row["lens_confusion"]["truth"] == "fp"


def test_score_fixture_unevaluated_lens_is_none(tmp_path):
    # rights label present but the gate never reported rights (post-HALT shape)
    r = LensResult(passed=False, per_lens={"truth": False})
    row = score_fixture("darkpattern", "HOLD", r, GOLD_LENS_LABELS["darkpattern"])
    assert row["lens_confusion"]["rights"] is None


# ── summarize_run fn_rate / fp_rate semantics ────────────────────

def _row(cls, verdict, acceptable, on_target=False):
    return {"class": cls, "verdict": verdict, "acceptable": acceptable,
            "on_target": on_target}


def test_summarize_unsourced_needs_sources_is_not_fn():
    s = summarize_run([_row("unsourced", "NEEDS_SOURCES", True, on_target=True)])
    assert s["fn_rate"] == 0.0          # NEEDS_SOURCES is a catch for unsourced


def test_summarize_unsourced_ship_is_fn():
    s = summarize_run([_row("unsourced", "SHIP", False)])
    assert s["fn_rate"] == 1.0          # silent SHIP on unsourced = the worst miss


def test_summarize_contradiction_non_hold_is_fn():
    s = summarize_run([_row("contradiction", "NEEDS_SOURCES", False)])
    assert s["fn_rate"] == 1.0          # only HOLD catches contradiction


def test_summarize_clean_block_is_fp():
    s = summarize_run([_row("clean", "HOLD", False)])
    assert s["fp_rate"] == 1.0


# ── load_battery hard errors ─────────────────────────────────────

def test_load_battery_missing_gold_is_hard_error(tmp_path):
    (tmp_path / "clean-01-coldemail.txt").write_text("body text here\n")
    with pytest.raises(ValueError, match="no gold sidecar"):
        load_battery(tmp_path)


def test_load_battery_empty_dir_is_error(tmp_path):
    with pytest.raises(ValueError, match="No .txt fixtures"):
        load_battery(tmp_path)


# ── Runner exit codes (stubbed gate, no network) ─────────────────

def _stub_run(monkeypatch, tmp_path, overrides):
    """Generate a battery, point make_gate at a stub gate, run it."""
    import lens_kit.calibration as cal
    profile = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"},
                                 "calibration": _profile().calibration.to_dict()})
    battery = tmp_path / "battery"
    cal.generate(profile, battery)
    monkeypatch.setattr(cal, "make_gate",
                        lambda prof: make_stub_gate(overrides=overrides))
    return cal.run_battery(battery, profile, output_dir=tmp_path / "out",
                           command_str="lens-kit calibrate run test"), tmp_path


def test_runner_all_clean_stub_has_failures_exit_5(monkeypatch, tmp_path):
    # An all-PASS stub SHIPs everything: clean is acceptable, but the planted
    # flaws (contradiction/darkpattern/unsourced) all wrongly SHIP -> failures.
    # Exit 5 (battery failures), not 3 (3 is the compile costing-gate stop).
    rc, _ = _stub_run(monkeypatch, tmp_path, overrides=None)
    assert rc == 5


def test_runner_writes_versioned_results_and_ledger(monkeypatch, tmp_path):
    rc, root = _stub_run(monkeypatch, tmp_path, overrides=None)
    assert rc == 5
    runs = sorted((root / "out").glob("calib-*-run*.json"))
    assert len(runs) == 1
    doc = json.loads(runs[0].read_text())
    assert doc["calib_schema_version"] == 1
    assert "summary" in doc and "results" in doc
    from pathlib import Path
    assert (Path.cwd() / "RUNS.md").exists()
    assert "CALIB" in (Path.cwd() / "RUNS.md").read_text()


def test_runner_envelope_reuse_on_reruns(monkeypatch, tmp_path):
    import lens_kit.calibration as cal
    profile = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"},
                                 "calibration": _profile().calibration.to_dict()})
    battery = tmp_path / "battery"
    cal.generate(profile, battery)
    monkeypatch.setattr(cal, "make_gate", lambda prof: make_stub_gate())
    rc = cal.run_battery(battery, profile, reruns=2, output_dir=tmp_path / "out",
                         command_str="lens-kit calibrate run test")
    assert rc == 5
    envs = sorted((tmp_path / "out").glob("calib-*-envelope*.json"))
    assert len(envs) == 1
    env = json.loads(envs[0].read_text())
    # Envelope reuses eval_harness.build_envelope with calibration directions.
    assert env["metrics"]["fp_rate"]["direction"] == "lower_is_better"
    assert env["metrics"]["acceptable_rate"]["direction"] == "higher_is_better"
    assert env["cache_caveat"]                      # shared-cache caveat recorded


def test_runner_all_acceptable_exit_0(monkeypatch, tmp_path):
    """A stub gate that blocks the planted-flaw classes AND passes clean
    yields every fixture acceptable -> exit 0."""
    import lens_kit.calibration as cal
    profile = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"},
                                 "calibration": _profile().calibration.to_dict()})
    battery = tmp_path / "battery"
    cal.generate(profile, battery)

    # Stub per-class: contradiction fixtures trip contradiction (critical=HOLD);
    # darkpattern trip rights HALT; unsourced trip truth with a source marker
    # (=NEEDS_SOURCES); clean stay all-PASS (=SHIP).
    def gate_for(prof):
        return _ClassRoutingGate()

    monkeypatch.setattr(cal, "make_gate", gate_for)
    rc = cal.run_battery(battery, profile, output_dir=tmp_path / "out",
                         command_str="lens-kit calibrate run test")
    assert rc == 0


_CONTRADICTION_MARKERS = ("fully automated", "completely free", "starts working immediately")
_DARKPATTERN_MARKERS = ("midnight tonight", "12-month minimum term", "watch your calendar")
_UNSOURCED_MARKERS = ("studies show", "growing 38% per year", "revenue per message rose 42%")
# Experimental classes (Machine-Bullshit taxonomy) — distinctive template phrases.
_PALTERING_MARKERS = ("every word true", "literally accurate", "perfect record")
_SYCOPHANTIC_MARKERS = ("what a brilliant team", "quality when you see it",
                        "outstanding decisions")
_WEASEL_MARKERS = ("experts suggest", "widely recognized", "commonly believed")
_EMPTYRHETORIC_MARKERS = ("best-in-class solution", "revolutionize how",
                          "exceptional value")


class _ClassRoutingGate:
    """Offline fake gate: returns a class-appropriate LensResult by inspecting
    the planted flaw in the fixture text. No LLM. Drives the exit-0 path:
    contradiction -> HOLD, darkpattern -> HALT(HOLD), unsourced -> NEEDS_SOURCES,
    paltering -> HOLD, sycophantic -> HOLD, weasel/emptyrhetoric -> NEEDS_SOURCES,
    clean -> SHIP."""

    def __call__(self, *, text, domain="general", context=""):
        low = text.lower()
        if any(m in low for m in _CONTRADICTION_MARKERS):
            return LensResult(passed=False, per_lens={"contradiction": False},
                              violations=[_viol("contradiction", "critical", "CONFLICT")])
        if any(m in low for m in _DARKPATTERN_MARKERS):
            return LensResult(passed=False, halted=True, per_lens={"rights": False},
                              violations=[_viol("rights", "critical", "manipulation")])
        if any(m in low for m in _PALTERING_MARKERS):
            return LensResult(passed=False,
                              per_lens={"truth": False, "contradiction": False, "rights": True},
                              violations=[_viol("contradiction", "critical", "implies false whole")])
        if any(m in low for m in _SYCOPHANTIC_MARKERS):
            return LensResult(passed=False,
                              per_lens={"truth": False, "relevance": False, "rights": True},
                              violations=[_viol("truth", "critical", "flattery without evidence")])
        if any(m in low for m in _UNSOURCED_MARKERS + _WEASEL_MARKERS + _EMPTYRHETORIC_MARKERS):
            return LensResult(passed=False, per_lens={"truth": False, "rights": True},
                              violations=[_viol("truth", "warning", "uncited statistic, lacks source")])
        # clean control
        return LensResult(passed=True, per_lens={lens: True for lens in GOLD_LENS_LABELS["clean"]})


# ══════════════════════════════════════════════════════════════════
# Item 1 — Machine-Bullshit experimental classes (arXiv 2507.07484)
# ══════════════════════════════════════════════════════════════════

def test_classes_split_original_plus_experimental():
    assert ORIGINAL_CLASSES == ("clean", "contradiction", "darkpattern", "unsourced")
    assert EXPERIMENTAL_CLASSES == ("weasel", "paltering", "emptyrhetoric", "sycophantic")
    assert CLASSES == ORIGINAL_CLASSES + EXPERIMENTAL_CLASSES


def test_templates_version_bumped_to_v3():
    assert CALIB_TEMPLATES_VERSION == "v3-2026-06-13"


def test_every_experimental_class_has_rules_labels_markers_templates(tmp_path):
    # Generation must emit each experimental class with a full gold sidecar.
    generate(_profile(), tmp_path)
    for cls in EXPERIMENTAL_CLASSES:
        assert cls in RULES
        assert cls in GOLD_LENS_LABELS
        assert cls in FLAW_MARKERS
        fixtures = sorted(tmp_path.glob(f"{cls}-*.txt"))
        assert len(fixtures) == 3, f"{cls} must ship >=3 templates"


def test_generate_emits_24_fixtures_with_8_classes(tmp_path):
    written = generate(_profile(), tmp_path)
    assert len(written) == 24                       # 8 classes x 3
    seen = {json.loads(g.read_text())["class"] for g in tmp_path.glob("*.gold.json")}
    assert seen == set(CLASSES)


def test_experimental_gold_sidecars_carry_experimental_flag(tmp_path):
    generate(_profile(), tmp_path)
    for gold_path in tmp_path.glob("*.gold.json"):
        gold = json.loads(gold_path.read_text())
        expected = gold["class"] in EXPERIMENTAL_CLASSES
        assert gold["experimental"] is expected, gold["class"]


def test_experimental_generate_is_byte_identical_for_same_seed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(_profile(), a, seed=11)
    generate(_profile(), b, seed=11)
    for cls in EXPERIMENTAL_CLASSES:
        for fa in sorted(a.glob(f"{cls}-*")):
            assert fa.read_bytes() == (b / fa.name).read_bytes(), fa.name


# ── Planted-defect gold-label correctness, per experimental class ──
# Each test proves: when the gate FAILS exactly the lenses the gold names,
# the row scores those lenses as true-positives (tp) with no mismatch — i.e.
# the gold labels mark the right lenses FAIL.

def _per_lens_failing(lenses):
    """A per_lens map where the named lenses FAIL (False), the rest PASS."""
    out = {lens: True for lens in
           ("rights", "truth", "causality", "contradiction",
            "extrapolation", "structure", "consistency", "relevance")}
    for lens in lenses:
        out[lens] = False
    return out


def test_weasel_gold_marks_truth_fail():
    assert GOLD_LENS_LABELS["weasel"] == {"truth": False}
    r = LensResult(passed=False, per_lens=_per_lens_failing(["truth"]),
                   violations=[_viol("truth", "warning", "lacks a source citation")])
    row = score_fixture("weasel", "NEEDS_SOURCES", r, GOLD_LENS_LABELS["weasel"])
    assert row["lens_confusion"]["truth"] == "tp"
    assert row["mismatches"] == []
    assert row["acceptable"]            # NEEDS_SOURCES is in weasel's acceptable set


def test_paltering_gold_marks_truth_and_contradiction_fail():
    assert GOLD_LENS_LABELS["paltering"] == {"truth": False, "contradiction": False}
    r = LensResult(passed=False, per_lens=_per_lens_failing(["truth", "contradiction"]),
                   violations=[_viol("contradiction", "critical", "implies a false whole")])
    row = score_fixture("paltering", "HOLD", r, GOLD_LENS_LABELS["paltering"])
    assert row["lens_confusion"]["truth"] == "tp"
    assert row["lens_confusion"]["contradiction"] == "tp"
    assert row["mismatches"] == []
    assert row["acceptable"] and row["on_target"]   # HOLD is paltering's target


def test_emptyrhetoric_gold_marks_truth_fail():
    assert GOLD_LENS_LABELS["emptyrhetoric"] == {"truth": False}
    r = LensResult(passed=False, per_lens=_per_lens_failing(["truth"]),
                   violations=[_viol("truth", "warning", "no checkable content, unsupported")])
    row = score_fixture("emptyrhetoric", "NEEDS_SOURCES", r,
                        GOLD_LENS_LABELS["emptyrhetoric"])
    assert row["lens_confusion"]["truth"] == "tp"
    assert row["mismatches"] == []


def test_sycophantic_gold_marks_relevance_and_truth_fail():
    assert GOLD_LENS_LABELS["sycophantic"] == {"relevance": False, "truth": False}
    r = LensResult(passed=False, per_lens=_per_lens_failing(["relevance", "truth"]),
                   violations=[_viol("truth", "warning", "agreement asserted without evidence")])
    row = score_fixture("sycophantic", "HOLD", r, GOLD_LENS_LABELS["sycophantic"])
    assert row["lens_confusion"]["relevance"] == "tp"
    assert row["lens_confusion"]["truth"] == "tp"
    assert row["mismatches"] == []


def test_experimental_gold_lenses_are_all_canonical():
    from lens_kit.gate import CANONICAL_LENSES
    for cls in EXPERIMENTAL_CLASSES:
        for lens in GOLD_LENS_LABELS[cls]:
            assert lens in CANONICAL_LENSES, f"{cls}: {lens} not canonical"


# ── Headline-exclusion regression: experimental classes NEVER move the
#    headline fn_rate; they appear only under experimental_fn_rate ──

def _row(cls, verdict, acceptable, on_target=False, experimental=None):
    if experimental is None:
        experimental = cls in EXPERIMENTAL_CLASSES
    return {"class": cls, "experimental": experimental, "verdict": verdict,
            "acceptable": acceptable, "on_target": on_target}


def test_headline_fn_rate_unchanged_by_original_four_classes():
    # The four ORIGINAL classes alone produce the SAME headline as before:
    # a missed contradiction (SHIP) over one original flaw -> fn_rate 1.0.
    rows = [_row("contradiction", "SHIP", False)]
    s = summarize_run(rows)
    assert s["fn_rate"] == 1.0
    # And a caught contradiction -> 0.0 (regression-identical to pre-Item-1).
    s2 = summarize_run([_row("contradiction", "HOLD", True, on_target=True)])
    assert s2["fn_rate"] == 0.0


def test_experimental_misses_excluded_from_headline_fn_rate():
    # One caught original flaw (HOLD) + four MISSED experimental flaws (SHIP).
    rows = [_row("contradiction", "HOLD", True, on_target=True),
            _row("weasel", "SHIP", False),
            _row("paltering", "SHIP", False),
            _row("emptyrhetoric", "SHIP", False),
            _row("sycophantic", "SHIP", False)]
    s = summarize_run(rows)
    # Headline sees ONLY the original flaw, which was caught -> 0.0, NOT 0.8.
    assert s["fn_rate"] == 0.0
    # The experimental misses are surfaced separately, every one at 1.0.
    assert s["experimental_fn_rate"] == {
        "weasel": 1.0, "paltering": 1.0, "emptyrhetoric": 1.0, "sycophantic": 1.0}


def test_experimental_fn_rate_block_present_and_typed():
    rows = [_row("weasel", "HOLD", True), _row("paltering", "SHIP", False)]
    s = summarize_run(rows)
    assert "experimental_fn_rate" in s
    assert s["experimental_fn_rate"]["weasel"] == 0.0      # HOLD catches weasel
    assert s["experimental_fn_rate"]["paltering"] == 1.0   # SHIP misses paltering
    # No original flaws present -> headline fn_rate is None, not skewed by exp.
    assert s["fn_rate"] is None


def test_headline_only_counts_original_when_both_present():
    # A missed unsourced (SHIP) among originals drives the headline; the
    # experimental sycophantic miss does NOT.
    rows = [_row("unsourced", "SHIP", False),
            _row("sycophantic", "SHIP", False)]
    s = summarize_run(rows)
    assert s["fn_rate"] == 1.0                  # 1 original miss / 1 original flaw
    assert s["experimental_fn_rate"]["sycophantic"] == 1.0


def test_experimental_fixtures_still_visible_in_acceptable_rate():
    # acceptable_rate spans ALL fixtures so a failing experimental fixture is
    # not invisible — it just doesn't move the headline fn denominator.
    rows = [_row("clean", "SHIP", True), _row("weasel", "SHIP", False)]
    s = summarize_run(rows)
    assert s["acceptable_rate"] == 0.5


# ── FIX 2: the experimental-exclusion invariant is LOCAL to _fn_rate ──
# headline_only=True must fail closed if an experimental row reaches the
# headline path — the guard can't be only the caller's pre-filter.

def test_fn_rate_headline_only_rejects_experimental_row():
    # A mixed list reaching the headline path MUST raise, not silently fold the
    # experimental miss into the headline count.
    mixed = [_row("contradiction", "HOLD", True),
             _row("weasel", "SHIP", False)]      # experimental miss
    with pytest.raises(AssertionError, match="experimental"):
        _fn_rate(mixed, headline_only=True)


def test_fn_rate_default_counts_experimental_rows_normally():
    # The per-experimental-class path (no headline_only) counts them as usual.
    rows = [_row("weasel", "SHIP", False), _row("paltering", "HOLD", True)]
    misses, denom = _fn_rate(rows)            # default: headline_only=False
    assert (misses, denom) == (1, 2)          # weasel SHIP misses, paltering HOLD catches


def test_fn_rate_headline_only_passes_clean_original_only_list():
    # The legitimate headline call (pre-filtered to original flaws) is unaffected.
    rows = [_row("contradiction", "SHIP", False), _row("unsourced", "HOLD", True)]
    misses, denom = _fn_rate(rows, headline_only=True)
    assert (misses, denom) == (1, 2)


def test_summarize_run_never_folds_experimental_into_headline_under_mix():
    # End-to-end: even a heavy experimental-miss mix leaves the headline driven
    # ONLY by the single original flaw (proves the guard + pre-filter agree).
    rows = [_row("contradiction", "HOLD", True, on_target=True)]
    rows += [_row(c, "SHIP", False) for c in EXPERIMENTAL_CLASSES] * 3
    s = summarize_run(rows)
    assert s["fn_rate"] == 0.0                # 1 original flaw, caught
    for c in EXPERIMENTAL_CLASSES:
        assert s["experimental_fn_rate"][c] == 1.0
