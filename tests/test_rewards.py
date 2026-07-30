"""BestOfN reward functions — direct unit tests, no LLM.

threshold=1.0 semantics: BestOfN stops early only when a candidate's
reward meets the threshold. Both reward fns return 1.0 only at 5+
violations, so for any result under 5 violations all N candidates run —
that is the intended "always run all N" behavior the gate relies on.
"""
import json

import dspy
import pytest

from lens_kit import LensGate

truth_reward = LensGate._truth_strictness_reward
extrap_reward = LensGate._extrap_strictness_reward


def _pred(violations) -> dspy.Prediction:
    if isinstance(violations, int):
        violations = json.dumps([{"issue": f"v{i}", "severity": "warning"} for i in range(violations)])
    return dspy.Prediction(violations=violations)


# ── truth reward ─────────────────────────────────────────────────

def test_truth_reward_scales_with_violation_count():
    assert truth_reward({}, _pred(0)) == 0.0
    assert truth_reward({}, _pred(1)) == pytest.approx(0.2)
    assert truth_reward({}, _pred(2)) == pytest.approx(0.4)
    assert truth_reward({}, _pred(4)) == pytest.approx(0.8)


def test_truth_reward_caps_at_five_violations():
    assert truth_reward({}, _pred(5)) == 1.0
    assert truth_reward({}, _pred(9)) == 1.0  # capped, not >1


def test_truth_reward_handles_garbage_output():
    assert truth_reward({}, _pred("not json at all")) == 0.0
    assert truth_reward({}, _pred("")) == 0.0
    assert truth_reward({}, dspy.Prediction()) == 0.0          # missing field
    assert truth_reward({}, _pred('{"issue": "dict not list"}')) == 0.0


def test_truth_reward_accepts_pre_parsed_list():
    assert truth_reward({}, _pred(json.dumps([{"issue": "a"}]))) == pytest.approx(0.2)
    pred = dspy.Prediction(violations=[{"issue": "a"}, {"issue": "b"}])
    assert truth_reward({}, pred) == pytest.approx(0.4)


# ── extrapolation reward ─────────────────────────────────────────

def test_extrap_reward_base_matches_truth_scaling():
    assert extrap_reward({}, _pred(0)) == 0.0
    assert extrap_reward({}, _pred(2)) == pytest.approx(0.4)


def test_extrap_reward_marker_bonus():
    one = json.dumps([{"issue": "needs [PROJECTION] marker"}])
    assert extrap_reward({}, _pred(one)) == pytest.approx(0.2 + 0.1)
    est = json.dumps([{"issue": "needs [ESTIMATE] marker"}])
    assert extrap_reward({}, _pred(est)) == pytest.approx(0.2 + 0.1)


def test_extrap_reward_confidence_bonus_stacks():
    both = json.dumps([{"issue": "needs [PROJECTION] with confidence level"}])
    assert extrap_reward({}, _pred(both)) == pytest.approx(0.2 + 0.1 + 0.05)


def test_extrap_reward_caps_at_one_with_bonuses():
    six = json.dumps([{"issue": f"[PROJECTION] confidence v{i}"} for i in range(6)])
    assert extrap_reward({}, _pred(six)) == 1.0


def test_extrap_reward_garbage_is_zero_but_bonus_still_reads_text():
    # Unparseable JSON -> count 0; bonuses scan the raw string by design.
    assert extrap_reward({}, _pred("totally broken")) == 0.0
    assert extrap_reward({}, _pred("broken but mentions [projection]")) == pytest.approx(0.1)


# ── wiring + threshold semantics ─────────────────────────────────

def test_gate_wires_rewards_with_threshold_one():
    gate = LensGate()  # default = BestOfN mode
    assert gate.truth.threshold == 1.0
    assert gate.truth.N == 3
    assert gate.extrapolation.threshold == 1.0
    assert gate.extrapolation.N == 3
    # BestOfN wraps reward_fn in an internal lambda — identity comparison is
    # impossible, so assert behavioral equivalence through the wrapper.
    marker = _pred(json.dumps([{"issue": "needs [PROJECTION] with confidence"}]))
    for n in (0, 2, 5, 9):
        assert gate.truth.reward_fn({}, _pred(n)) == truth_reward({}, _pred(n))
        assert gate.extrapolation.reward_fn({}, _pred(n)) == extrap_reward({}, _pred(n))
    assert gate.extrapolation.reward_fn({}, marker) == extrap_reward({}, marker)
    # The two lenses use DIFFERENT rewards: bonus separates them
    assert gate.truth.reward_fn({}, marker) != gate.extrapolation.reward_fn({}, marker)


def test_threshold_one_means_all_n_run_below_five_violations():
    # Early-stop requires reward >= threshold (1.0). Under 5 violations the
    # reward is strictly below 1.0 -> BestOfN runs all N candidates.
    for count in range(5):
        assert truth_reward({}, _pred(count)) < 1.0
    assert truth_reward({}, _pred(5)) >= 1.0  # only a 5+ catch stops early