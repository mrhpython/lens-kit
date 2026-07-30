"""Compile-harness units: training loader, split, F2 metric. No network."""
import json

import dspy
import pytest

from lens_kit.compile_harness import (lens_gate_metric, load_training_examples,
                                      split_examples)


def _example(per_lens: dict) -> dspy.Example:
    return dspy.Example(
        text="text long enough for the loader to accept it fine",
        domain="general",
        expected_per_lens=json.dumps(per_lens),
    ).with_inputs("text", "domain")


# ── Loader ───────────────────────────────────────────────────────

def test_loader_skips_short_text_and_empty_per_lens(tmp_path):
    rows = [
        {"text": "short", "per_lens": {"truth": True}},                  # <20 chars
        {"text": "long enough but carries no labels at all", "per_lens": {}},
        {"text": "long enough and properly labeled, keep this one",
         "per_lens": {"truth": True}, "domain": "agency"},
    ]
    f = tmp_path / "train.json"
    f.write_text(json.dumps(rows))
    examples = load_training_examples(f)
    assert len(examples) == 1
    assert examples[0].domain == "agency"
    assert json.loads(examples[0].expected_per_lens) == {"truth": True}


def test_loader_shuffle_is_deterministic(tmp_path):
    rows = [{"text": f"deterministic shuffle example number {i:02d} here",
             "per_lens": {"truth": True}} for i in range(20)]
    f = tmp_path / "train.json"
    f.write_text(json.dumps(rows))
    first = [e.text for e in load_training_examples(f)]
    second = [e.text for e in load_training_examples(f)]
    assert first == second
    assert first != [r["text"] for r in rows]   # actually shuffled


def test_split_keeps_at_least_three_train():
    examples = list(range(5))
    train, val = split_examples(examples, val_split=0.9)
    assert len(train) == 3


# ── F2 metric (positive class = FAIL) ────────────────────────────

def test_metric_perfect_match_scores_one():
    ex = _example({"truth": False, "causality": True})
    pred = dspy.Prediction(per_lens={"truth": False, "causality": True})
    assert lens_gate_metric(ex, pred) == 1.0


def test_metric_fn_scores_worse_than_fp():
    """Missing a violation (FN) must hurt more than a false alarm (FP)."""
    ex = _example({"truth": False, "causality": False, "rights": True})
    fn_pred = dspy.Prediction(per_lens={"truth": True, "causality": False,
                                        "rights": True})   # missed truth
    fp_pred = dspy.Prediction(per_lens={"truth": False, "causality": False,
                                        "rights": False})  # flagged clean rights
    assert lens_gate_metric(ex, fn_pred) < lens_gate_metric(ex, fp_pred)


def test_metric_clean_example_rewards_tn_rate():
    ex = _example({"truth": True, "causality": True})
    clean_pred = dspy.Prediction(per_lens={"truth": True, "causality": True})
    noisy_pred = dspy.Prediction(per_lens={"truth": False, "causality": True})
    assert lens_gate_metric(ex, clean_pred) == 1.0   # not 0.0 (the F2-only trap)
    assert lens_gate_metric(ex, noisy_pred) == 0.5


def test_metric_missing_per_lens_scores_zero():
    ex = _example({"truth": True})
    assert lens_gate_metric(ex, dspy.Prediction(passed=True)) == 0.0


def test_metric_returns_feedback_for_gepa():
    ex = _example({"truth": False})
    pred = dspy.Prediction(per_lens={"truth": True})
    result = lens_gate_metric(ex, pred, None, pred_name="truth", pred_trace=None)
    assert hasattr(result, "feedback")
    assert result["score"] == 0.0
    assert "truth" in result["feedback"]
    assert "expected FAIL, got PASS" in result["feedback"]


def test_metric_clean_example_feedback_is_truthful():
    """Correct clean example scores 1.0 via TN-rate — feedback must not
    claim 'Perfect F2=0.00' (incoherent signal for the reflection LM)."""
    ex = _example({"truth": True, "causality": True})
    pred = dspy.Prediction(per_lens={"truth": True, "causality": True})
    result = lens_gate_metric(ex, pred, None, pred_name="truth", pred_trace=None)
    assert result["score"] == 1.0
    assert "clean content passed all lenses" in result["feedback"]
    assert "F2=0.00" not in result["feedback"]


def test_metric_caught_violation_feedback_is_truthful():
    ex = _example({"truth": False})
    pred = dspy.Prediction(per_lens={"truth": False})
    result = lens_gate_metric(ex, pred, None, pred_name="truth", pred_trace=None)
    assert result["score"] == 1.0
    assert "all planted violations caught" in result["feedback"]


def test_loader_rejects_unknown_lens_labels(tmp_path):
    """Label typos are a load-time hard error, never a silent 0 score."""
    rows = [{"text": "long enough text whose label has a typo in it",
             "per_lens": {"trooth": False}}]
    f = tmp_path / "train.json"
    f.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="trooth"):
        load_training_examples(f)


# ── item 7: severity-weighted metric option ──────────────────────

from lens_kit.compile_harness import validate_weight_map, _weighted_metric  # noqa: E402


# A fixed fixture set of (expected, actual_per_lens) pairs. The DEFAULT-path
# scores below were CAPTURED from the pre-change metric (no weight map) — this
# is the discriminating regression: weighting must leave the default untouched.
_REGRESSION_FIXTURES = [
    ({"truth": False, "causality": True}, {"truth": False, "causality": True}),
    ({"truth": False, "causality": False, "rights": True},
     {"truth": True, "causality": False, "rights": True}),
    ({"truth": False, "causality": False, "rights": True},
     {"truth": False, "causality": False, "rights": False}),
    ({"truth": True, "causality": True}, {"truth": True, "causality": True}),
    ({"truth": True, "causality": True}, {"truth": False, "causality": True}),
    ({"truth": False, "contradiction": False, "structure": True, "consistency": True},
     {"truth": False, "contradiction": True, "structure": False, "consistency": True}),
    ({"rights": False, "truth": False, "causality": False, "contradiction": False,
      "extrapolation": False, "structure": False, "consistency": True},
     {"rights": True, "truth": False, "causality": True, "contradiction": False,
      "extrapolation": False, "structure": True, "consistency": True}),
    ({"truth": False, "structure": True}, {"structure": True}),
    ({"truth": True}, {"truth": True}),
    ({"truth": False}, {"truth": False}),
]
_BASELINE_SCORES = [1.0, 0.555555555556, 0.909090909091, 1.0, 0.5, 0.5,
                    0.555555555556, 1.0, 1.0, 1.0]


def test_metric_default_is_byte_identical_to_baseline():
    """Regression: with NO weight map the metric reproduces the captured
    pre-change scores exactly (discriminating — any drift fails)."""
    for (exp, act), expected_score in zip(_REGRESSION_FIXTURES, _BASELINE_SCORES):
        s = lens_gate_metric(_example(exp), dspy.Prediction(per_lens=act))
        assert round(s, 12) == expected_score


def test_metric_weights_none_equals_no_arg():
    """weights=None must be exactly the no-argument path."""
    for exp, act in _REGRESSION_FIXTURES:
        ex, pred = _example(exp), dspy.Prediction(per_lens=act)
        assert lens_gate_metric(ex, pred, weights=None) == lens_gate_metric(ex, pred)


def test_upweighting_truth_changes_score_in_expected_direction():
    """A fixture where TRUTH is caught (TP) but a low-severity lens FPs:
    upweighting truth should RAISE the score (truth's correct TP now counts
    more relative to the structure false alarm)."""
    # truth: planted + caught (TP). structure: clean but flagged (FP).
    exp = {"truth": False, "structure": True}
    act = {"truth": False, "structure": False}
    base = lens_gate_metric(_example(exp), dspy.Prediction(per_lens=act))
    up = lens_gate_metric(_example(exp), dspy.Prediction(per_lens=act),
                          weights={"truth": 5.0})
    assert up > base


def test_downweighting_offender_changes_score():
    """A fixture where structure is the lens that's wrong (FP) — downweighting
    structure should move the score toward the unweighted-by-other lenses."""
    exp = {"truth": False, "structure": True}
    act = {"truth": False, "structure": False}   # truth TP, structure FP
    base = lens_gate_metric(_example(exp), dspy.Prediction(per_lens=act))
    down = lens_gate_metric(_example(exp), dspy.Prediction(per_lens=act),
                            weights={"structure": 0.1})
    assert down > base                            # the FP weighs less -> higher


def test_weight_map_validates_canonical_keys():
    from lens_kit.gate import CANONICAL_LENSES
    assert validate_weight_map(None) is None
    ok = validate_weight_map({"truth": 2.0, "structure": 0.5})
    assert ok == {"truth": 2.0, "structure": 0.5}
    # every canonical key is accepted
    assert validate_weight_map({l: 1.0 for l in CANONICAL_LENSES}) is not None


def test_weight_map_unknown_lens_is_error():
    with pytest.raises(ValueError, match="notalens"):
        validate_weight_map({"notalens": 2.0})


def test_weight_map_negative_weight_is_error():
    with pytest.raises(ValueError, match="negative"):
        validate_weight_map({"truth": -1.0})


def test_weight_map_non_numeric_is_error():
    with pytest.raises(ValueError, match="not a number"):
        validate_weight_map({"truth": "heavy"})


def test_weighted_metric_none_returns_bare_function():
    assert _weighted_metric(None) is lens_gate_metric


def test_weighted_metric_preserves_gepa_feedback_shape():
    """The bound metric still returns ScoreWithFeedback when pred_name given."""
    metric = _weighted_metric({"truth": 2.0})
    ex = _example({"truth": False})
    pred = dspy.Prediction(per_lens={"truth": True})
    result = metric(ex, pred, None, "truth", None)
    assert hasattr(result, "feedback")
    assert "expected FAIL, got PASS" in result["feedback"]


def test_weighted_metric_eager_validation():
    with pytest.raises(ValueError, match="notalens"):
        _weighted_metric({"notalens": 1.0})
