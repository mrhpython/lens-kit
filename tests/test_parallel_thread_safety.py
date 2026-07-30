"""Parallel forward must not lose serve's scoped LM inside worker threads.

dspy's LM is thread-local; a ThreadPoolExecutor worker starts from the
GLOBAL config, not the parent's dspy.context override. Without the capture+
re-enter fix, a worker sees no LM and the predictor raises -> forward fails.
This test is discriminating: it fails on the naked dispatch, passes once the
captured context is re-entered per worker.
"""
import dspy
from dspy.utils.dummies import DummyLM

from lens_kit import LensGate, Profile


def test_parallel_forward_uses_context_lm_in_workers():
    # No GLOBAL lm configured on purpose: only a scoped context override exists,
    # exactly like serve's lm_context. Workers must re-enter it or they get None.
    canned = [{"halt": "false", "has_violations": "false", "violations": "[]",
               "scrubbed_text": "", "fixed_text": "", "flags": "[]",
               "missed_violations": "[]", "claims": "[]"}] * 200
    gate = LensGate(profile=Profile.empty(), fast_mode=True, auto_fix=False, parallel=True)
    with dspy.context(lm=DummyLM(canned)):
        result = gate(text="A short neutral sentence.")
    assert result.passed is True
    assert result.halted is False
