"""Shared no-network test fixtures.

No test in this suite may call a live API. The gate's LLM predictors are
replaced with StubPredictor instances returning canned dspy.Prediction
objects; dspy.settings is never configured with a real LM.

Ambient-credential immunity: litellm/python-dotenv can autoload a .env
from an ancestor directory at import time and silently repopulate API-key
env vars — which would make fail-closed tests pass for the wrong reason
(key present, no error path exercised) or flip them to network-capable.
The autouse fixture below strips every *_API_KEY-style variable AND moves
the cwd to a neutral tmp dir for every test, regardless of how the keys
got into the environment.
"""
import os

import dspy
import pytest

from lens_kit import LensGate, Profile, builtin_profile_path

# Provider credential vars that don't follow the *_API_KEY pattern.
_EXTRA_CREDENTIAL_VARS = {
    "OPENAI_ORGANIZATION", "ANTHROPIC_AUTH_TOKEN", "AZURE_API_BASE",
    "GOOGLE_APPLICATION_CREDENTIALS", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
}


@pytest.fixture(autouse=True)
def no_ambient_credentials(monkeypatch, tmp_path):
    """Strip provider keys + run from a neutral cwd. See module docstring."""
    for name in list(os.environ):
        if name.endswith("_API_KEY") or name in _EXTRA_CREDENTIAL_VARS:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no ancestor .env reachable from cwd


class StubPredictor:
    """Stands in for dspy.Predict/BestOfN — returns fixed outputs, counts calls."""

    def __init__(self, **outputs):
        self.outputs = outputs
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return dspy.Prediction(**self.outputs)


CLEAN = {
    "rights": dict(halt="false", has_violations="false", violations="[]", scrubbed_text=""),
    "truth": dict(has_violations="false", violations="[]", fixed_text=""),
    "causality": dict(has_violations="false", violations="[]"),
    "definitional_integrity": dict(has_violations="false", violations="[]"),
    "contradiction": dict(has_violations="false", violations="[]"),
    "extrapolation": dict(has_violations="false", violations="[]", fixed_text=""),
    "structure": dict(has_violations="false", violations="[]"),
    "judgment": dict(flags="[]"),
    "consistency": dict(has_violations="false", violations="[]"),
    "relevance": dict(has_violations="false", violations="[]"),
    "cross_check": dict(missed_violations="[]"),
}


def make_stub_gate(profile=None, overrides=None, **gate_kwargs):
    """LensGate with every LLM call stubbed out. overrides: lens -> output dict."""
    gate_kwargs.setdefault("fast_mode", True)
    gate_kwargs.setdefault("auto_fix", False)
    gate = LensGate(profile=profile, **gate_kwargs)
    outputs = {k: dict(v) for k, v in CLEAN.items()}
    for lens, out in (overrides or {}).items():
        outputs[lens].update(out)
    for lens, out in outputs.items():
        setattr(gate, lens, StubPredictor(**out))
    # Claim extractor pre-layer: no LLM call
    gate.claim_extractor.extract = lambda text: []
    return gate


@pytest.fixture
def stub_gate_factory():
    return make_stub_gate


@pytest.fixture
def agency_profile():
    return Profile.load(builtin_profile_path("agency-example"))
