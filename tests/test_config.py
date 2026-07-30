"""Config loading, round-trip, dspy.LM mapping, LM scoping — no network."""
import pytest

import dspy
from lens_kit import (
    ConfigError, ConsistencyConfig, LLMConfig, Profile, builtin_profile_path,
    configure_from_profile, lm_context, make_lm,
)


def _profile(model: str) -> Profile:
    p = Profile.empty()
    p.llm = LLMConfig(model=model)  # no api_key_env -> no key needed, no network
    return p


def test_agency_example_loads():
    profile = Profile.load(builtin_profile_path("agency-example"))
    assert profile.name == "agency-example"
    # Asserted structurally, not pinned to a vendor: the shipped example profile
    # deliberately carries a placeholder endpoint so a fresh install cannot
    # silently inherit someone else's provider choice. Both fields must still be
    # present and well-formed ("<provider>/<model-id>", and an env-var NAME).
    assert "/" in profile.llm.model and profile.llm.model.strip()
    assert profile.llm.api_key_env and profile.llm.api_key_env.isupper()
    assert profile.llm.temperature == 0.1
    assert profile.llm.max_tokens == 8000
    # Domain vocabulary present
    assert "best practice" in profile.vocabulary.truth_whitelist
    assert "bull case" in profile.vocabulary.scenario_labels
    assert "from hledger" in profile.vocabulary.internal_source_phrases
    assert set(profile.domain_rules) == {"finance", "marketing", "seo", "competitor"}
    assert list(profile.domain_detection)[0] == "finance"  # priority order preserved


def test_empty_profile_defaults():
    profile = Profile.empty()
    assert profile.vocabulary.truth_whitelist == []
    assert profile.vocabulary.scenario_labels == []
    assert profile.domain_rules == {}
    assert profile.domain_detection == {}


def test_round_trip(tmp_path):
    original = Profile.load(builtin_profile_path("agency-example"))
    out = tmp_path / "copy.yaml"
    original.save(out)
    reloaded = Profile.load(out)
    assert reloaded.to_dict() == original.to_dict()


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nbogus_section: {}\n")
    with pytest.raises(ConfigError):
        Profile.load(p)
    p2 = tmp_path / "bad2.yaml"
    p2.write_text("name: x\nvocabulary:\n  not_a_list_we_know: []\n")
    with pytest.raises(ConfigError):
        Profile.load(p2)


def test_missing_profile_rejected(tmp_path):
    with pytest.raises(ConfigError):
        Profile.load(tmp_path / "nope.yaml")


def test_make_lm_maps_to_dspy(monkeypatch):
    monkeypatch.setenv("LENS_KIT_TEST_KEY", "sk-test-not-real")
    cfg = LLMConfig(model="openai/gpt-4o", api_key_env="LENS_KIT_TEST_KEY",
                    temperature=0.3, max_tokens=4000)
    lm = make_lm(cfg)
    assert isinstance(lm, dspy.LM)
    assert lm.model == "openai/gpt-4o"
    assert lm.kwargs["temperature"] == 0.3
    assert lm.kwargs["max_tokens"] == 4000
    assert lm.kwargs["api_key"] == "sk-test-not-real"


def test_make_lm_local_endpoint_no_key():
    cfg = LLMConfig(model="ollama_chat/llama3", api_base="http://localhost:11434")
    lm = make_lm(cfg)
    assert lm.model == "ollama_chat/llama3"
    assert lm.kwargs["api_base"] == "http://localhost:11434"
    assert "api_key" not in lm.kwargs


def test_make_lm_fails_closed_on_missing_key(monkeypatch):
    monkeypatch.delenv("LENS_KIT_ABSENT_KEY", raising=False)
    cfg = LLMConfig(model="openai/gpt-4o", api_key_env="LENS_KIT_ABSENT_KEY")
    with pytest.raises(ConfigError):
        make_lm(cfg)


def test_make_lm_requires_model():
    with pytest.raises(ConfigError):
        make_lm(LLMConfig())


def test_configure_from_profile_returns_previous_lm():
    original = getattr(dspy.settings, "lm", None)
    try:
        lm_a, prev_a = configure_from_profile(_profile("openai/gpt-4o"))
        assert prev_a is original
        assert dspy.settings.lm is lm_a
        lm_b, prev_b = configure_from_profile(_profile("ollama_chat/llama3"))
        assert prev_b is lm_a  # restore path: dspy.configure(lm=prev_b)
        assert dspy.settings.lm is lm_b
        dspy.configure(lm=prev_b)
        assert dspy.settings.lm is lm_a
    finally:
        dspy.configure(lm=original)  # never leak global state out of the test


def test_lm_context_restores_on_exit_and_exception():
    original = getattr(dspy.settings, "lm", None)
    with lm_context(_profile("openai/gpt-4o")) as lm:
        assert dspy.settings.lm is lm
        # Nesting: inner override wins, then unwinds
        with lm_context(_profile("ollama_chat/llama3")) as inner:
            assert dspy.settings.lm is inner
        assert dspy.settings.lm is lm
    assert getattr(dspy.settings, "lm", None) is original

    with pytest.raises(RuntimeError):
        with lm_context(_profile("openai/gpt-4o")):
            raise RuntimeError("boom")
    assert getattr(dspy.settings, "lm", None) is original


def test_lm_context_fails_closed_before_entering(monkeypatch):
    monkeypatch.delenv("LENS_KIT_ABSENT_KEY", raising=False)
    p = Profile.empty()
    p.llm = LLMConfig(model="openai/gpt-4o", api_key_env="LENS_KIT_ABSENT_KEY")
    original = getattr(dspy.settings, "lm", None)
    with pytest.raises(ConfigError):
        with lm_context(p):
            pass  # pragma: no cover — must not be reached
    assert getattr(dspy.settings, "lm", None) is original


# ── ConsistencyConfig ────────────────────────────────────────────

def test_consistency_config_defaults():
    cc = ConsistencyConfig()
    assert cc.markers == ["[UNVERIFIED]", "[ESTIMATE]", "[PROJECTION]"]
    assert cc.deny == []


def test_consistency_config_from_dict_and_roundtrip():
    cc = ConsistencyConfig.from_dict({"markers": ["[GUESS]"],
                                      "deny": ["gross margin", "cogs"]})
    assert cc.markers == ["[GUESS]"]
    assert cc.deny == ["gross margin", "cogs"]
    assert ConsistencyConfig.from_dict(cc.to_dict()).to_dict() == cc.to_dict()


def test_consistency_config_rejects_unknown_key():
    with pytest.raises(ConfigError, match="consistency"):
        ConsistencyConfig.from_dict({"bogus": []})


def test_consistency_config_rejects_non_list():
    with pytest.raises(ConfigError, match="markers"):
        ConsistencyConfig.from_dict({"markers": "not-a-list"})
    with pytest.raises(ConfigError, match="deny"):
        ConsistencyConfig.from_dict({"deny": "not-a-list"})


def test_profile_carries_consistency_section_and_roundtrips():
    p = Profile.from_dict({
        "name": "t", "llm": {"model": "openai/gpt-4o"},
        "consistency": {"deny": ["internal cost"]},
    })
    assert p.consistency.deny == ["internal cost"]
    assert Profile.from_dict(p.to_dict()).consistency.deny == ["internal cost"]


def test_agency_example_profile_ships_deny_list():
    p = Profile.load(builtin_profile_path("agency-example"))
    assert "lens gate" in p.consistency.deny
    assert p.consistency.markers == ["[UNVERIFIED]", "[ESTIMATE]", "[PROJECTION]"]


# ── item 7: compile.metric_weights (optional, backward-compatible) ──

from lens_kit.config import CompileConfig  # noqa: E402


def test_compile_metric_weights_default_none_old_profiles_unchanged():
    # an old profile with no compile.metric_weights key -> None (unweighted)
    p = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"}})
    assert p.compile.metric_weights is None
    # round-trips as None
    assert Profile.from_dict(p.to_dict()).compile.metric_weights is None


def test_compile_metric_weights_set_and_roundtrip():
    cc = CompileConfig.from_dict({"metric_weights": {"truth": 2.0, "structure": 0.5}})
    assert cc.metric_weights == {"truth": 2.0, "structure": 0.5}
    assert CompileConfig.from_dict(cc.to_dict()).metric_weights == cc.metric_weights


def test_compile_metric_weights_rejects_non_mapping():
    with pytest.raises(ConfigError, match="metric_weights"):
        CompileConfig.from_dict({"metric_weights": ["truth", 2.0]})


def test_agency_example_profile_has_no_metric_weights():
    # the shipped profile stays on the unweighted default
    p = Profile.load(builtin_profile_path("agency-example"))
    assert p.compile.metric_weights is None
