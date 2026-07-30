"""Checkpoint save/load: sha256 + provenance siblings, key-remap round trips.

No network: dspy Predict construction and state (de)serialization never call
an LM. Stub-LM discipline: no dspy.settings LM is ever configured here.
"""
import json
from pathlib import Path

import pytest

from lens_kit import LensGate, Profile
from lens_kit.checkpoint import (load_checkpoint, save_checkpoint, sha256_file,
                                 _candidate_keys)

MARKER = "REMAP-ROUND-TRIP-MARKER instructions"


def _save_gate_state(tmp_path, name="ckpt.json") -> Path:
    """Save a bare LensGate checkpoint with marked truth instructions."""
    gate = LensGate()
    path = tmp_path / name
    gate.save(str(path))
    state = json.loads(path.read_text())
    state["truth.module"]["signature"]["instructions"] = MARKER
    path.write_text(json.dumps(state))
    return path


# ── Save: sha256 sibling + FULL provenance sidecar (C4) ──────────

def test_save_checkpoint_writes_sha_and_provenance(tmp_path):
    profile = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"}})
    gate = LensGate(profile=profile)
    ckpt = tmp_path / "out" / "gate.json"

    sidecar = save_checkpoint(gate, ckpt, profile=profile, train_size=66, val_size=17)

    assert ckpt.exists()
    sha_file = Path(str(ckpt) + ".sha256")
    assert sha_file.exists()
    digest, _, name = sha_file.read_text().strip().partition("  ")
    assert digest == sha256_file(ckpt) == sidecar["sha256"]
    assert name == "gate.json"

    prov = json.loads(ckpt.with_suffix(".provenance.json").read_text())
    # C4: full sidecar, provenance_stub now False (NOT True).
    assert prov["provenance_stub"] is False
    # Backward compat: every C2 stub field still present, same meaning.
    assert prov["model"] == "openai/gpt-4o"
    assert prov["checkpoint"] == "gate.json"
    assert prov["profile_name"] == "t"
    assert prov["train_size"] == 66 and prov["val_size"] == 17
    assert prov["dspy_version"] and prov["gepa_version"]
    assert prov["date"]
    # C4 additions.
    assert prov["sidecar_schema_version"] == 1
    assert "not_a_claim_of" in prov and prov["not_a_claim_of"]
    assert prov["re_measurements"] == []
    assert prov["provider_note"] and "openai/gpt-4o" in prov["provider_note"]


# ── Load: direct + remap round trips ─────────────────────────────

def test_round_trip_same_shape_loads_directly(tmp_path):
    path = _save_gate_state(tmp_path)
    gate2 = LensGate()
    report = load_checkpoint(gate2, path, log=lambda *_: None)
    assert report["remapped"] is False
    assert gate2.truth.module.signature.instructions == MARKER


def test_remap_bare_gate_checkpoint_into_wrapper(tmp_path):
    """Bare-gate keys (truth.module) load into LensGateWrapper (gate.truth.module)."""
    from lens_kit.compile_harness import LensGateWrapper

    path = _save_gate_state(tmp_path)
    wrapper = LensGateWrapper()
    report = load_checkpoint(wrapper, path, log=lambda *_: None)
    assert report["remapped"] is True
    assert "gate.truth.module" in report["loaded"]
    assert wrapper.gate.truth.module.signature.instructions == MARKER


def test_remap_legacy_wrapper_checkpoint_into_bare_gate(tmp_path):
    """Legacy optimizer save: gate.* prefix + pre-BestOfN 'gate.truth' key."""
    path = _save_gate_state(tmp_path)
    state = json.loads(path.read_text())
    legacy = {"gate." + (k[:-len(".module")] if k.endswith(".module") else k): v
              for k, v in state.items() if k != "metadata"}
    legacy["gate.obsolete_lens"] = {"junk": True}   # stale key must be skipped
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy))

    gate = LensGate()
    report = load_checkpoint(gate, legacy_path, log=lambda *_: None)
    assert report["remapped"] is True
    assert "truth.module" in report["loaded"]        # gate.truth -> truth.module
    assert "gate.obsolete_lens" in report["stale"]
    assert report["missing"] == []
    assert gate.truth.module.signature.instructions == MARKER


def test_candidate_keys_cover_both_remap_directions():
    assert "gate.truth" in _candidate_keys("truth.module")
    assert "truth" in _candidate_keys("truth.module")
    assert "truth.module" in _candidate_keys("gate.truth.module")
    assert "gate.truth.module" in _candidate_keys("truth.module")


def test_nothing_matches_is_a_hard_error(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"completely": {"unrelated": 1}}))
    with pytest.raises(ValueError, match="refusing to silently start fresh"):
        load_checkpoint(LensGate(), path, log=lambda *_: None)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(LensGate(), tmp_path / "absent.json")
