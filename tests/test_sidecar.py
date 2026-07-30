"""Provenance sidecar: field completeness, sha correctness, eval/dataset blocks,
not_a_claim_of discipline, and stub backward-compat. All no-network.
"""
import json

import pytest

from lens_kit.checkpoint import sha256_file
from lens_kit.config import Profile
from lens_kit.sidecar import (DEFAULT_NOT_A_CLAIM_OF, SINGLE_RUN_CAVEAT,
                              build_sidecar, generate_sidecar)


def _profile(model="openai/gpt-4o"):
    return Profile.from_dict({"name": "demo", "llm": {"model": model}})


def _ckpt(tmp_path, body='{"metadata": {"k": 1}}'):
    p = tmp_path / "gate.json"
    p.write_text(body)
    return p


# ── Field completeness + sha correctness ─────────────────────────

def test_sidecar_has_all_required_fields(tmp_path):
    sc = build_sidecar(_ckpt(tmp_path), profile=_profile(),
                       train_size=66, val_size=17)
    for field in ("provenance_stub", "sidecar_schema_version", "checkpoint",
                  "sha256", "model", "profile_name", "dspy_version",
                  "gepa_version", "train_size", "val_size", "date",
                  "provider_note", "size_bytes", "dependency_versions",
                  "dataset", "eval", "not_a_claim_of", "re_measurements"):
        assert field in sc, f"missing sidecar field: {field}"


def test_sha256_matches_checkpoint_bytes(tmp_path):
    ckpt = _ckpt(tmp_path)
    sc = build_sidecar(ckpt, profile=_profile())
    assert sc["sha256"] == sha256_file(ckpt)
    assert sc["size_bytes"] == ckpt.stat().st_size


def test_missing_checkpoint_is_a_hard_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_sidecar(tmp_path / "absent.json", profile=_profile())


def test_provider_note_names_provider(tmp_path):
    sc_xai = build_sidecar(_ckpt(tmp_path), profile=_profile("xai/grok-4-1-fast"))
    assert "xAI" in sc_xai["provider_note"]
    assert "xai/grok-4-1-fast" in sc_xai["provider_note"]


# ── not_a_claim_of discipline ────────────────────────────────────

def test_defaults_always_present(tmp_path):
    sc = build_sidecar(_ckpt(tmp_path), profile=_profile())
    for default in DEFAULT_NOT_A_CLAIM_OF:
        assert default in sc["not_a_claim_of"]


def test_extra_not_a_claim_of_appends_never_replaces(tmp_path):
    sc = build_sidecar(_ckpt(tmp_path), profile=_profile(),
                       extra_not_a_claim_of=["our specific trading edge"])
    assert "our specific trading edge" in sc["not_a_claim_of"]
    for default in DEFAULT_NOT_A_CLAIM_OF:                    # defaults survive
        assert default in sc["not_a_claim_of"]


def test_single_run_caveat_added_without_envelope(tmp_path):
    eval_doc = tmp_path / "eval.json"
    eval_doc.write_text(json.dumps({
        "summary": {"overall_accuracy": 0.88, "catch_rate": 0.72, "fp_rate": 0.09},
        "reruns": 1, "per_example": [],
    }))
    sc = build_sidecar(_ckpt(tmp_path), profile=_profile(),
                       eval_results_path=eval_doc)
    assert SINGLE_RUN_CAVEAT in sc["not_a_claim_of"]
    assert sc["eval"]["has_envelope"] is False
    assert sc["eval"]["metrics"]["overall_accuracy"] == 0.88


def test_envelope_eval_omits_single_run_caveat(tmp_path):
    env_doc = tmp_path / "envelope.json"
    env_doc.write_text(json.dumps({
        "reruns": 3,
        "metrics": {
            "overall_accuracy": {"worst_of_n": 0.85, "mean": 0.87,
                                 "direction": "higher_is_better"},
            "fp_rate": {"worst_of_n": 0.12, "mean": 0.10,
                        "direction": "lower_is_better"},
        },
    }))
    sc = build_sidecar(_ckpt(tmp_path), profile=_profile(),
                       eval_results_path=env_doc)
    assert SINGLE_RUN_CAVEAT not in sc["not_a_claim_of"]
    assert sc["eval"]["has_envelope"] is True
    assert sc["eval"]["metrics"]["overall_accuracy"] == 0.85   # worst-of-N floor
    assert "envelope" in sc["eval"]


# ── Dataset block ────────────────────────────────────────────────

def test_dataset_block_records_counts_and_labels_version(tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps({
        "_metadata": {"labels_version": "v3", "created": "2026-06-01"},
        "examples": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
    }))
    sc = build_sidecar(_ckpt(tmp_path), profile=_profile(), dataset_path=data)
    assert sc["dataset"]["example_count"] == 3
    assert sc["dataset"]["labels_version"] == "v3"
    assert sc["dataset"]["dataset_sha256"] == sha256_file(data)


# ── Stub backward-compat ─────────────────────────────────────────

def test_backward_compat_with_c2_stub_fields(tmp_path):
    """Every field the C2 stub guaranteed is still present, same meaning."""
    sc = build_sidecar(_ckpt(tmp_path), profile=_profile(),
                       train_size=10, val_size=2)
    # provenance_stub flips to False (the one intended change).
    assert sc["provenance_stub"] is False
    # The stub's identity + lineage fields are intact.
    assert sc["checkpoint"] == "gate.json"
    assert sc["model"] == "openai/gpt-4o"
    assert sc["profile_name"] == "demo"
    assert sc["train_size"] == 10 and sc["val_size"] == 2
    assert sc["dspy_version"] and sc["gepa_version"]
    assert sc["date"]


def test_extra_block_merges_like_the_stub(tmp_path):
    sc = build_sidecar(_ckpt(tmp_path), profile=_profile(),
                       extra={"budget": "auto=light", "val_score": 0.5})
    assert sc["budget"] == "auto=light" and sc["val_score"] == 0.5


def test_generate_sidecar_writes_file(tmp_path):
    ckpt = _ckpt(tmp_path)
    sc, path = generate_sidecar(ckpt, profile=_profile())
    assert path == ckpt.with_suffix(".provenance.json")
    on_disk = json.loads(path.read_text())
    assert on_disk["sha256"] == sc["sha256"]
    assert on_disk["re_measurements"] == []
