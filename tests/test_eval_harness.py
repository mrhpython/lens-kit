"""Eval harness: mismatch/confusion direction, envelope floors, versioning.

No network: gates are conftest stub gates; the CLI-level test monkeypatches
eval_harness.make_gate.
"""
import json
from pathlib import Path

import pytest

from lens_kit.config import Profile
from lens_kit.eval_harness import (CACHE_WARNING, build_envelope, load_holdout,
                                   next_free_envelope_path, next_free_run_path,
                                   run_assessment, run_eval_command, summarize)
from tests.conftest import make_stub_gate

VIOLATION = '[{"issue": "planted", "severity": "warning"}]'


# ── Mismatch strings + confusion direction ───────────────────────
# Positive class = FAIL (violation present).
#   expected PASS, got FAIL -> FP;  expected FAIL, got PASS -> FN.

def test_fp_direction_expected_pass_got_fail():
    gate = make_stub_gate(overrides={
        "truth": dict(has_violations="true", violations=VIOLATION)})
    examples = [{"text": "clean text that the gate wrongly flags here",
                 "per_lens": {"truth": True}}]
    results = run_assessment(examples, gate, progress=False)
    ex = results["per_example"][0]
    assert ex["mismatches"] == ["truth: FP (expected PASS, got FAIL)"]
    assert ex["lens_confusion"]["truth"] == "fp"
    assert results["per_lens"]["truth"] == {"tp": 0, "tn": 0, "fp": 1, "fn": 0}


def test_fn_direction_expected_fail_got_pass():
    gate = make_stub_gate()  # all-clean stub misses the planted violation
    examples = [{"text": "text with a planted causality violation in it",
                 "per_lens": {"causality": False}}]
    results = run_assessment(examples, gate, progress=False)
    ex = results["per_example"][0]
    assert ex["mismatches"] == ["causality: FN (expected FAIL, got PASS)"]
    assert ex["lens_confusion"]["causality"] == "fn"
    assert results["per_lens"]["causality"] == {"tp": 0, "tn": 0, "fp": 0, "fn": 1}


def test_tp_tn_and_summary_math():
    gate = make_stub_gate(overrides={
        "contradiction": dict(has_violations="true",
                              violations='[{"claim_a": "a", "claim_b": "b"}]')})
    examples = [{"text": "text where contradiction is correctly caught yes",
                 "per_lens": {"contradiction": False, "truth": True}}]
    results = run_assessment(examples, gate, progress=False)
    assert results["per_example"][0]["mismatches"] == []
    assert results["per_lens"]["contradiction"]["tp"] == 1
    assert results["per_lens"]["truth"]["tn"] == 1
    s = summarize(results)
    assert s == {"overall_accuracy": 1.0, "catch_rate": 1.0, "fp_rate": 0.0}


def test_unevaluated_lens_is_skipped_not_guessed():
    gate = make_stub_gate()
    examples = [{"text": "labels include a lens the gate never reports on",
                 "per_lens": {"truth": True, "relevance": True}}]
    results = run_assessment(examples, gate, progress=False)
    ex = results["per_example"][0]
    # relevance not run (no context) -> null confusion, no mismatch, no count
    assert ex["lens_confusion"]["relevance"] is None
    assert "relevance" not in results["per_lens"]


def test_gate_error_is_counted_not_crashed():
    gate = make_stub_gate()
    gate.claim_extractor.extract = lambda text: (_ for _ in ()).throw(RuntimeError("boom"))
    examples = [{"text": "this example explodes inside the gate pipeline",
                 "per_lens": {"truth": True}}]
    results = run_assessment(examples, gate, progress=False)
    assert results["errors"] == 1
    assert results["per_example"][0]["mismatches"][0].startswith("ERROR:")


# ── Envelope floors (direction-aware worst-of-N) ─────────────────

def test_envelope_floor_directions():
    summaries = [
        {"overall_accuracy": 0.80, "catch_rate": 0.70, "fp_rate": 0.20},
        {"overall_accuracy": 0.90, "catch_rate": 0.60, "fp_rate": 0.10},
        {"overall_accuracy": 0.85, "catch_rate": 0.65, "fp_rate": 0.15},
    ]
    env = build_envelope(summaries)
    assert env["overall_accuracy"]["worst_of_n"] == 0.80   # higher-is-better -> MIN
    assert env["catch_rate"]["worst_of_n"] == 0.60
    assert env["fp_rate"]["worst_of_n"] == 0.20            # lower-is-better -> MAX
    assert env["fp_rate"]["direction"] == "lower_is_better"
    assert env["overall_accuracy"]["min"] == 0.80
    assert env["overall_accuracy"]["max"] == 0.90
    assert abs(env["overall_accuracy"]["mean"] - 0.85) < 1e-12
    import statistics
    assert env["overall_accuracy"]["stddev"] == statistics.pstdev([0.8, 0.9, 0.85])


# ── Versioned filenames never overwrite ──────────────────────────

def test_next_free_run_path_skips_existing(tmp_path):
    assert next_free_run_path(tmp_path, "2026-06-12").name == "eval-2026-06-12-run1.json"
    (tmp_path / "eval-2026-06-12-run1.json").write_text("{}")
    (tmp_path / "eval-2026-06-12-run2.json").write_text("{}")
    assert next_free_run_path(tmp_path, "2026-06-12").name == "eval-2026-06-12-run3.json"


def test_next_free_envelope_path_skips_existing(tmp_path):
    assert next_free_envelope_path(tmp_path, "2026-06-12").name == "eval-2026-06-12-envelope.json"
    (tmp_path / "eval-2026-06-12-envelope.json").write_text("{}")
    assert (next_free_envelope_path(tmp_path, "2026-06-12").name
            == "eval-2026-06-12-envelope-2.json")


# ── Holdout loading: both shapes ─────────────────────────────────

def test_load_holdout_accepts_list_and_wrapper(tmp_path):
    rows = [{"text": "x" * 30, "per_lens": {"truth": True}}]
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(rows))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"_metadata": {"created": "2026"}, "examples": rows}))
    ex1, meta1 = load_holdout(plain)
    ex2, meta2 = load_holdout(wrapped)
    assert ex1 == ex2 == rows
    assert meta1 == {}
    assert meta2 == {"created": "2026"}


def test_load_holdout_rejects_unknown_lens_labels(tmp_path):
    """Label typos never intersect gate output -> would silently score 0.
    Choice: hard error at load (fail closed), not a warning."""
    f = tmp_path / "holdout.json"
    f.write_text(json.dumps([
        {"text": "fine example with a canonical label present here",
         "per_lens": {"truth": True}},
        {"text": "example whose gold label carries a fatal typo oh",
         "per_lens": {"trooth": False}},
    ]))
    with pytest.raises(ValueError, match="trooth") as exc:
        load_holdout(f)
    assert "example 1" in str(exc.value)        # points at the offending row


# ── Definitional Integrity: canon change + loadability (Items 3-5) ──

def test_load_holdout_accepts_definitional_integrity_label(tmp_path):
    """The 10th canonical lens key is accepted; a near-miss typo is rejected."""
    f = tmp_path / "ok.json"
    f.write_text(json.dumps([
        {"text": "an arg resting on an undefined load-bearing term here",
         "per_lens": {"definitionalIntegrity": False}},
    ]))
    ex, _ = load_holdout(f)
    assert ex[0]["per_lens"]["definitionalIntegrity"] is False

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"text": "this label has a definitional-integrity casing/typo bug",
         "per_lens": {"definitional_integrity": False}},  # snake_case is NOT canonical
    ]))
    with pytest.raises(ValueError, match="definitional_integrity"):
        load_holdout(bad)


def test_load_holdout_tolerates_old_format_without_new_key(tmp_path):
    """Item 4 finding: load_holdout/validate_lens_labels only reject UNKNOWN
    keys — a MISSING canonical key is tolerated (treated as unlabeled for that
    lens). So pre-canon-change datasets (no definitionalIntegrity key) still
    load unchanged; we do NOT mechanically backfill old data."""
    f = tmp_path / "old.json"
    f.write_text(json.dumps([
        {"text": "an old example authored before the 10th lens existed yes",
         "per_lens": {"truth": True, "causality": True}},  # no definitionalIntegrity
    ]))
    ex, _ = load_holdout(f)
    assert "definitionalIntegrity" not in ex[0]["per_lens"]  # still absent, still loads


def test_definitional_holdout_file_loads_clean():
    """The shipped definitional holdout loads through load_holdout with all
    labels canonical, every example carrying the new key, and the documented
    positive/control counts."""
    holdout = (Path(__file__).resolve().parents[2]
               / "probes" / "definitional-integrity-2026-06-14"
               / "definitional-holdout.json")
    if not holdout.exists():
        pytest.skip(f"definitional holdout not present at {holdout}")
    examples, meta = load_holdout(holdout)  # raises if any label non-canonical
    assert all("definitionalIntegrity" in e["per_lens"] for e in examples)
    pos = [e for e in examples if e["per_lens"]["definitionalIntegrity"] is False]
    ctl = [e for e in examples if e["per_lens"]["definitionalIntegrity"] is True]
    assert len(pos) >= 12 and len(ctl) >= 10        # ~18 positives / ~12 controls target
    # The 12 pre-registered probe args are present, verbatim ids.
    probe_ids = {e["source_id"] for e in examples if e.get("source") == "probe"}
    assert {"P1", "P2", "P3", "P4", "P5", "P6"} <= probe_ids
    assert {"C1", "C2", "C3", "C4", "C5", "C6"} <= probe_ids
    # The P4 boundary decision: pure circular causation is NOT a definitional
    # defect on this lens.
    p4 = next(e for e in examples if e.get("source_id") == "P4")
    assert p4["per_lens"]["definitionalIntegrity"] is True


def test_existing_warm_cache_datasets_still_load():
    """Item 4: datasets authored before the canon change (no
    definitionalIntegrity key) must still load after the 10th lens was added to
    CANONICAL_LENSES.

    Reads an optional sibling dataset directory if one is present; skips
    cleanly when it is not, which is the normal case for a fresh checkout.
    Set LENS_KIT_LEGACY_DATASETS to point at your own pre-canon datasets to
    exercise this against real data.
    """
    import os
    base = Path(os.environ.get(
        "LENS_KIT_LEGACY_DATASETS",
        Path(__file__).resolve().parents[2] / "legacy-datasets"))
    loaded_any = False
    for name in ("dataset-train.json", "dataset-holdout.json"):
        fp = base / name
        if not fp.exists():
            continue
        examples, _ = load_holdout(fp)  # raises if it broke
        assert examples  # non-empty
        loaded_any = True
    if not loaded_any:
        pytest.skip("warm-cache datasets not present")


# ── Full command: reruns, envelope file, cache warning, ledger ───

def test_run_eval_command_reruns_envelope_and_cache_warning(
        tmp_path, monkeypatch, capsys):
    import lens_kit.eval_harness as eh

    monkeypatch.setattr(eh, "make_gate", lambda profile, ckpt: make_stub_gate())
    profile = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"}})
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps([
        {"text": "first holdout example with plenty of characters",
         "per_lens": {"truth": True}},
        {"text": "second holdout example with a planted defect oh no",
         "per_lens": {"causality": False}},
    ]))
    out_dir = tmp_path / "results"

    rc = run_eval_command(holdout, profile, reruns=2, output_dir=out_dir,
                          command_str="lens-kit eval test")
    assert rc == 0
    out = capsys.readouterr().out
    assert CACHE_WARNING.splitlines()[0].removeprefix("WARNING: ") in out

    run_files = sorted(out_dir.glob("eval-*-run*.json"))
    env_files = sorted(out_dir.glob("eval-*-envelope*.json"))
    assert len(run_files) == 2 and len(env_files) == 1

    doc = json.loads(run_files[0].read_text())
    assert doc["eval_schema_version"] == 2
    assert doc["checkpoint"] == "baseline (uncompiled)"
    assert doc["data_sha256"]
    assert doc["summary"]["catch_rate"] == 0.0          # stub missed the defect
    assert doc["per_example"][1]["mismatches"] == [
        "causality: FN (expected FAIL, got PASS)"]

    env = json.loads(env_files[0].read_text())
    assert env["metrics"]["fp_rate"]["worst_of_n"] == 0.0
    assert env["cache_caveat"]                          # shared-cache caveat recorded

    ledger = Path.cwd() / "RUNS.md"
    assert ledger.exists()
    assert "EVAL" in ledger.read_text()


def test_fresh_cache_uses_new_dir_per_rerun(tmp_path, monkeypatch):
    import lens_kit.eval_harness as eh

    seen = []

    def fake_configure_cache(**kwargs):
        seen.append(kwargs)

    import dspy
    monkeypatch.setattr(dspy, "configure_cache", fake_configure_cache)
    monkeypatch.setattr(eh, "make_gate", lambda profile, ckpt: make_stub_gate())
    profile = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"}})
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps([
        {"text": "an example long enough to be a real holdout row",
         "per_lens": {"truth": True}}]))

    rc = run_eval_command(holdout, profile, reruns=2, fresh_cache=True,
                          output_dir=tmp_path / "r")
    assert rc == 0
    assert len(seen) == 2
    dirs = [k["disk_cache_dir"] for k in seen]
    assert dirs[0] != dirs[1]                       # fresh dir per rerun
    assert all(k["enable_memory_cache"] is False for k in seen)


def test_fresh_cache_restores_global_dspy_cache(tmp_path, monkeypatch):
    """run_eval_command must not clobber the embedding app's cache config."""
    import dspy
    import lens_kit.eval_harness as eh

    monkeypatch.setattr(eh, "make_gate", lambda profile, ckpt: make_stub_gate())
    profile = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"}})
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps([
        {"text": "an example long enough to be a real holdout row",
         "per_lens": {"truth": True}}]))

    prior_cache = dspy.cache
    rc = run_eval_command(holdout, profile, reruns=2, fresh_cache=True,
                          output_dir=tmp_path / "r")     # real configure_cache
    assert rc == 0
    assert dspy.cache is prior_cache                     # restored, not replaced
