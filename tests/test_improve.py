"""lens-kit improve — challenger/promote loop. All no-network (stubbed seams).

A real improve run makes LLM calls (compile + eval); every test here replaces
the three module-level seams (``_compile_challenger``, ``_eval_checkpoint``,
``_mutation_control``) with deterministic fakes, so the suite never touches the
network. The real seams (which call compile_harness / eval_harness / mutation)
are reached only on a live run with real flags — out of scope for tests.
"""
import json

import pytest

import lens_kit.improve as imp
from lens_kit.improve import (EXIT_OK, EXIT_USAGE, MeasuredEval, beats_baseline,
                              run_improve)


# ── Promotion rule (pure) ────────────────────────────────────────

def _eval(accuracy, catch, fp, *, envelope=None):
    """A MeasuredEval with the headline summary and an optional envelope."""
    return MeasuredEval(
        summary={"overall_accuracy": accuracy, "catch_rate": catch, "fp_rate": fp},
        envelope=envelope, reruns=1 if envelope is None else 3)


def _env(metric, *, worst):
    """Minimal envelope block for one metric (only worst_of_n is load-bearing)."""
    return {metric: {"worst_of_n": worst}}


def test_beats_baseline_clear_win_beyond_envelope():
    # Challenger worst-case catch (0.80) strictly above baseline catch (0.70).
    base = _eval(0.90, 0.70, 0.05)
    chal = _eval(0.95, 0.85, 0.04, envelope=_env("catch_rate", worst=0.80))
    verdict = beats_baseline(base, chal)
    assert verdict.promote is True
    assert "0.80" in verdict.reason or "0.8" in verdict.reason


def test_beats_baseline_loss_keeps_baseline():
    base = _eval(0.90, 0.80, 0.05)
    chal = _eval(0.85, 0.70, 0.06, envelope=_env("catch_rate", worst=0.68))
    verdict = beats_baseline(base, chal)
    assert verdict.promote is False


def test_beats_baseline_within_envelope_is_not_a_win():
    # Challenger MEAN catch is nominally higher (0.82 > 0.80) but its worst-of-N
    # floor (0.79) does NOT clear the baseline's headline (0.80) -> KEEP.
    base = _eval(0.90, 0.80, 0.05)
    chal = _eval(0.91, 0.82, 0.05, envelope=_env("catch_rate", worst=0.79))
    verdict = beats_baseline(base, chal)
    assert verdict.promote is False
    assert "envelope" in verdict.reason.lower() or "worst" in verdict.reason.lower()


def test_beats_baseline_exact_tie_keeps_baseline():
    base = _eval(0.90, 0.80, 0.05)
    chal = _eval(0.90, 0.80, 0.05, envelope=_env("catch_rate", worst=0.80))
    verdict = beats_baseline(base, chal)
    assert verdict.promote is False  # a tie is not a win


def test_beats_baseline_single_run_caveat_no_envelope():
    # No envelope on the challenger: comparison is on a single measurement.
    base = _eval(0.90, 0.70, 0.05)
    chal = _eval(0.95, 0.85, 0.04)  # no envelope
    verdict = beats_baseline(base, chal)
    assert verdict.promote is True
    assert verdict.single_measurement is True
    assert "single" in verdict.reason.lower()


# ── FIX 1: the FP-explosion bound (flag-everything must NOT promote) ──

def _env2(*, catch_worst, fp_worst):
    """Envelope with BOTH axes: catch_rate floor (min) + fp_rate floor (max)."""
    return {"catch_rate": {"worst_of_n": catch_worst},
            "fp_rate": {"worst_of_n": fp_worst}}


def test_fp_explosion_keeps_baseline_even_when_catch_wins():
    """The reviewer's exact counterexample: catch_rate up huge (0.70 -> 0.90
    worst-of-N) but fp_rate exploded (0.05 -> 0.80) and accuracy collapsed.
    Under the OLD catch-only rule this promoted; it MUST now keep baseline."""
    base = _eval(0.90, 0.70, 0.05)
    chal = _eval(0.40, 1.00, 0.80, envelope=_env2(catch_worst=0.90, fp_worst=0.80))
    verdict = beats_baseline(base, chal)
    assert verdict.promote is False
    assert "fp" in verdict.reason.lower() or "false" in verdict.reason.lower()
    # The receipt must surface the FP delta.
    assert verdict.fp_baseline == 0.05
    assert verdict.fp_challenger == 0.80
    assert round(verdict.fp_delta, 4) == 0.75


def test_clean_win_catch_up_fp_flat_still_promotes():
    base = _eval(0.90, 0.70, 0.05)
    chal = _eval(0.95, 0.85, 0.05, envelope=_env2(catch_worst=0.80, fp_worst=0.05))
    verdict = beats_baseline(base, chal)
    assert verdict.promote is True
    assert verdict.fp_delta == 0.0


def test_clean_win_catch_up_fp_down_still_promotes():
    base = _eval(0.90, 0.70, 0.10)
    chal = _eval(0.96, 0.85, 0.04, envelope=_env2(catch_worst=0.80, fp_worst=0.04))
    verdict = beats_baseline(base, chal)
    assert verdict.promote is True
    assert verdict.fp_delta < 0  # FP improved


def test_fp_within_tolerance_promotes_boundary():
    """catch up, FP up by exactly the tolerance -> still promote (<=, not <)."""
    base = _eval(0.90, 0.70, 0.05)
    chal = _eval(0.92, 0.85, 0.07, envelope=_env2(catch_worst=0.80, fp_worst=0.07))
    verdict = beats_baseline(base, chal, fp_regression_tolerance=0.02)
    assert verdict.promote is True  # 0.07 <= 0.05 + 0.02


def test_fp_just_over_tolerance_keeps_baseline_boundary():
    """catch up, FP up just past the tolerance -> keep baseline (the other side)."""
    base = _eval(0.90, 0.70, 0.05)
    chal = _eval(0.92, 0.85, 0.08, envelope=_env2(catch_worst=0.80, fp_worst=0.08))
    verdict = beats_baseline(base, chal, fp_regression_tolerance=0.02)
    assert verdict.promote is False  # 0.08 > 0.05 + 0.02
    assert "fp" in verdict.reason.lower() or "false" in verdict.reason.lower()


def test_default_tolerance_is_zero_no_fp_increase_allowed():
    """Default fp_regression_tolerance = 0.0: ANY FP increase blocks promotion."""
    base = _eval(0.90, 0.70, 0.05)
    chal = _eval(0.92, 0.85, 0.051, envelope=_env2(catch_worst=0.80, fp_worst=0.051))
    verdict = beats_baseline(base, chal)  # default tolerance 0.0
    assert verdict.promote is False


def test_fp_judged_at_worst_max_across_reruns():
    """fp_rate is judged PESSIMISTICALLY: the envelope floor for fp is the MAX
    observed. A challenger whose headline fp looks fine but whose worst-of-N fp
    regressed must be blocked."""
    base = _eval(0.90, 0.70, 0.05)
    # headline fp 0.04 (looks better) but worst-of-N fp 0.30 (a bad draw).
    chal = _eval(0.92, 0.85, 0.04, envelope=_env2(catch_worst=0.80, fp_worst=0.30))
    verdict = beats_baseline(base, chal)
    assert verdict.promote is False
    assert verdict.fp_challenger == 0.30  # the worst, not the flattering headline


# ── Orchestration (stubbed seams) ────────────────────────────────

def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def _labelled_rows(n, *, fail_truth=False):
    out = []
    for i in range(n):
        per = {"truth": not fail_truth, "causality": True}
        out.append({"text": f"example number {i:02d} long enough to be kept here",
                    "per_lens": per, "domain": "general"})
    return out


@pytest.fixture
def improve_sandbox(tmp_path, monkeypatch):
    """Inputs + the three seams stubbed. Caller tweaks the stubs per test."""
    trainset = _write(tmp_path / "train.json", _labelled_rows(8))
    holdout = _write(tmp_path / "holdout.json", _labelled_rows(6))
    baseline_ckpt = tmp_path / "baseline.json"
    baseline_ckpt.write_text(json.dumps({"truth.predict": {}, "metadata": {}}))
    catches = tmp_path / "catches.jsonl"
    catches.write_text("")  # empty by default
    outdir = tmp_path / "out"
    profile = tmp_path / "profile.yaml"
    profile.write_text("name: t\nllm:\n  model: openai/gpt-4o\n")

    state = {"compile_calls": [], "compile_rc": EXIT_OK,
             "base_eval": _eval(0.90, 0.70, 0.05),
             "chal_eval": _eval(0.95, 0.85, 0.04,
                                envelope=_env("catch_rate", worst=0.80)),
             "mutation_rc": 0}

    def fake_compile(training_path, profile_obj, output_path, **kw):
        # Record what the challenger was compiled on; write a fake checkpoint
        # so the artifact exists on disk (only on a non-error rc).
        state["compile_calls"].append(
            {"training_path": str(training_path), "output_path": str(output_path),
             "kw": kw})
        rc = state["compile_rc"]
        if rc == EXIT_OK:
            from pathlib import Path as _P
            _P(output_path).parent.mkdir(parents=True, exist_ok=True)
            _P(output_path).write_text(json.dumps({"truth.predict": {"x": 1}}))
        return rc

    def fake_eval(checkpoint, holdout_path, profile_obj, **kw):
        # Baseline checkpoint path -> base_eval; challenger -> chal_eval.
        if checkpoint is not None and "baseline" in str(checkpoint):
            return state["base_eval"]
        if checkpoint is None:
            return state["base_eval"]
        return state["chal_eval"]

    def fake_mutation(checkpoint, holdout_path, profile_obj, **kw):
        return state["mutation_rc"]

    monkeypatch.setattr(imp, "_compile_challenger", fake_compile)
    monkeypatch.setattr(imp, "_eval_checkpoint", fake_eval)
    monkeypatch.setattr(imp, "_mutation_control", fake_mutation)

    return {"trainset": trainset, "holdout": holdout, "baseline": baseline_ckpt,
            "catches": catches, "outdir": outdir, "profile": profile,
            "state": state, "tmp_path": tmp_path, "monkeypatch": monkeypatch}


def _run(sb, **over):
    from lens_kit import Profile
    profile = Profile.load(str(sb["profile"]))
    kw = dict(trainset=str(sb["trainset"]), holdout=str(sb["holdout"]),
              baseline_checkpoint=str(sb["baseline"]), catches=str(sb["catches"]),
              output_dir=str(sb["outdir"]), profile=profile, command_str="lens-kit improve")
    kw.update(over)
    return run_improve(**kw)


def test_promote_on_win(improve_sandbox, capsys):
    sb = improve_sandbox
    rc = _run(sb)
    assert rc == EXIT_OK
    # Promoted checkpoint written to the output dir.
    promoted = sb["outdir"] / "promoted.json"
    assert promoted.exists()
    # Sidecar for the promoted checkpoint.
    assert promoted.with_suffix(".provenance.json").exists()
    # Ledger row says PROMOTED.
    from pathlib import Path
    ledger = Path.cwd() / "RUNS.md"
    assert ledger.exists()
    assert "PROMOTED" in ledger.read_text()
    out = capsys.readouterr().out
    assert "PROMOTED" in out


def test_keep_on_loss(improve_sandbox, capsys):
    sb = improve_sandbox
    sb["state"]["chal_eval"] = _eval(0.80, 0.60, 0.10,
                                     envelope=_env("catch_rate", worst=0.55))
    rc = _run(sb)
    assert rc == EXIT_OK  # keep-baseline is a successful, honest outcome
    # No promoted checkpoint.
    assert not (sb["outdir"] / "promoted.json").exists()
    # Challenger artifact retained for inspection.
    assert (sb["outdir"] / "challenger.json").exists()
    # Baseline untouched (still the original fake checkpoint).
    assert json.loads(sb["baseline"].read_text()) == {"truth.predict": {}, "metadata": {}}
    from pathlib import Path
    ledger_text = (Path.cwd() / "RUNS.md").read_text()
    assert "KEPT-BASELINE" in ledger_text


def test_keep_on_within_envelope(improve_sandbox):
    sb = improve_sandbox
    # Nominally higher catch but worst-of-N floor below baseline headline.
    sb["state"]["chal_eval"] = _eval(0.91, 0.82, 0.05,
                                     envelope=_env("catch_rate", worst=0.69))
    rc = _run(sb)
    assert rc == EXIT_OK
    assert not (sb["outdir"] / "promoted.json").exists()
    from pathlib import Path
    assert "KEPT-BASELINE" in (Path.cwd() / "RUNS.md").read_text()


def test_mutation_control_veto(improve_sandbox, capsys):
    sb = improve_sandbox
    # Challenger wins on accuracy, but mutation control fails (rubber-stamp).
    sb["state"]["mutation_rc"] = 5
    rc = _run(sb)
    assert rc == 5  # the integrity signal propagates
    assert not (sb["outdir"] / "promoted.json").exists()
    # Baseline untouched.
    assert json.loads(sb["baseline"].read_text()) == {"truth.predict": {}, "metadata": {}}
    from pathlib import Path
    ledger_text = (Path.cwd() / "RUNS.md").read_text()
    assert "KEPT-BASELINE" in ledger_text or "MUTATION" in ledger_text.upper()
    out = capsys.readouterr().out
    assert "mutation" in out.lower()


def test_costing_gate_stop_propagates(improve_sandbox, capsys):
    sb = improve_sandbox
    sb["state"]["compile_rc"] = 3  # the >2h hard stop
    rc = _run(sb)
    assert rc == 3
    assert not (sb["outdir"] / "promoted.json").exists()
    from pathlib import Path
    ledger_text = (Path.cwd() / "RUNS.md").read_text()
    assert "STOPPED" in ledger_text.upper() or "COST" in ledger_text.upper()


def test_pace_kill_propagates(improve_sandbox):
    sb = improve_sandbox
    sb["state"]["compile_rc"] = 4
    rc = _run(sb)
    assert rc == 4
    assert not (sb["outdir"] / "promoted.json").exists()


# ── Trainset merge (catch labels) ────────────────────────────────

def test_trainset_merge_adds_catch_labels(improve_sandbox, capsys):
    sb = improve_sandbox
    # A catch with snippet + lenses_failed becomes a training label.
    catch = {
        "id": "c1", "date": "2026-06-13", "domain": "general",
        "artifact_type": "landing-copy", "catch": "fabricated stat",
        "pattern": "p", "rule": "r", "self_catch": False, "schema_version": 1,
        "snippet": "This product guarantees a 73.4% improvement for everyone.",
        "lenses_failed": ["truth"],
    }
    sb["catches"].write_text(json.dumps(catch) + "\n")
    rc = _run(sb)
    assert rc == EXIT_OK
    # The challenger trainset (passed to compile) has base + 1 catch label.
    call = sb["state"]["compile_calls"][0]
    merged = json.loads(__import__("pathlib").Path(call["training_path"]).read_text())
    texts = [r["text"] for r in merged]
    assert "This product guarantees a 73.4% improvement for everyone." in texts
    assert len(merged) == 8 + 1  # base 8 + 1 catch label
    out = capsys.readouterr().out
    assert "1 catch-label" in out or "1 catch label" in out


def test_trainset_merge_dedupes_against_base(improve_sandbox):
    sb = improve_sandbox
    # The catch snippet is IDENTICAL to a base trainset text -> deduped out.
    base = json.loads(sb["trainset"].read_text())
    dup_text = base[0]["text"]
    catch = {
        "id": "c1", "date": "2026-06-13", "domain": "general",
        "artifact_type": "x", "catch": "dup", "pattern": "p", "rule": "r",
        "self_catch": False, "schema_version": 1,
        "snippet": dup_text, "lenses_failed": ["truth"],
    }
    sb["catches"].write_text(json.dumps(catch) + "\n")
    rc = _run(sb)
    assert rc == EXIT_OK
    call = sb["state"]["compile_calls"][0]
    merged = json.loads(__import__("pathlib").Path(call["training_path"]).read_text())
    # No growth: the catch label collided with the base text.
    assert len(merged) == 8


def test_zero_eligible_catches_proceeds_base_only(improve_sandbox, capsys):
    sb = improve_sandbox
    # A catch with NO snippet/lenses_failed -> not eligible -> 0 labels added.
    catch = {"id": "c1", "date": "2026-06-13", "domain": "general",
             "artifact_type": "x", "catch": "vague catch with no snippet",
             "pattern": "p", "rule": "r", "self_catch": False, "schema_version": 1}
    sb["catches"].write_text(json.dumps(catch) + "\n")
    rc = _run(sb)
    assert rc == EXIT_OK  # base-only recompile is still valid
    call = sb["state"]["compile_calls"][0]
    merged = json.loads(__import__("pathlib").Path(call["training_path"]).read_text())
    assert len(merged) == 8
    out = capsys.readouterr().out
    assert "0 catch-label" in out or "0 catch label" in out


# ── FIX 2: single-measurement promotion needs an explicit opt-in ──

def test_single_measurement_blocked_without_optin(improve_sandbox, capsys):
    sb = improve_sandbox
    # A clear win but NO envelope (reruns==1, single measurement).
    sb["state"]["chal_eval"] = _eval(0.95, 0.85, 0.04)  # no envelope
    rc = _run(sb)  # default: allow_single_measurement=False
    assert rc == EXIT_OK  # keep-baseline is a clean, honest outcome
    assert not (sb["outdir"] / "promoted.json").exists()
    from pathlib import Path
    ledger_text = (Path.cwd() / "RUNS.md").read_text()
    assert "KEPT-BASELINE" in ledger_text
    out = capsys.readouterr().out
    assert "--reruns" in out or "--allow-single-measurement" in out


def test_single_measurement_promotes_with_optin(improve_sandbox, capsys):
    sb = improve_sandbox
    sb["state"]["chal_eval"] = _eval(0.95, 0.85, 0.04)  # no envelope
    rc = _run(sb, allow_single_measurement=True)
    assert rc == EXIT_OK
    promoted = sb["outdir"] / "promoted.json"
    assert promoted.exists()
    # The honest single-measurement caveat is still recorded in the sidecar.
    sidecar = json.loads(promoted.with_suffix(".provenance.json").read_text())
    assert any("single" in c.lower() for c in sidecar["not_a_claim_of"])
    assert sidecar["single_measurement"] is True


def test_single_measurement_optin_does_not_bypass_other_gates(improve_sandbox):
    """The opt-in only waives the envelope requirement — it does NOT let a loss,
    an FP regression, or a mutation failure through."""
    sb = improve_sandbox
    # Single measurement, catch wins, but FP exploded -> still keep baseline.
    sb["state"]["chal_eval"] = _eval(0.40, 0.90, 0.80)  # no envelope, FP up
    rc = _run(sb, allow_single_measurement=True)
    assert rc == EXIT_OK
    assert not (sb["outdir"] / "promoted.json").exists()


# ── FIX 3: an inconclusive (non-{0,5}) mutation rc maps to one outcome ──

def test_inconclusive_mutation_rc_keeps_baseline(improve_sandbox, capsys):
    sb = improve_sandbox
    # Challenger wins; mutation control returns a non-{0,5} rc (e.g. 2 usage).
    sb["state"]["mutation_rc"] = 2
    rc = _run(sb)
    from lens_kit.improve import EXIT_INCONCLUSIVE_INTEGRITY
    assert rc == EXIT_INCONCLUSIVE_INTEGRITY
    assert not (sb["outdir"] / "promoted.json").exists()
    # Baseline untouched.
    assert json.loads(sb["baseline"].read_text()) == {"truth.predict": {}, "metadata": {}}
    from pathlib import Path
    assert "KEPT-BASELINE" in (Path.cwd() / "RUNS.md").read_text()
    out = capsys.readouterr().out
    assert "inconclusive" in out.lower()


# ── Input validation ─────────────────────────────────────────────

def test_missing_trainset_exits_2(improve_sandbox):
    sb = improve_sandbox
    rc = _run(sb, trainset=str(sb["tmp_path"] / "nope.json"))
    assert rc == EXIT_USAGE


def test_missing_holdout_exits_2(improve_sandbox):
    sb = improve_sandbox
    rc = _run(sb, holdout=str(sb["tmp_path"] / "nope.json"))
    assert rc == EXIT_USAGE


def test_missing_baseline_checkpoint_exits_2(improve_sandbox):
    sb = improve_sandbox
    rc = _run(sb, baseline_checkpoint=str(sb["tmp_path"] / "nope.json"))
    assert rc == EXIT_USAGE


def test_missing_catches_exits_2(improve_sandbox):
    sb = improve_sandbox
    rc = _run(sb, catches=str(sb["tmp_path"] / "nope.json"))
    assert rc == EXIT_USAGE


# ── --status ─────────────────────────────────────────────────────

def test_status_reports_last_outcome(improve_sandbox, capsys):
    sb = improve_sandbox
    # First run a promote so the ledger has an improve row.
    _run(sb)
    capsys.readouterr()  # clear
    from pathlib import Path
    rc = imp.run_improve_status(ledger_path=str(Path.cwd() / "RUNS.md"))
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "PROMOTED" in out


def test_status_empty_ledger(improve_sandbox, capsys):
    sb = improve_sandbox
    rc = imp.run_improve_status(ledger_path=str(sb["tmp_path"] / "RUNS.md"))
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "no improve" in out.lower()
