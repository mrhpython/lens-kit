"""Costing gate: projection math, hard stop, pace monitor kill record.

No network: pace probes are monkeypatched, the GEPA compile call is replaced
with sentinels that either record invocation or assert they were never
reached. The hard-stop tests are the point — projection over threshold
without --approve-cost must exit non-zero BEFORE any compile call.
"""
import json
from pathlib import Path

import pytest

from lens_kit.costing import (CostProjection, PaceExceeded, PaceMonitor,
                              gepa_projected_metric_calls, measure_pace,
                              measure_pace_fresh_cache)


# ── Projection math ──────────────────────────────────────────────

def test_max_evals_projection_is_budget_times_train_plus_val():
    assert gepa_projected_metric_calls(
        num_preds=11, train_size=80, val_size=20, max_evals=8) == 800


def test_auto_projection_matches_gepa_auto_budget():
    from dspy.teleprompt import GEPA
    from dspy.teleprompt.gepa.gepa import AUTO_RUN_SETTINGS

    for auto in ("light", "medium", "heavy"):
        expected = int(GEPA.auto_budget(
            None, num_preds=11,
            num_candidates=AUTO_RUN_SETTINGS[auto]["n"], valset_size=20))
        assert gepa_projected_metric_calls(
            num_preds=11, train_size=80, val_size=20, auto=auto) == expected


def test_projection_requires_exactly_one_budget_mode():
    with pytest.raises(ValueError):
        gepa_projected_metric_calls(num_preds=11, train_size=80, val_size=20)
    with pytest.raises(ValueError):
        gepa_projected_metric_calls(num_preds=11, train_size=80, val_size=20,
                                    auto="light", max_evals=4)


def test_cost_projection_hours_cost_and_threshold():
    p = CostProjection(
        projected_rollouts=1000, budget="max_evals=10", train_size=80,
        val_size=20, seconds_per_rollout=18.0, threads=2,
        cost_per_rollout_usd=0.01, threshold_hours=2.0, probe_rollouts=3)
    assert p.wall_seconds_per_rollout == 9.0
    assert p.projected_wall_seconds == 9000.0
    assert p.projected_hours == 2.5
    assert p.projected_cost_usd == 10.0
    assert p.requires_approval is True

    under = CostProjection(
        projected_rollouts=100, budget="max_evals=1", train_size=80,
        val_size=20, seconds_per_rollout=18.0, threads=2,
        cost_per_rollout_usd=None, threshold_hours=2.0)
    assert under.projected_cost_usd is None   # cost UNKNOWN without profile rate
    assert under.requires_approval is False
    assert "UNKNOWN" in under.format()


# ── Pace monitor / kill record ───────────────────────────────────

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_pace_monitor_quiet_under_approved_pace():
    clock = FakeClock()
    m = PaceMonitor(projected_rollouts=100, approved_wall_seconds_per_rollout=2.0,
                    kill_factor=1.5, min_calls=3, clock=clock)
    m.start()
    for _ in range(10):
        clock.t += 2.0          # exactly on pace
        m.record_call()
    assert not m.killed
    assert m.calls == 10


def test_pace_monitor_exempts_startup_noise():
    clock = FakeClock()
    m = PaceMonitor(projected_rollouts=100, approved_wall_seconds_per_rollout=1.0,
                    kill_factor=1.5, min_calls=5, clock=clock)
    m.start()
    clock.t += 100.0            # horrendous first call, but calls < min_calls
    m.record_call()
    assert not m.killed


def test_pace_monitor_kill_record_shape():
    clock = FakeClock()
    m = PaceMonitor(projected_rollouts=200, approved_wall_seconds_per_rollout=1.0,
                    kill_factor=1.5, min_calls=4, clock=clock)
    m.start()
    with pytest.raises(PaceExceeded) as exc:
        for _ in range(4):
            clock.t += 2.0      # 2.0s/rollout > 1.0 * 1.5
            m.record_call()
    rec = exc.value.record
    assert m.killed
    assert rec["kill"] is True
    assert rec["rollouts_done"] == 4
    assert rec["projected_rollouts"] == 200
    assert rec["percent_done"] == 2.0                    # 4/200
    assert rec["elapsed_seconds"] == 8.0
    assert rec["measured_seconds_per_rollout"] == 2.0
    assert rec["approved_wall_seconds_per_rollout"] == 1.0
    assert rec["kill_factor"] == 1.5
    assert rec["projected_remaining_seconds"] == (200 - 4) * 2.0
    assert "exceeds approved" in rec["reason"]
    assert rec["timestamp"]


# ── Hard stop through the CLI (no compile call, no network) ──────

def _write_training(path: Path, n: int = 10) -> Path:
    rows = [{"text": f"Example number {i} with enough characters to count.",
             "per_lens": {"truth": True, "causality": bool(i % 2)},
             "domain": "general"} for i in range(n)]
    path.write_text(json.dumps(rows))
    return path


def _write_profile(path: Path, extra: str = "") -> Path:
    path.write_text("name: t\nllm:\n  model: openai/gpt-4o\n" + extra)
    return path


@pytest.fixture
def compile_sandbox(tmp_path, monkeypatch):
    """Training data + profile + stubbed pace probe and GEPA call."""
    import lens_kit.compile_harness as ch

    training = _write_training(tmp_path / "training.json")
    profile = _write_profile(tmp_path / "profile.yaml")
    output = tmp_path / "ckpt.json"

    calls = {"gepa": 0}

    def fake_gepa(gate, train_set, val_set, **kwargs):
        calls["gepa"] += 1
        return gate

    monkeypatch.setattr(ch, "_run_gepa", fake_gepa)
    monkeypatch.setattr(ch, "_val_score", lambda *a, **k: 0.5)
    return {"training": training, "profile": profile, "output": output,
            "calls": calls, "module": ch, "monkeypatch": monkeypatch,
            "tmp_path": tmp_path}


def test_hard_stop_over_threshold_without_approve_cost(compile_sandbox, capsys):
    sb = compile_sandbox
    # 3600s/rollout -> hundreds of projected hours >> 2h default threshold
    sb["monkeypatch"].setattr(sb["module"], "measure_pace", lambda *a, **k: 3600.0)

    from lens_kit.cli import main
    rc = main(["compile", str(sb["training"]), "--profile", str(sb["profile"]),
               "--output", str(sb["output"]), "--max-evals", "2"])

    assert rc == 3                          # costing-gate stop, non-zero
    assert sb["calls"]["gepa"] == 0         # compile NEVER called
    assert not sb["output"].exists()        # no checkpoint written
    out = capsys.readouterr().out
    assert "HARD STOP" in out
    assert "--approve-cost" in out
    ledger = Path.cwd() / "RUNS.md"         # conftest chdirs to a tmp cwd
    assert ledger.exists()
    assert "STOPPED_COST" in ledger.read_text()


def test_approve_cost_flag_unblocks_compile(compile_sandbox):
    sb = compile_sandbox
    sb["monkeypatch"].setattr(sb["module"], "measure_pace", lambda *a, **k: 3600.0)

    from lens_kit.cli import main
    rc = main(["compile", str(sb["training"]), "--profile", str(sb["profile"]),
               "--output", str(sb["output"]), "--max-evals", "2",
               "--approve-cost"])

    assert rc == 0
    assert sb["calls"]["gepa"] == 1
    assert sb["output"].exists()


def test_under_threshold_compiles_without_flag(compile_sandbox):
    sb = compile_sandbox
    sb["monkeypatch"].setattr(sb["module"], "measure_pace", lambda *a, **k: 0.01)

    from lens_kit.cli import main
    rc = main(["compile", str(sb["training"]), "--profile", str(sb["profile"]),
               "--output", str(sb["output"]), "--max-evals", "2"])

    assert rc == 0
    assert sb["calls"]["gepa"] == 1
    assert sb["output"].exists()
    assert Path(str(sb["output"]) + ".sha256").exists()
    prov = json.loads(sb["output"].with_suffix(".provenance.json").read_text())
    assert prov["model"] == "openai/gpt-4o"
    assert prov["val_score"] == 0.5
    ledger_text = (Path.cwd() / "RUNS.md").read_text()
    assert "KEEP" in ledger_text


def test_profile_threshold_is_configurable(compile_sandbox):
    sb = compile_sandbox
    # ~0.022h projected; profile threshold 0.001h forces the stop
    profile = _write_profile(
        sb["tmp_path"] / "tight.yaml",
        extra="compile:\n  approval_threshold_hours: 0.001\n")
    sb["monkeypatch"].setattr(sb["module"], "measure_pace", lambda *a, **k: 2.0)

    from lens_kit.cli import main
    rc = main(["compile", str(sb["training"]), "--profile", str(profile),
               "--output", str(sb["output"]), "--max-evals", "2"])
    assert rc == 3
    assert sb["calls"]["gepa"] == 0


def test_val_score_failure_still_saves_checkpoint(compile_sandbox, capsys):
    """A billed compile must never lose its artifact to a val-eval crash."""
    sb = compile_sandbox
    sb["monkeypatch"].setattr(sb["module"], "measure_pace", lambda *a, **k: 0.01)

    def boom(*a, **k):
        raise RuntimeError("val eval blew up")

    sb["monkeypatch"].setattr(sb["module"], "_val_score", boom)

    from lens_kit.cli import main
    rc = main(["compile", str(sb["training"]), "--profile", str(sb["profile"]),
               "--output", str(sb["output"]), "--max-evals", "2"])

    assert rc == 0
    assert sb["output"].exists()                       # checkpoint still saved
    prov = json.loads(sb["output"].with_suffix(".provenance.json").read_text())
    assert prov["val_score"] is None                   # honest: no invented score
    assert "WARNING: val-score eval failed" in capsys.readouterr().out
    text = (Path.cwd() / "RUNS.md").read_text()
    assert "KEEP" in text and "val_score=None (val eval failed)" in text


def test_save_failure_falls_back_and_writes_failed_row(compile_sandbox, capsys):
    """Save crash: emergency fallback attempted, FAILED row written, exit != 0."""
    sb = compile_sandbox
    sb["monkeypatch"].setattr(sb["module"], "measure_pace", lambda *a, **k: 0.01)

    def bad_save(*a, **k):
        raise OSError("disk full")

    sb["monkeypatch"].setattr(sb["module"], "save_checkpoint", bad_save)

    from lens_kit.cli import main
    rc = main(["compile", str(sb["training"]), "--profile", str(sb["profile"]),
               "--output", str(sb["output"]), "--max-evals", "2"])

    assert rc == 1                                     # non-zero, not a silent 0
    assert not sb["output"].exists()
    emergencies = list(Path.cwd().glob("lens-kit-emergency-*.json"))
    assert len(emergencies) == 1                       # fallback save succeeded
    assert "emergency fallback saved" in capsys.readouterr().out
    text = (Path.cwd() / "RUNS.md").read_text()
    assert "FAILED" in text and "disk full" in text


def test_pace_kill_writes_kill_record_and_exits_4(compile_sandbox):
    sb = compile_sandbox
    sb["monkeypatch"].setattr(sb["module"], "measure_pace", lambda *a, **k: 0.01)

    record = {"kill": True, "reason": "measured pace exceeds approved",
              "percent_done": 12.5, "rollouts_done": 5,
              "projected_rollouts": 40, "elapsed_seconds": 50.0,
              "measured_seconds_per_rollout": 10.0,
              "approved_wall_seconds_per_rollout": 0.01,
              "kill_factor": 1.5, "projected_remaining_seconds": 350.0,
              "timestamp": "2026-06-12T00:00:00"}

    def killing_gepa(gate, train_set, val_set, **kwargs):
        raise PaceExceeded(record)

    sb["monkeypatch"].setattr(sb["module"], "_run_gepa", killing_gepa)

    from lens_kit.cli import main
    rc = main(["compile", str(sb["training"]), "--profile", str(sb["profile"]),
               "--output", str(sb["output"]), "--max-evals", "2"])

    assert rc == 4
    kill_path = Path(str(sb["output"]) + ".kill.json")
    assert kill_path.exists()
    assert json.loads(kill_path.read_text()) == record
    assert not sb["output"].exists()
    assert "KILLED" in (Path.cwd() / "RUNS.md").read_text()


# ── S0: cache-replay pace probe (warm-cache false pace kill) ──────
#
# The warm-cache compile measured 0.11s/rollout from a WARM dspy cache
# (every probe call a cache replay), armed the pace monitor at 1.5x that, and
# the first real 37s/rollout call tripped a FALSE pace kill (exit 4). The fix:
# the probe is measured against an ISOLATED fresh cache by default, so the
# approved pace reflects real-call speed, not cache replay.

class _CacheModel:
    """Models a dspy disk cache: warm replays are fast, cold calls are slow.

    measure_pace_fresh_cache must reset the cache (cold) before probing, so a
    probe sees the SLOW pace — the honest figure. A warm probe would see the
    fast replay pace and lie.
    """

    def __init__(self, *, warm_pace: float, cold_pace: float):
        self.warm = True            # cache starts warm (prior run / eval left it warm)
        self.warm_pace = warm_pace
        self.cold_pace = cold_pace
        self.reset_count = 0

    def reset(self):                # what configure_fresh_cache effectively does
        self.warm = False
        self.reset_count += 1

    def pace(self) -> float:
        return self.warm_pace if self.warm else self.cold_pace


def test_fresh_cache_probe_resets_cache_then_restores(monkeypatch):
    """measure_pace_fresh_cache resets the cache before measuring + restores."""
    import lens_kit.costing as costing

    model = _CacheModel(warm_pace=0.1, cold_pace=37.0)
    restored = {"called": False}

    class _Restore:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            restored["called"] = True
            return False

    monkeypatch.setattr("lens_kit.eval_harness.preserve_dspy_cache", _Restore)
    monkeypatch.setattr("lens_kit.eval_harness.configure_fresh_cache", model.reset)
    # measure_pace is what actually times the program; here it reports the
    # cache model's current pace (cold after the fresh-cache reset).
    monkeypatch.setattr(costing, "measure_pace",
                        lambda *a, **k: model.pace())

    pace = measure_pace_fresh_cache(object(), [object()], probe_n=3)

    assert model.reset_count == 1           # cache was reset before probing
    assert pace == 37.0                     # measured the REAL (cold) pace, not 0.1
    assert restored["called"] is True       # live cache restored on exit


def test_warm_probe_would_have_measured_the_replay_pace():
    """Documents the bug: a warm-cache probe reports the cache-replay pace.

    measure_pace times the program as-is; with a warm cache that's the fast
    replay pace (0.1) — the figure that armed the warm-cache false kill.
    """
    model = _CacheModel(warm_pace=0.1, cold_pace=37.0)
    assert model.warm is True
    pace = measure_pace(lambda **kw: None, [], probe_n=0) if False else model.pace()
    assert pace == 0.1                      # warm probe lies — the bug


def test_default_probe_is_fresh_no_false_pace_kill(compile_sandbox):
    """End-to-end regression: warm-fast probe + slow real run must NOT pace-kill.

    Reproduces warm-cache. We model the cache so the probe, run with the
    DEFAULT (fresh) path, measures the SLOW real pace (37s) — high enough that
    the pace monitor's threshold (1.5x of 37s) is never blown by the real 37s
    rollouts. Before the fix the probe would have measured 0.1s and the first
    real rollout would have tripped a false kill.
    """
    sb = compile_sandbox
    model = _CacheModel(warm_pace=0.1, cold_pace=37.0)

    # Patch the fresh-cache machinery the probe relies on. preserve_dspy_cache
    # must stay a real context manager (measure_pace_fresh_cache uses `with`).
    sb["monkeypatch"].setattr(
        "lens_kit.eval_harness.configure_fresh_cache", model.reset)
    # The harness passes compile_harness.measure_pace as the timing function;
    # report the cache model's pace (cold after the fresh-cache reset = 37s).
    sb["monkeypatch"].setattr(sb["module"], "measure_pace",
                              lambda *a, **k: model.pace())

    # A GEPA run whose real rollouts tick the pace monitor at the real (cold)
    # 37s pace — within 1.5x of the (correctly fresh-probed) 37s approved pace.
    def real_paced_gepa(gate, train_set, val_set, *, monitor, **kwargs):
        clock = [0.0]
        monitor.clock = lambda: clock[0]
        monitor.start()
        for _ in range(monitor.projected_rollouts):
            clock[0] += 37.0          # real pace, equals approved → no kill
            monitor.record_call()
        return gate

    sb["monkeypatch"].setattr(sb["module"], "_run_gepa", real_paced_gepa)

    from lens_kit.cli import main
    # approval threshold raised so the (now honest, large) projection doesn't
    # trip the COST hard stop — this test targets the PACE kill, not the cost gate.
    profile = _write_profile(sb["tmp_path"] / "loose.yaml",
                             extra="compile:\n  approval_threshold_hours: 1000.0\n")
    rc = main(["compile", str(sb["training"]), "--profile", str(profile),
               "--output", str(sb["output"]), "--max-evals", "2"])

    assert model.reset_count >= 1           # probe used the fresh cache
    assert rc == 0                          # NOT 4 — no false pace kill
    assert not Path(str(sb["output"]) + ".kill.json").exists()
    assert sb["output"].exists()


def test_warm_probe_cache_flag_opts_out_of_fresh_probe(compile_sandbox):
    """--warm-probe-cache restores the old behavior (probe the live cache)."""
    sb = compile_sandbox
    reset = {"count": 0}
    sb["monkeypatch"].setattr(
        "lens_kit.eval_harness.configure_fresh_cache",
        lambda: reset.__setitem__("count", reset["count"] + 1))
    sb["monkeypatch"].setattr(sb["module"], "measure_pace", lambda *a, **k: 0.01)

    from lens_kit.cli import main
    rc = main(["compile", str(sb["training"]), "--profile", str(sb["profile"]),
               "--output", str(sb["output"]), "--max-evals", "2",
               "--warm-probe-cache"])

    assert rc == 0
    assert reset["count"] == 0               # fresh cache NEVER configured for the probe
