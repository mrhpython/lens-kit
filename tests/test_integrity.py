"""Gate-integrity trend monitor — the continuous version of mutation control.

No network (deterministic, reads RUNS.md + catches.jsonl). A validator that
approves everything for N runs must be SURFACED, not trusted. These tests
prove the rubber-stamp ledger is flagged (and --strict exits 6), a healthy
ledger stays quiet (--strict exits 0), and malformed/empty inputs are
usage-grade errors (exit 2), never crashes.
"""
import json
from pathlib import Path

import pytest

from lens_kit.config import IntegrityConfig, Profile
from lens_kit.integrity import (EXIT_INTEGRITY, EXIT_OK, EXIT_USAGE,
                                IntegrityError, analyze_integrity, parse_ledger,
                                run_integrity)
from lens_kit.ledger import HEADER


# ── Synthetic ledger builders ────────────────────────────────────

def _ledger(rows: list[str]) -> str:
    return HEADER + "".join(rows)


def _row(run_id, command, sha, data_file, metrics, status, date="2026-06-01 10:00"):
    sha_cell = (sha[:12] if sha else "-")
    return (f"| {run_id} | {date} | {command} | {sha_cell} | {data_file} "
            f"| {metrics} | {status} |\n")


def _eval_row(idx, accuracy, date="2026-06-01 10:00"):
    return _row(f"eval-2026060{idx}-100000", "lens-kit eval holdout.json", None,
                "holdout.json", f"overall_accuracy_floor={accuracy:.3f}; "
                f"catch_rate_floor=0.900; fp_rate_floor=0.050", "EVAL", date=date)


def _rubber_stamp_ledger(n=6):
    """High AND rising pass-rate (accuracy pinned near 1.0), zero catches.

    Mean accuracy clears the default pass_rate_warn (0.95): the gate is
    approving almost everything, run after run — the rubber-stamp shape.
    """
    rows = []
    for i in range(n):
        acc = 0.95 + i * 0.008          # 0.95 -> ~0.99, high and rising
        rows.append(_eval_row(1, min(acc, 0.995), date=f"2026-06-0{i+1} 10:00"))
    return _ledger(rows)


def _healthy_ledger():
    """Mixed verdicts, a recent mutation run, accuracy NOT pinned near 1.0."""
    rows = [
        _eval_row(1, 0.82, date="2026-06-01 10:00"),
        _eval_row(2, 0.79, date="2026-06-02 10:00"),
        _eval_row(3, 0.84, date="2026-06-03 10:00"),
        _row("mutate-20260603-110000", "lens-kit mutate holdout.json", None,
             "holdout.json", "missed=0; mutants=3", "MUTATE",
             date="2026-06-03 11:00"),
        _eval_row(4, 0.81, date="2026-06-04 10:00"),
    ]
    return _ledger(rows)


def _catches(records: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


# ── parse_ledger ─────────────────────────────────────────────────

def test_parse_ledger_extracts_typed_rows(tmp_path):
    p = tmp_path / "RUNS.md"
    p.write_text(_healthy_ledger())
    rows = parse_ledger(p)
    assert len(rows) == 5
    assert rows[0]["kind"] == "eval"
    assert rows[3]["kind"] == "mutate"
    # accuracy parsed from the metrics cell
    assert abs(rows[0]["accuracy"] - 0.82) < 1e-9


def test_parse_ledger_missing_file_is_usage_error(tmp_path):
    with pytest.raises(IntegrityError):
        parse_ledger(tmp_path / "absent.md")


def test_parse_ledger_empty_ledger_is_usage_error(tmp_path):
    p = tmp_path / "RUNS.md"
    p.write_text(HEADER)               # header only, no rows
    with pytest.raises(IntegrityError):
        parse_ledger(p)


# ── analyze_integrity: the rubber-stamp tripwire ─────────────────

def test_rubber_stamp_ledger_is_flagged(tmp_path):
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_rubber_stamp_ledger(n=6))
    catches = tmp_path / "catches.jsonl"
    catches.write_text("")             # zero catches
    report = analyze_integrity(ledger, catches, last=6, config=IntegrityConfig())
    assert report["flagged"] is True
    # the specific tripwire: high+rising pass-rate AND zero catches in window
    assert any("rubber" in f.lower() or "approv" in f.lower()
               for f in report["flags"])
    assert report["pass_rate"] is not None
    assert report["trend"] in ("rising", "flat", "falling")


def test_healthy_ledger_is_quiet(tmp_path):
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_healthy_ledger())
    catches = tmp_path / "catches.jsonl"
    catches.write_text(_catches([
        {"id": "c1", "date": "2026-06-04", "domain": "marketing",
         "catch": "a real recent defect", "pattern": "p", "rule": "r"},
    ]))
    report = analyze_integrity(ledger, catches, last=5, config=IntegrityConfig())
    assert report["flagged"] is False
    assert report["flags"] == []


# ── run_integrity exit codes ─────────────────────────────────────

def test_run_integrity_report_only_default_exit_0_even_when_flagged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_rubber_stamp_ledger(n=6))
    (tmp_path / "catches.jsonl").write_text("")
    rc = run_integrity(ledger, tmp_path / "catches.jsonl", last=6,
                       strict=False, config=IntegrityConfig(),
                       command_str="lens-kit integrity")
    assert rc == EXIT_OK               # report-only: a flag does not gate by default


def test_run_integrity_strict_exits_6_on_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_rubber_stamp_ledger(n=6))
    (tmp_path / "catches.jsonl").write_text("")
    rc = run_integrity(ledger, tmp_path / "catches.jsonl", last=6,
                       strict=True, config=IntegrityConfig(),
                       command_str="lens-kit integrity --strict")
    assert rc == EXIT_INTEGRITY


def test_run_integrity_strict_exits_0_on_healthy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_healthy_ledger())
    catches = tmp_path / "catches.jsonl"
    catches.write_text(_catches([
        {"id": "c1", "date": "2026-06-04", "domain": "marketing",
         "catch": "a real recent defect", "pattern": "p", "rule": "r"}]))
    rc = run_integrity(ledger, catches, last=5, strict=True,
                       config=IntegrityConfig(), command_str="lens-kit integrity --strict")
    assert rc == EXIT_OK


def test_run_integrity_appends_a_ledger_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_healthy_ledger())
    catches = tmp_path / "catches.jsonl"
    catches.write_text(_catches([{"id": "c1", "date": "2026-06-04",
                                  "domain": "g", "catch": "x", "pattern": "p", "rule": "r"}]))
    before = ledger.read_text().count("\n")
    run_integrity(ledger, catches, last=5, strict=False, config=IntegrityConfig(),
                  command_str="lens-kit integrity")
    after = ledger.read_text()
    assert after.count("\n") == before + 1
    assert "integrity-" in after


# ── Honest "unknown" surfacing (mutation recency unidentifiable) ──

def test_no_mutation_runs_surfaces_unknown_not_a_guess(tmp_path):
    ledger = tmp_path / "RUNS.md"
    # eval-only ledger: no mutate- rows at all
    ledger.write_text(_ledger([_eval_row(1, 0.82, date=f"2026-06-0{i+1} 10:00")
                               for i in range(5)]))
    catches = tmp_path / "catches.jsonl"
    catches.write_text(_catches([{"id": "c1", "date": "2026-06-05", "domain": "g",
                                  "catch": "x", "pattern": "p", "rule": "r"}]))
    report = analyze_integrity(ledger, catches, last=5, config=IntegrityConfig())
    assert report["last_mutation"] == "unknown"


# ── catches recency ──────────────────────────────────────────────

def test_zero_catches_window_recorded(tmp_path):
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_rubber_stamp_ledger(n=6))
    catches = tmp_path / "catches.jsonl"
    catches.write_text("")
    report = analyze_integrity(ledger, catches, last=6, config=IntegrityConfig())
    assert report["catches_total"] == 0
    assert report["runs_since_last_catch"] is None   # never caught -> None, honest


def test_recent_catch_clears_the_stale_flag(tmp_path):
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_rubber_stamp_ledger(n=6))
    catches = tmp_path / "catches.jsonl"
    # a catch dated within the window defuses the rubber-stamp tripwire
    catches.write_text(_catches([{"id": "c1", "date": "2026-06-06", "domain": "g",
                                  "catch": "a real defect just found", "pattern": "p",
                                  "rule": "r"}]))
    report = analyze_integrity(ledger, catches, last=6, config=IntegrityConfig())
    # rising pass-rate alone is not enough — a recent catch is evidence of life
    assert report["flagged"] is False


# ── malformed inputs are usage-grade, not crashes ────────────────

def test_malformed_catches_is_usage_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_healthy_ledger())
    catches = tmp_path / "catches.jsonl"
    catches.write_text("{not valid json\n")
    rc = run_integrity(ledger, catches, last=5, strict=False, config=IntegrityConfig(),
                       command_str="lens-kit integrity")
    assert rc == EXIT_USAGE


def test_run_integrity_missing_ledger_is_usage_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = run_integrity(tmp_path / "absent.md", tmp_path / "catches.jsonl",
                       last=5, strict=False, config=IntegrityConfig(),
                       command_str="lens-kit integrity")
    assert rc == EXIT_USAGE


# ── IntegrityConfig profile plumbing ─────────────────────────────

def test_integrity_config_defaults_are_conservative():
    cfg = IntegrityConfig()
    assert cfg.pass_rate_warn > 0.5
    assert cfg.min_runs >= 2
    assert cfg.stale_catch_runs >= 1


def test_profile_without_integrity_block_still_valid():
    # Old profiles (no integrity: key) must stay loadable with defaults.
    p = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"}})
    assert isinstance(p.integrity, IntegrityConfig)


def test_profile_integrity_block_round_trips():
    p = Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"},
                           "integrity": {"pass_rate_warn": 0.97, "min_runs": 4}})
    assert p.integrity.pass_rate_warn == 0.97
    assert p.integrity.min_runs == 4
    assert p.to_dict()["integrity"]["pass_rate_warn"] == 0.97


# ── STALE MEMORY flag branch (positive assertion, discriminating) ──

def _stale_memory_ledger():
    """11 moderate-accuracy eval runs, ALL dated after an OLD catch.

    pass_rate 0.80 keeps the rubber-stamp branch from firing, so the only flags
    that can appear here are the two STALE MEMORY branches — discriminating.
    """
    rows = [_eval_row(1, 0.80, date=f"2026-06-{i:02d} 10:00") for i in range(1, 12)]
    return _ledger(rows)


def test_stale_memory_flag_fires_on_old_catch(tmp_path):
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_stale_memory_ledger())
    catches = tmp_path / "catches.jsonl"
    # last catch on 2026-05-01 -> 11 eval runs after it (>= 10) AND 41 days
    # (>= 30): both STALE MEMORY thresholds breached.
    catches.write_text(_catches([{"id": "c1", "date": "2026-05-01", "domain": "g",
                                  "catch": "an old defect", "pattern": "p", "rule": "r"}]))
    report = analyze_integrity(ledger, catches, last=11, config=IntegrityConfig())
    assert report["flagged"] is True
    stale_runs = [f for f in report["flags"] if f.startswith("STALE MEMORY:")
                  and "runs since" in f]
    stale_days = [f for f in report["flags"] if f.startswith("STALE MEMORY:")
                  and "days since" in f]
    assert stale_runs, "runs-since-catch STALE MEMORY flag did not fire"
    assert stale_days, "days-since-catch STALE MEMORY flag did not fire"
    # discriminating: the rubber-stamp branch must NOT fire (pass-rate is 0.80)
    assert not any("RUBBER-STAMP" in f for f in report["flags"])
    # numbers in the flag track the real thresholds (fail if branch math breaks)
    assert "11 runs" in stale_runs[0]
    assert report["runs_since_last_catch"] == 11
    assert report["days_since_last_catch"] >= 30


def test_stale_memory_silent_when_catch_is_recent(tmp_path):
    # The discriminator's negative half: a recent catch on the SAME ledger
    # clears both STALE MEMORY branches (the flag is not unconditional).
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_stale_memory_ledger())
    catches = tmp_path / "catches.jsonl"
    catches.write_text(_catches([{"id": "c1", "date": "2026-06-11", "domain": "g",
                                  "catch": "a recent defect", "pattern": "p", "rule": "r"}]))
    report = analyze_integrity(ledger, catches, last=11, config=IntegrityConfig())
    assert not any(f.startswith("STALE MEMORY:") for f in report["flags"])


# ── STALE MUTATION CONTROL flag branch (positive, discriminating) ──

def _stale_mutation_ledger():
    """A mutation run, then 21 moderate-accuracy eval runs all dated after it.

    A RECENT catch is added by the test so STALE MEMORY does not fire; pass_rate
    0.80 keeps the rubber-stamp branch quiet — so the only flag that can appear
    is STALE MUTATION CONTROL. Returns (ledger_text, last_eval_date_iso).
    """
    import datetime
    rows = [_row("mutate-20260401-100000", "lens-kit mutate holdout.json", None,
                 "holdout.json", "missed=0; mutants=3", "MUTATE",
                 date="2026-04-01 10:00")]
    base = datetime.date(2026, 4, 2)
    last_date = base
    for i in range(21):                        # 21 evals (>= stale_mutation_runs 20)
        last_date = base + datetime.timedelta(days=i * 3)
        rows.append(_eval_row(1, 0.80, date=f"{last_date.isoformat()} 10:00"))
    return _ledger(rows), last_date.isoformat()


def test_stale_mutation_control_flag_fires(tmp_path):
    ledger = tmp_path / "RUNS.md"
    text, last_eval_iso = _stale_mutation_ledger()
    ledger.write_text(text)
    catches = tmp_path / "catches.jsonl"
    # recent catch (dated at the last eval) so STALE MEMORY stays quiet
    catches.write_text(_catches([{"id": "c1", "date": last_eval_iso, "domain": "g",
                                  "catch": "a recent defect", "pattern": "p", "rule": "r"}]))
    report = analyze_integrity(ledger, catches, last=21, config=IntegrityConfig())
    assert report["flagged"] is True
    mut_flags = [f for f in report["flags"] if f.startswith("STALE MUTATION CONTROL:")]
    assert mut_flags, "STALE MUTATION CONTROL flag did not fire"
    assert "21 eval runs" in mut_flags[0]
    assert report["runs_since_mutation"] == 21
    assert report["last_mutation"] == "2026-04-01 10:00"     # not 'unknown'
    # discriminating: the OTHER branches must NOT fire on this fixture
    assert not any("RUBBER-STAMP" in f for f in report["flags"])
    assert not any(f.startswith("STALE MEMORY:") for f in report["flags"])


def test_stale_mutation_silent_when_mutation_recent(tmp_path):
    # Negative half: a fresh mutation run after the evals clears the flag.
    import datetime
    rows = []
    base = datetime.date(2026, 4, 2)
    last_date = base
    for i in range(21):
        last_date = base + datetime.timedelta(days=i * 3)
        rows.append(_eval_row(1, 0.80, date=f"{last_date.isoformat()} 10:00"))
    fresh = last_date + datetime.timedelta(days=1)
    rows.append(_row("mutate-20260601-100000", "lens-kit mutate holdout.json", None,
                     "holdout.json", "missed=0; mutants=3", "MUTATE",
                     date=f"{fresh.isoformat()} 10:00"))
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(_ledger(rows))
    catches = tmp_path / "catches.jsonl"
    catches.write_text(_catches([{"id": "c1", "date": last_date.isoformat(),
                                  "domain": "g", "catch": "recent", "pattern": "p", "rule": "r"}]))
    report = analyze_integrity(ledger, catches, last=21, config=IntegrityConfig())
    assert report["runs_since_mutation"] == 0
    assert not any(f.startswith("STALE MUTATION CONTROL:") for f in report["flags"])
