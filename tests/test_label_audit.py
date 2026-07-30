"""Label-audit workflow: report shape, suspicion detection, impact math,
backup enforcement, and the LABEL_AUDIT ledger row. All no-network.
"""
import datetime
import json
from pathlib import Path

import pytest

from lens_kit.label_audit import (backup_dataset, find_suspicious, impact_math,
                                  render_report, run_label_audit)


def _dataset(tmp_path):
    """Gold labels: 3 examples, indices 0/1/2."""
    data = tmp_path / "holdout.json"
    data.write_text(json.dumps({
        "_metadata": {"created": "2026-06-01"},
        "examples": [
            {"text": "Sessions rose to 5030 in May, math checks out.",
             "source_id": "ex-0", "per_lens": {"consistency": True}},
            {"text": "8 developers, then 14 engineers — same team.",
             "source_id": "ex-1", "per_lens": {"consistency": False}},
            {"text": "A clean sentence with no defects at all.",
             "source_id": "ex-2", "per_lens": {"truth": True}},
        ],
    }))
    return data


def _eval_results(tmp_path):
    """Eval per_example: ex-0 has an FP (model flagged clean), ex-1 an FN
    (model missed a planted flaw), ex-2 agrees."""
    res = tmp_path / "eval-run1.json"
    res.write_text(json.dumps({
        "per_example": [
            {"index": 0, "source_id": "ex-0", "score": 0.0,
             "lens_confusion": {"consistency": "fp"}},
            {"index": 1, "source_id": "ex-1", "score": 0.0,
             "lens_confusion": {"consistency": "fn"}},
            {"index": 2, "source_id": "ex-2", "score": 1.0,
             "lens_confusion": {"truth": "tn"}},
        ],
    }))
    return res


# ── Suspicion detection + impact math ────────────────────────────

def test_find_suspicious_flags_only_disagreements(tmp_path):
    gold = json.loads(_dataset(tmp_path).read_text())["examples"]
    per_ex = json.loads(_eval_results(tmp_path).read_text())["per_example"]
    sus = find_suspicious(gold, per_ex)
    ids = {r["source_id"] for r in sus}
    assert ids == {"ex-0", "ex-1"}        # ex-2 agrees -> not suspicious


def test_impact_math_counts_fn_and_fp_upper_bound(tmp_path):
    gold = json.loads(_dataset(tmp_path).read_text())["examples"]
    per_ex = json.loads(_eval_results(tmp_path).read_text())["per_example"]
    impact = impact_math(find_suspicious(gold, per_ex))
    assert impact["suspicious_examples"] == 2
    assert impact["max_fn_flips"] == 1     # ex-1 consistency fn
    assert impact["max_fp_flips"] == 1     # ex-0 consistency fp
    assert impact["per_lens"]["consistency"] == {"fn": 1, "fp": 1}


def test_eval_without_per_example_is_a_hard_error(tmp_path):
    data = _dataset(tmp_path)
    bad = tmp_path / "envelope.json"
    bad.write_text(json.dumps({"metrics": {}}))      # envelope, no per_example
    with pytest.raises(ValueError, match="no 'per_example'"):
        run_label_audit(data, bad)


# ── Report shape ─────────────────────────────────────────────────

def test_report_has_three_sections_and_impact(tmp_path):
    gold = json.loads(_dataset(tmp_path).read_text())["examples"]
    per_ex = json.loads(_eval_results(tmp_path).read_text())["per_example"]
    sus = find_suspicious(gold, per_ex)
    report = render_report(sus, impact_math(sus), dataset_name="holdout.json",
                           eval_name="eval-run1.json", new_run_id="relabel-X")
    for header in ("## Confirmed Label Errors", "## Borderline / Worth Reviewing",
                   "## False Alarms", "## Impact on Eval Scores", "## Applied"):
        assert header in report
    assert "relabel-X" in report                     # new-run designation surfaced
    assert "ex-0" in report and "ex-1" in report     # both suspicious rows present
    assert "UPPER BOUND" in report.upper()


# ── Backup enforcement (mechanical discipline) ───────────────────

def test_backup_dataset_creates_dated_copy_never_overwrites(tmp_path):
    data = _dataset(tmp_path)
    b1 = backup_dataset(data)
    b2 = backup_dataset(data)
    date = datetime.date.today().strftime("%Y%m%d")
    assert b1.name == f"holdout.backup.{date}.json"
    assert b2.name == f"holdout.backup.{date}-2.json"   # second never overwrites first
    assert b1.read_text() == data.read_text()


def test_backup_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        backup_dataset(tmp_path / "absent.json")


def test_run_without_backup_prints_new_run_rule(tmp_path, capsys):
    rc = run_label_audit(_dataset(tmp_path), _eval_results(tmp_path),
                         output_path=tmp_path / "audit.md")
    assert rc == 0
    out = capsys.readouterr().out
    assert "NEW-RUN-DESIGNATION RULE" in out
    assert "--backup" in out
    # No backup created without the flag.
    assert not list(tmp_path.glob("*.backup.*.json"))


def test_run_with_backup_creates_copy(tmp_path, capsys):
    rc = run_label_audit(_dataset(tmp_path), _eval_results(tmp_path),
                         backup=True, output_path=tmp_path / "audit.md")
    assert rc == 0
    assert "Backup created" in capsys.readouterr().out
    assert list(tmp_path.glob("holdout.backup.*.json"))


def test_run_appends_label_audit_ledger_row(tmp_path):
    run_label_audit(_dataset(tmp_path), _eval_results(tmp_path),
                    output_path=tmp_path / "audit.md")
    runs = (Path.cwd() / "RUNS.md").read_text()
    assert "LABEL_AUDIT" in runs


def test_audit_report_written_to_output_path(tmp_path):
    out = tmp_path / "my-audit.md"
    run_label_audit(_dataset(tmp_path), _eval_results(tmp_path), output_path=out)
    assert out.exists()
    assert "Hold-Out Label Audit" in out.read_text()
