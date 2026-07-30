"""CLI argument validation and stubs — no network."""
import pytest

from lens_kit.cli import main
from lens_kit.config import builtin_profile_path


def test_calibrate_requires_subcommand():
    with pytest.raises(SystemExit) as exc:
        main(["calibrate"])
    assert exc.value.code == 2


def test_calibrate_generate_requires_profile_and_out():
    with pytest.raises(SystemExit) as exc:
        main(["calibrate", "generate"])
    assert exc.value.code == 2


def test_calibrate_generate_per_class_must_be_positive(tmp_path, capsys):
    rc = main(["calibrate", "generate",
               "--profile", str(builtin_profile_path("agency-example")),
               "--out", str(tmp_path / "battery"), "--per-class", "0"])
    assert rc == 2
    assert "per-class" in capsys.readouterr().err


def test_calibrate_run_missing_dir_exits_2(tmp_path, capsys):
    rc = main(["calibrate", "run", str(tmp_path / "absent"),
               "--profile", str(builtin_profile_path("agency-example"))])
    assert rc == 2
    assert "battery directory not found" in capsys.readouterr().err


def test_calibrate_generate_writes_fixtures_and_gold(tmp_path):
    out = tmp_path / "battery"
    rc = main(["calibrate", "generate",
               "--profile", str(builtin_profile_path("agency-example")),
               "--out", str(out)])
    assert rc == 0
    txts = sorted(out.glob("*.txt"))
    golds = sorted(out.glob("*.gold.json"))
    assert len(txts) == 24 and len(golds) == 24   # 8 classes x 3


def test_compile_missing_training_exits_2(tmp_path, capsys):
    rc = main(["compile", str(tmp_path / "absent.json"),
               "--profile", str(builtin_profile_path("agency-example")),
               "--output", str(tmp_path / "ckpt.json")])
    assert rc == 2
    assert "training data not found" in capsys.readouterr().err


def test_compile_budget_flags_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["compile", str(tmp_path / "data.json"),
              "--profile", "p.yaml", "--output", "c.json",
              "--max-evals", "4", "--auto", "light"])
    assert exc.value.code == 2


def test_eval_missing_holdout_exits_2(tmp_path, capsys):
    rc = main(["eval", str(tmp_path / "absent.json"),
               "--profile", str(builtin_profile_path("agency-example"))])
    assert rc == 2
    assert "holdout file not found" in capsys.readouterr().err


def test_eval_checkpoint_baseline_mutually_exclusive(tmp_path):
    f = tmp_path / "holdout.json"
    f.write_text("[]")
    with pytest.raises(SystemExit) as exc:
        main(["eval", str(f), "--profile", "p.yaml",
              "--checkpoint", "c.json", "--baseline"])
    assert exc.value.code == 2


def test_eval_missing_checkpoint_exits_2(tmp_path, capsys):
    f = tmp_path / "holdout.json"
    f.write_text("[]")
    rc = main(["eval", str(f),
               "--profile", str(builtin_profile_path("agency-example")),
               "--checkpoint", str(tmp_path / "absent-ckpt.json")])
    assert rc == 2
    assert "checkpoint not found" in capsys.readouterr().err


def test_no_command_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_validate_requires_profile():
    with pytest.raises(SystemExit) as exc:
        main(["validate", "somefile.md"])
    assert exc.value.code == 2


def test_validate_missing_file_exits_2(tmp_path, capsys):
    rc = main(["validate", str(tmp_path / "absent.md"),
               "--profile", str(builtin_profile_path("agency-example"))])
    assert rc == 2
    assert "file not found" in capsys.readouterr().err


def test_validate_missing_profile_exits_2(tmp_path, capsys):
    f = tmp_path / "text.md"
    f.write_text("hello")
    rc = main(["validate", str(f), "--profile", str(tmp_path / "absent.yaml")])
    assert rc == 2
    assert "Profile not found" in capsys.readouterr().err


def test_validate_fails_closed_on_missing_api_key(tmp_path, monkeypatch, capsys):
    """Profile names an env var that isn't set -> exit 2 before any LM call."""
    monkeypatch.delenv("LENS_KIT_CLI_ABSENT_KEY", raising=False)
    f = tmp_path / "text.md"
    f.write_text("hello")
    p = tmp_path / "profile.yaml"
    p.write_text(
        "name: t\nllm:\n  model: openai/gpt-4o\n  api_key_env: LENS_KIT_CLI_ABSENT_KEY\n"
    )
    rc = main(["validate", str(f), "--profile", str(p)])
    assert rc == 2
    assert "LENS_KIT_CLI_ABSENT_KEY" in capsys.readouterr().err


# ── C4: mutate / label-audit / sidecar arg validation + exit-code map ──

def test_mutate_missing_holdout_exits_2(tmp_path, capsys):
    rc = main(["mutate", str(tmp_path / "absent.json"),
               "--profile", str(builtin_profile_path("agency-example"))])
    assert rc == 2
    assert "holdout file not found" in capsys.readouterr().err


def test_mutate_per_type_must_be_positive(tmp_path, capsys):
    h = tmp_path / "holdout.json"
    h.write_text("[]")
    rc = main(["mutate", str(h),
               "--profile", str(builtin_profile_path("agency-example")),
               "--per-type", "0"])
    assert rc == 2
    assert "per-type" in capsys.readouterr().err


def test_mutate_empty_holdout_exits_2(tmp_path, capsys):
    h = tmp_path / "holdout.json"
    h.write_text("[]")
    rc = main(["mutate", str(h),
               "--profile", str(builtin_profile_path("agency-example"))])
    assert rc == 2
    assert "No usable examples" in capsys.readouterr().err


def test_label_audit_missing_dataset_exits_2(tmp_path, capsys):
    res = tmp_path / "eval.json"
    res.write_text("{}")
    rc = main(["label-audit", str(tmp_path / "absent.json"), str(res)])
    assert rc == 2
    assert "dataset not found" in capsys.readouterr().err


def test_label_audit_missing_eval_exits_2(tmp_path, capsys):
    data = tmp_path / "data.json"
    data.write_text("[]")
    rc = main(["label-audit", str(data), str(tmp_path / "absent-eval.json")])
    assert rc == 2
    assert "eval-results not found" in capsys.readouterr().err


def test_sidecar_missing_checkpoint_exits_2(tmp_path, capsys):
    rc = main(["sidecar", str(tmp_path / "absent.json"),
               "--profile", str(builtin_profile_path("agency-example"))])
    assert rc == 2
    assert "checkpoint not found" in capsys.readouterr().err


def test_sidecar_writes_full_sidecar(tmp_path, capsys):
    import json
    ckpt = tmp_path / "gate.json"
    ckpt.write_text(json.dumps({"metadata": {"k": 1}}))
    rc = main(["sidecar", str(ckpt),
               "--profile", str(builtin_profile_path("agency-example")),
               "--not-a-claim-of", "our trading edge"])
    assert rc == 0
    prov = json.loads(ckpt.with_suffix(".provenance.json").read_text())
    assert prov["provenance_stub"] is False
    assert "our trading edge" in prov["not_a_claim_of"]


def test_exit_code_map_documents_three_and_five_distinctly():
    """3 is uniquely the compile costing-gate stop; 5 is gate-failed-to-catch."""
    from lens_kit import cli
    doc = cli.__doc__
    assert "3 = costing-gate hard stop (compile ONLY" in doc
    assert "5 = gate failed to catch planted flaws" in doc
    # The compile constant is still 3; calibration moved to 5.
    from lens_kit.compile_harness import EXIT_COST_STOP
    from lens_kit.calibration import EXIT_BATTERY_FAIL
    from lens_kit.mutation import EXIT_RUBBER_STAMP
    assert EXIT_COST_STOP == 3
    assert EXIT_BATTERY_FAIL == 5 == EXIT_RUBBER_STAMP


def test_exit_code_map_documents_six_consistency():
    from lens_kit import cli
    from lens_kit.consistency import EXIT_CONSISTENCY
    assert "6 = consistency violation" in cli.__doc__
    assert EXIT_CONSISTENCY == 6


# ── consistency command (deterministic, no network) ──────────────

def _w(p, text):
    p.write_text(text, encoding="utf-8")
    return p


def test_consistency_requires_subcommand():
    with pytest.raises(SystemExit) as exc:
        main(["consistency"])
    assert exc.value.code == 2


def test_consistency_markers_clean_exits_0(tmp_path):
    src = _w(tmp_path / "src.md", "Cost [ESTIMATE] up.")
    out = _w(tmp_path / "out.md", "Cost [ESTIMATE] up indeed.")
    rc = main(["consistency", "markers", str(src), str(out)])
    assert rc == 0


def test_consistency_markers_dropped_exits_6(tmp_path):
    src = _w(tmp_path / "src.md", "Cost [PROJECTION] up.")
    out = _w(tmp_path / "out.md", "Cost is up.")
    rc = main(["consistency", "markers", str(src), str(out)])
    assert rc == 6


def test_consistency_markers_flag_override(tmp_path):
    src = _w(tmp_path / "src.md", "value [GUESS] here")
    out = _w(tmp_path / "out.md", "value here")
    assert main(["consistency", "markers", str(src), str(out)]) == 0
    assert main(["consistency", "markers", str(src), str(out),
                 "--markers", "[GUESS]"]) == 6


def test_consistency_markers_missing_file_exits_2(tmp_path, capsys):
    src = _w(tmp_path / "src.md", "[ESTIMATE] x")
    rc = main(["consistency", "markers", str(src), str(tmp_path / "absent.md")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_consistency_leaks_clean_exits_0(tmp_path):
    f = _w(tmp_path / "copy.md", "Our platform helps agencies.")
    rc = main(["consistency", "leaks", str(f), "--deny", "gross margin"])
    assert rc == 0


def test_consistency_leaks_hit_exits_6(tmp_path):
    f = _w(tmp_path / "copy.md", "Our GROSS MARGIN leaked.")
    rc = main(["consistency", "leaks", str(f), "--deny", "gross margin"])
    assert rc == 6


def test_consistency_leaks_uses_profile_deny(tmp_path):
    # agency-example profile ships a deny-list containing "lens gate"
    f = _w(tmp_path / "copy.md", "Internally the lens gate scored it.")
    rc = main(["consistency", "leaks", str(f),
               "--profile", str(builtin_profile_path("agency-example"))])
    assert rc == 6


def test_consistency_numbers_clean_exits_0(tmp_path):
    summary = _w(tmp_path / "summary.md", "Revenue $1.2M.")
    body = _w(tmp_path / "body.md", "Revenue reached $1,200,000.")
    rc = main(["consistency", "numbers", str(summary), str(body)])
    assert rc == 0


def test_consistency_numbers_orphan_exits_6(tmp_path):
    summary = _w(tmp_path / "summary.md", "Uptime 99%.")
    body = _w(tmp_path / "body.md", "Uptime is excellent.")
    rc = main(["consistency", "numbers", str(summary), str(body)])
    assert rc == 6


def test_consistency_all_runs_selected_checks(tmp_path):
    import yaml
    src = _w(tmp_path / "src.md", "Cost [ESTIMATE] is $5M.")
    rendered = _w(tmp_path / "out.md", "Cost is $5M.")    # drops [ESTIMATE]
    summary = _w(tmp_path / "summary.md", "Revenue $5M.")
    body = _w(tmp_path / "body.md", "Revenue $5M total.")
    cfg = _w(tmp_path / "cfg.yaml", yaml.safe_dump({
        "markers": {"source": str(src), "rendered": [str(rendered)]},
        "numbers": {"summary": str(summary), "body": str(body)},
    }))
    rc = main(["consistency", "all", "--config", str(cfg)])
    assert rc == 6                          # markers fails, numbers clean -> overall 6


def test_consistency_all_all_clean_exits_0(tmp_path):
    import yaml
    summary = _w(tmp_path / "summary.md", "Revenue $5M.")
    body = _w(tmp_path / "body.md", "Revenue $5M total.")
    cfg = _w(tmp_path / "cfg.yaml", yaml.safe_dump({
        "numbers": {"summary": str(summary), "body": str(body)},
    }))
    assert main(["consistency", "all", "--config", str(cfg)]) == 0


def test_consistency_all_empty_config_exits_2(tmp_path, capsys):
    import yaml
    cfg = _w(tmp_path / "cfg.yaml", yaml.safe_dump({"unknown": {}}))
    rc = main(["consistency", "all", "--config", str(cfg)])
    assert rc == 2
    assert "no checks" in capsys.readouterr().err


def test_consistency_writes_ledger_row(tmp_path):
    f = _w(tmp_path / "copy.md", "clean copy here")
    main(["consistency", "leaks", str(f), "--deny", "absent-term"])
    from pathlib import Path
    ledger = Path.cwd() / "RUNS.md"         # conftest chdirs to tmp cwd
    assert ledger.exists()
    assert "CONSISTENCY_OK" in ledger.read_text()


# ── integrity subcommand (CLI-level, no network) ─────────────────

def test_integrity_missing_ledger_exits_2(tmp_path):
    # conftest chdirs to a neutral tmp cwd; no RUNS.md there.
    rc = main(["integrity", "--runs-md", str(tmp_path / "absent.md")])
    assert rc == 2


def test_integrity_last_must_be_positive(tmp_path):
    rc = main(["integrity", "--runs-md", str(tmp_path / "x.md"), "--last", "0"])
    assert rc == 2


def test_integrity_strict_flags_rubber_stamp_ledger_exit_6(tmp_path):
    from lens_kit.ledger import HEADER
    rows = []
    for i in range(6):
        acc = min(0.95 + i * 0.008, 0.995)
        rows.append(f"| eval-2026060{i+1}-100000 | 2026-06-0{i+1} 10:00 "
                    f"| lens-kit eval holdout.json | - | holdout.json "
                    f"| overall_accuracy_floor={acc:.3f} | EVAL |\n")
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(HEADER + "".join(rows))
    (tmp_path / "catches.jsonl").write_text("")     # zero catches
    rc = main(["integrity", "--runs-md", str(ledger),
               "--catches", str(tmp_path / "catches.jsonl"),
               "--last", "6", "--strict"])
    assert rc == 6


def test_integrity_report_only_default_exit_0(tmp_path):
    from lens_kit.ledger import HEADER
    rows = []
    for i in range(6):
        acc = min(0.95 + i * 0.008, 0.995)
        rows.append(f"| eval-2026060{i+1}-100000 | 2026-06-0{i+1} 10:00 "
                    f"| lens-kit eval holdout.json | - | holdout.json "
                    f"| overall_accuracy_floor={acc:.3f} | EVAL |\n")
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(HEADER + "".join(rows))
    (tmp_path / "catches.jsonl").write_text("")
    rc = main(["integrity", "--runs-md", str(ledger),
               "--catches", str(tmp_path / "catches.jsonl"), "--last", "6"])
    assert rc == 0                                   # report-only: flag does not gate
    assert "INTEGRITY_FLAG" in ledger.read_text()    # but the row records the flag


# ── improve (item 8) ─────────────────────────────────────────────

def _improve_inputs(tmp_path):
    """Minimal valid file inputs for `lens-kit improve`."""
    import json
    rows = [{"text": f"example number {i:02d} long enough to be kept around",
             "per_lens": {"truth": True}, "domain": "general"} for i in range(8)]
    (tmp_path / "train.json").write_text(json.dumps(rows))
    (tmp_path / "holdout.json").write_text(json.dumps(rows[:6]))
    (tmp_path / "baseline.json").write_text(json.dumps({"truth.predict": {}}))
    (tmp_path / "catches.jsonl").write_text("")
    return {
        "trainset": str(tmp_path / "train.json"),
        "holdout": str(tmp_path / "holdout.json"),
        "baseline": str(tmp_path / "baseline.json"),
        "catches": str(tmp_path / "catches.jsonl"),
        "profile": str(builtin_profile_path("agency-example")),
        "out": str(tmp_path / "out"),
    }


def test_improve_missing_required_exits_2(capsys):
    rc = main(["improve"])  # no --status, no inputs
    assert rc == 2
    assert "--trainset" in capsys.readouterr().err


def test_improve_missing_trainset_file_exits_2(tmp_path, capsys):
    inp = _improve_inputs(tmp_path)
    rc = main(["improve", "--trainset", str(tmp_path / "nope.json"),
               "--holdout", inp["holdout"],
               "--baseline-checkpoint", inp["baseline"],
               "--catches", inp["catches"], "--profile", inp["profile"]])
    assert rc == 2
    assert "trainset not found" in capsys.readouterr().err


def test_improve_budget_flags_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        main(["improve", "--trainset", "t", "--holdout", "h",
              "--baseline-checkpoint", "c", "--profile", "p",
              "--max-evals", "4", "--auto", "light"])
    assert exc.value.code == 2


def test_improve_status_empty_ledger(tmp_path, capsys):
    rc = main(["improve", "--status", "--runs-md", str(tmp_path / "RUNS.md")])
    assert rc == 0
    assert "no improve" in capsys.readouterr().out.lower()


def test_improve_promote_end_to_end_stubbed(tmp_path, monkeypatch, capsys):
    """Full main() path with the three improve seams stubbed (no network)."""
    import json

    import lens_kit.improve as imp
    from lens_kit.improve import MeasuredEval

    inp = _improve_inputs(tmp_path)

    def fake_compile(training_path, profile_obj, output_path, **kw):
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps({"truth.predict": {"x": 1}}))
        return 0

    def fake_eval(checkpoint, holdout_path, profile_obj, **kw):
        if checkpoint is None or "baseline" in str(checkpoint):
            return MeasuredEval(summary={"overall_accuracy": 0.9, "catch_rate": 0.7,
                                         "fp_rate": 0.05})
        return MeasuredEval(
            summary={"overall_accuracy": 0.95, "catch_rate": 0.85, "fp_rate": 0.04},
            envelope={"catch_rate": {"worst_of_n": 0.80}}, reruns=3)

    monkeypatch.setattr(imp, "_compile_challenger", fake_compile)
    monkeypatch.setattr(imp, "_eval_checkpoint", fake_eval)
    monkeypatch.setattr(imp, "_mutation_control", lambda *a, **k: 0)

    rc = main(["improve", "--trainset", inp["trainset"], "--holdout", inp["holdout"],
               "--baseline-checkpoint", inp["baseline"], "--catches", inp["catches"],
               "--profile", inp["profile"], "--output", inp["out"], "--reruns", "3"])
    assert rc == 0
    from pathlib import Path
    assert (Path(inp["out"]) / "promoted.json").exists()
    assert "PROMOTED" in capsys.readouterr().out


def test_improve_single_measurement_needs_optin_via_cli(tmp_path, monkeypatch, capsys):
    """Through main(): a single-measurement (no-envelope) win is blocked without
    --allow-single-measurement, and promotes with it."""
    import json

    import lens_kit.improve as imp
    from lens_kit.improve import MeasuredEval

    inp = _improve_inputs(tmp_path)

    def fake_compile(training_path, profile_obj, output_path, **kw):
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps({"truth.predict": {"x": 1}}))
        return 0

    def fake_eval(checkpoint, holdout_path, profile_obj, **kw):
        if checkpoint is None or "baseline" in str(checkpoint):
            return MeasuredEval(summary={"overall_accuracy": 0.9, "catch_rate": 0.7,
                                         "fp_rate": 0.05})
        # No envelope -> single measurement.
        return MeasuredEval(summary={"overall_accuracy": 0.95, "catch_rate": 0.85,
                                     "fp_rate": 0.04})

    monkeypatch.setattr(imp, "_compile_challenger", fake_compile)
    monkeypatch.setattr(imp, "_eval_checkpoint", fake_eval)
    monkeypatch.setattr(imp, "_mutation_control", lambda *a, **k: 0)

    from pathlib import Path
    base = ["improve", "--trainset", inp["trainset"], "--holdout", inp["holdout"],
            "--baseline-checkpoint", inp["baseline"], "--catches", inp["catches"],
            "--profile", inp["profile"]]

    out1 = str(tmp_path / "out1")
    rc = main(base + ["--output", out1])  # no opt-in
    assert rc == 0
    assert not (Path(out1) / "promoted.json").exists()
    assert "--allow-single-measurement" in capsys.readouterr().out

    out2 = str(tmp_path / "out2")
    rc = main(base + ["--output", out2, "--allow-single-measurement"])
    assert rc == 0
    assert (Path(out2) / "promoted.json").exists()
