"""RUNS.md run ledger: header creation, append-only rows, escaping."""
from pathlib import Path

from lens_kit.ledger import HEADER, LEDGER_FILENAME, append_run, new_run_id


def test_first_append_creates_header(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    append_run(run_id="compile-20260612-101010", command="lens-kit compile d.json",
               checkpoint_sha="a" * 64, data_file="d.json",
               metrics="val_score=0.81", status="KEEP", ledger_path=path)
    text = path.read_text()
    assert text.startswith("# RUNS — lens-kit run ledger")
    # The run-(re)designation rule line is part of the contract.
    assert "NEW run designation" in text
    assert "| Run id | Date | Command | Checkpoint sha | Labels/data file | Key metrics | Status |" in text
    assert "| compile-20260612-101010 |" in text
    assert "| " + "a" * 12 + " |" in text          # sha truncated to 12
    assert "| KEEP |" in text


def test_second_append_does_not_duplicate_header(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    for status in ("KEEP", "DISCARD"):
        append_run(run_id=f"r-{status}", command="c", checkpoint_sha=None,
                   data_file="d.json", metrics="m", status=status,
                   ledger_path=path)
    text = path.read_text()
    assert text.count("# RUNS") == 1
    assert "| DISCARD |" in text and "| KEEP |" in text
    assert text.count("| r-") == 2


def test_defaults_to_cwd_runs_md(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    append_run(run_id="r1", command="c", checkpoint_sha=None,
               data_file=None, metrics="m", status="EVAL")
    assert (tmp_path / LEDGER_FILENAME).exists()


def test_cells_escape_pipes_and_newlines(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    append_run(run_id="r1", command="cmd | with pipe", checkpoint_sha=None,
               data_file="d.json", metrics="a=1\nb=2", status="KEEP",
               ledger_path=path)
    row = path.read_text().splitlines()[-1]
    assert "\\|" in row
    assert row.count("|") - row.count("\\|") == 8   # 7 cells -> 8 bare pipes


def test_new_run_id_carries_command_prefix():
    assert new_run_id("compile").startswith("compile-")
