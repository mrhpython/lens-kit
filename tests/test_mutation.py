"""Mutation control: mutant determinism, every type detectable by construction,
and the rubber-stamp detection path. All no-network.

A FakeGate returns LensResult directly (no LLM) so a test can drive a gate that
catches all mutants (alive) or passes them (rubber-stamping). The mutant
constructors are pure-Python and asserted byte-deterministic.
"""
import json

import pytest

from lens_kit.config import Profile
from lens_kit.gate import LensResult, LensViolation
from lens_kit.mutation import (CAUGHT_VERDICTS, MUTATION_TYPES, EXIT_RUBBER_STAMP,
                               generate_mutants, mutate_text, run_mutant,
                               run_mutation_control)


def _profile():
    return Profile.from_dict({"name": "t", "llm": {"model": "openai/gpt-4o"}})


def _holdout(tmp_path, n=4):
    """A small clean holdout with attributed, citable text (so stripping works)."""
    examples = [
        {"text": f"According to the May analytics dashboard, sessions rose to "
                 f"{4000 + i} this month, as reported by the platform's own report. "
                 f"We recommend keeping the current schedule for example {i}.",
         "source_id": f"ex-{i}", "domain": "general"}
        for i in range(n)
    ]
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps({"examples": examples}))
    return path


# ── Mutant determinism ───────────────────────────────────────────

def test_mutants_are_byte_identical_for_same_seed(tmp_path):
    h = _holdout(tmp_path)
    a = generate_mutants(h, seed=5)
    b = generate_mutants(h, seed=5)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_mutant_text_differs_from_source(tmp_path):
    h = _holdout(tmp_path)
    src = json.loads(h.read_text())["examples"]
    for m in generate_mutants(h, seed=0):
        original = src[m["source_index"]]["text"]
        assert m["text"] != original, f"{m['id']} did not mutate the text"


def test_one_mutant_per_type_by_default(tmp_path):
    mutants = generate_mutants(_holdout(tmp_path), seed=0)
    types = sorted(m["mutation_type"] for m in mutants)
    assert types == sorted(MUTATION_TYPES)


def test_per_type_controls_count(tmp_path):
    mutants = generate_mutants(_holdout(tmp_path, n=6), seed=0, per_type=2)
    assert len(mutants) == 2 * len(MUTATION_TYPES)


def test_empty_holdout_is_a_hard_error(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"examples": [{"text": "short"}]}))  # < 20 chars
    with pytest.raises(ValueError, match="No usable examples"):
        generate_mutants(path, seed=0)


# ── Every mutation type detectable by construction ───────────────

def test_every_type_plants_a_recoverable_defect(tmp_path):
    """Each mutation type produces text whose marker names the planted defect,
    and the defect class maps to a non-SHIP caught-verdict set."""
    for m in generate_mutants(_holdout(tmp_path), seed=1):
        assert m["marker"]                                   # defect described
        assert set(m["caught_verdicts"]) == CAUGHT_VERDICTS[m["mutation_type"]]
        # No caught-verdict set permits a silent SHIP.
        assert "SHIP" not in m["caught_verdicts"]


def test_fabricated_stat_appends_uncited_number(tmp_path):
    import random
    text = "We help teams review AI output before it ships, by hand today."
    mutated, marker = mutate_text(text, "fabricated_stat", random.Random("0:x"))
    assert len(mutated) > len(text)
    assert any(c.isdigit() for c in mutated[len(text):])     # a number was added
    assert "uncited" in marker


def test_stripped_source_removes_attribution(tmp_path):
    import random
    text = ("According to the dashboard, sessions rose. As reported by Analytics, "
            "clicks doubled. We recommend continuing.")
    mutated, marker = mutate_text(text, "stripped_source", random.Random("0:x"))
    assert "according to" not in mutated.lower()
    assert "as reported by" not in mutated.lower()
    assert "stripped" in marker


def test_unknown_mutation_type_raises():
    import random
    with pytest.raises(ValueError, match="Unknown mutation type"):
        mutate_text("text", "not_a_type", random.Random())


# ── run_mutant classification ────────────────────────────────────

class _CatchingGate:
    """Returns a blocking LensResult per mutation marker -> catches everything."""
    def __call__(self, text, domain="general", context=""):
        # Any non-clean text -> a HOLD-mapping result (contradiction blocking).
        return LensResult(passed=False, per_lens={"contradiction": False},
                          violations=[LensViolation(lens="contradiction",
                                                    severity="high", issue="x")])


class _RubberStampGate:
    """Passes everything -> SHIP -> misses every mutant."""
    def __call__(self, text, domain="general", context=""):
        return LensResult(passed=True, per_lens={"truth": True, "rights": True})


class _CrashGate:
    def __call__(self, text, domain="general", context=""):
        raise RuntimeError("gate exploded")


def test_run_mutant_caught_when_gate_blocks(tmp_path):
    m = generate_mutants(_holdout(tmp_path), seed=0)[0]
    row = run_mutant(m, _CatchingGate())
    assert row["caught"] is True
    assert row["verdict"] == "HOLD"


def test_run_mutant_missed_when_gate_ships(tmp_path):
    m = generate_mutants(_holdout(tmp_path), seed=0)[0]
    row = run_mutant(m, _RubberStampGate())
    assert row["caught"] is False
    assert row["verdict"] == "SHIP"


def test_run_mutant_crash_is_a_miss(tmp_path):
    m = generate_mutants(_holdout(tmp_path), seed=0)[0]
    row = run_mutant(m, _CrashGate())
    assert row["caught"] is False
    assert row["verdict"] == "ERROR"
    assert "gate exploded" in row["error"]


# ── End-to-end: alive gate exits 0, rubber-stamp exits 5 + names misses ──

def test_alive_gate_exits_zero(tmp_path, capsys):
    rc = run_mutation_control(_holdout(tmp_path), _profile(),
                              output_dir=tmp_path / "out", gate=_CatchingGate())
    assert rc == 0
    assert "All" in capsys.readouterr().out


def test_rubber_stamp_exits_five_and_names_missed(tmp_path, capsys):
    rc = run_mutation_control(_holdout(tmp_path), _profile(),
                              output_dir=tmp_path / "out", gate=_RubberStampGate())
    assert rc == EXIT_RUBBER_STAMP == 5
    out = capsys.readouterr().out
    assert "RUBBER-STAMP WARNING" in out
    # Every mutant id is named in the warning.
    for m in generate_mutants(_holdout(tmp_path), seed=0):
        assert m["id"] in out


def test_run_writes_results_and_ledger(tmp_path):
    out_dir = tmp_path / "out"
    run_mutation_control(_holdout(tmp_path), _profile(),
                         output_dir=out_dir, gate=_CatchingGate())
    runs = sorted(out_dir.glob("mutation-*.json"))
    assert len(runs) == 1
    doc = json.loads(runs[0].read_text())
    assert doc["mutation_schema_version"] == 1
    assert doc["total"] == len(MUTATION_TYPES) and doc["missed"] == 0
    from pathlib import Path
    assert "MUTATE" in (Path.cwd() / "RUNS.md").read_text()


def test_ledger_marks_rubber_stamp(tmp_path):
    run_mutation_control(_holdout(tmp_path), _profile(), gate=_RubberStampGate())
    from pathlib import Path
    assert "MUTATE (rubber-stamp)" in (Path.cwd() / "RUNS.md").read_text()
