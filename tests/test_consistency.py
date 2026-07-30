"""Deterministic consistency checks: marker parity, leaks, number parity.

No network (deterministic, no LLM). Each check has a PLANTED-DEFECT fixture
that FAILS and a CLEAN fixture that PASSES (success criterion 1), plus
unicode/currency/percent edges, marker-set override, deny-list
case-insensitivity, exit codes, and the ledger row.
"""
import json
from pathlib import Path

import pytest

from lens_kit.consistency import (DEFAULT_MARKERS, EXIT_CLEAN,
                                  EXIT_CONSISTENCY, check_leaks, check_markers,
                                  check_numbers, count_marker, extract_numbers,
                                  normalize_number, run_consistency,
                                  scan_leaks)


def _w(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ── 1. Marker parity ─────────────────────────────────────────────

def test_markers_clean_rendered_carries_all_markers_passes(tmp_path):
    src = _w(tmp_path / "src.md",
             "Revenue [ESTIMATE] is up. Growth [PROJECTION] next year.")
    rendered = _w(tmp_path / "out.md",
                  "Revenue [ESTIMATE] is up; growth [PROJECTION] next year.")
    result = check_markers(src, [rendered])
    assert result.violated is False
    assert result.violations == []
    assert result.exit_code == EXIT_CLEAN


def test_markers_dropped_in_render_fails(tmp_path):
    """Marker-loss class: a marker present in source vanishes in render."""
    src = _w(tmp_path / "src.md",
             "Cost [ESTIMATE] and uptime [PROJECTION] and risk [UNVERIFIED].")
    # render keeps ESTIMATE but DROPS the PROJECTION and UNVERIFIED markers
    rendered = _w(tmp_path / "summary.md",
                  "Cost [ESTIMATE], uptime is 99.9%, risk is low.")
    result = check_markers(src, [rendered])
    assert result.violated is True
    assert result.exit_code == EXIT_CONSISTENCY
    joined = "\n".join(result.violations)
    assert "[PROJECTION]" in joined
    assert "[UNVERIFIED]" in joined
    assert "summary.md" in joined


def test_markers_counts_under_source_flagged(tmp_path):
    src = _w(tmp_path / "src.md", "[ESTIMATE] a [ESTIMATE] b [ESTIMATE] c")
    rendered = _w(tmp_path / "out.md", "[ESTIMATE] only one here")
    result = check_markers(src, [rendered])
    assert result.violated is True
    assert "count 1 < source 3" in result.violations[0]


def test_markers_equal_or_greater_passes(tmp_path):
    src = _w(tmp_path / "src.md", "[ESTIMATE] once")
    rendered = _w(tmp_path / "out.md", "[ESTIMATE] and [ESTIMATE] twice is fine")
    assert check_markers(src, [rendered]).violated is False


def test_markers_override_set(tmp_path):
    src = _w(tmp_path / "src.md", "value [GUESS] here, also [ESTIMATE] there")
    rendered = _w(tmp_path / "out.md", "value here, also [ESTIMATE] there")
    # default markers: [GUESS] not tracked -> clean
    assert check_markers(src, [rendered]).violated is False
    # override to track [GUESS] -> the dropped [GUESS] now fails
    result = check_markers(src, [rendered], markers=["[GUESS]"])
    assert result.violated is True
    assert "[GUESS]" in result.violations[0]


def test_markers_multiple_rendered_files(tmp_path):
    src = _w(tmp_path / "src.md", "[UNVERIFIED] claim")
    good = _w(tmp_path / "good.md", "[UNVERIFIED] claim preserved")
    bad = _w(tmp_path / "bad.md", "claim with marker stripped")
    result = check_markers(src, [good, bad])
    assert result.violated is True
    assert len(result.violations) == 1
    assert "bad.md" in result.violations[0]
    assert "good.md" not in result.violations[0]


def test_count_marker_is_literal_and_case_sensitive():
    assert count_marker("[ESTIMATE] x [ESTIMATE]", "[ESTIMATE]") == 2
    assert count_marker("[estimate]", "[ESTIMATE]") == 0   # markers are case-sensitive tokens
    assert count_marker("anything", "") == 0


# ── 2. Forbidden-strings leak scan ───────────────────────────────

def test_leaks_clean_file_passes(tmp_path):
    f = _w(tmp_path / "copy.md", "Our platform helps agencies move faster.")
    result = check_leaks([f], deny=["gross margin", "lens gate"])
    assert result.violated is False
    assert result.exit_code == EXIT_CLEAN


def test_leaks_planted_internal_term_fails(tmp_path):
    """Vocabulary-leak class: internal vocabulary reaches customer-facing copy."""
    f = _w(tmp_path / "copy.md",
           "Welcome to our platform.\nInternally the lens gate scored this 9/9.\nThanks!")
    result = check_leaks([f], deny=["lens gate"])
    assert result.violated is True
    assert result.exit_code == EXIT_CONSISTENCY
    assert "copy.md:2" in result.violations[0]
    assert "lens gate" in result.violations[0]


def test_leaks_case_insensitive(tmp_path):
    f = _w(tmp_path / "copy.md", "Our GROSS MARGIN is healthy.")
    result = check_leaks([f], deny=["gross margin"])
    assert result.violated is True
    assert "copy.md:1" in result.violations[0]


def test_leaks_literal_not_regex(tmp_path):
    """Deny terms are literal: regex metacharacters match themselves, not patterns."""
    f = _w(tmp_path / "copy.md", "price is $9.99 today")
    # '.' in the deny term must match a literal dot, not 'any char'
    assert check_leaks([f], deny=["9x99"]).violated is False    # would match if '.'=anychar
    assert check_leaks([f], deny=["9.99"]).violated is True


def test_leaks_per_hit_line_refs(tmp_path):
    f = _w(tmp_path / "copy.md", "line one\nCOGS leaked here\nline three\nCOGS again")
    result = check_leaks([f], deny=["cogs"])
    assert len(result.violations) == 2
    assert "copy.md:2" in result.violations[0]
    assert "copy.md:4" in result.violations[1]


def test_leaks_empty_deny_is_clean_not_error(tmp_path):
    f = _w(tmp_path / "copy.md", "anything at all")
    result = check_leaks([f], deny=[])
    assert result.violated is False
    assert "no deny-list" in result.summary


def test_scan_leaks_returns_line_and_term():
    hits = scan_leaks("clean\nDENY-ME present\nclean", ["deny-me"])
    assert hits == [(2, "DENY-ME present", "deny-me")]


# ── 3. Number parity ─────────────────────────────────────────────

def test_numbers_clean_all_present_in_body_passes(tmp_path):
    summary = _w(tmp_path / "summary.md", "Revenue $1.2M, growth 76%, 250 clients.")
    body = _w(tmp_path / "body.md",
              "We reached $1,200,000 in revenue.\nGrowth was 76% YoY.\n"
              "We now serve 250 clients across regions.")
    result = check_numbers(summary, body)
    assert result.violated is False
    assert result.exit_code == EXIT_CLEAN


def test_numbers_orphan_in_summary_fails(tmp_path):
    """Summary-invention class: a summary asserts a number the body never states."""
    summary = _w(tmp_path / "summary.md", "Revenue $5M and 99% uptime.")
    body = _w(tmp_path / "body.md", "Revenue was $5M last year. Uptime is strong.")
    result = check_numbers(summary, body)
    assert result.violated is True
    assert result.exit_code == EXIT_CONSISTENCY
    assert "99%" in result.violations[0]
    assert "summary.md:1" in result.violations[0]


def test_numbers_currency_and_thousands_normalize(tmp_path):
    summary = _w(tmp_path / "summary.md", "Budget £1,200,000 this year.")
    body = _w(tmp_path / "body.md", "The budget came to £1.2M after review.")
    assert check_numbers(summary, body).violated is False


def test_numbers_percent_kept_distinct_from_bare(tmp_path):
    """'76%' must not be satisfied by a bare '76' in the body (and vice versa)."""
    summary = _w(tmp_path / "summary.md", "Margin 76%.")
    body = _w(tmp_path / "body.md", "We have 76 customers.")
    result = check_numbers(summary, body)
    assert result.violated is True
    assert "76%" in result.violations[0]


def test_numbers_unicode_currency_symbols(tmp_path):
    summary = _w(tmp_path / "summary.md", "Costs €5,500 and ¥10000.")
    body = _w(tmp_path / "body.md", "Spent 5,500 euros and 10000 yen total.")
    # €5,500 -> 5500 ; ¥10000 -> 10000 ; both present (5,500 -> 5500, 10000)
    assert check_numbers(summary, body).violated is False


def test_numbers_derived_value_false_positive_is_documented(tmp_path):
    """DOCUMENTED boundary: '13 of 17' is NOT resolved to '76%' (false positive)."""
    summary = _w(tmp_path / "summary.md", "We passed 76% of checks.")
    body = _w(tmp_path / "body.md", "We passed 13 of 17 checks.")
    result = check_numbers(summary, body)
    # 76% has no literal/normalized match -> flagged, as the boundary promises
    assert result.violated is True
    assert "76%" in result.violations[0]


def test_numbers_orphan_deduplicated(tmp_path):
    summary = _w(tmp_path / "summary.md", "Up 99% here and 99% there.")
    body = _w(tmp_path / "body.md", "No matching figure in the body.")
    result = check_numbers(summary, body)
    assert len(result.violations) == 1     # 99% reported once, not twice


def test_normalize_number_table():
    assert normalize_number("$1,200,000") == "1200000"
    assert normalize_number("1.2", "M") == "1200000"
    assert normalize_number("76", "%") == "76%"
    assert normalize_number("1 200 000") == "1200000"
    assert normalize_number("4.20") == "4.2"
    assert normalize_number("3", "x") == "3x"
    assert normalize_number("€5,500") == "5500"


def test_extract_numbers_handles_million_word_not_as_suffix():
    nums = extract_numbers("We raised 5 million dollars")
    # '5 million' -> '5' (the word 'million' is not the 'm' suffix)
    assert ("5", "5") in [(d, n) for _, d, n in nums]


# ── run_consistency: exit codes + ledger ─────────────────────────

def test_run_consistency_clean_exits_0_and_writes_ok_ledger(tmp_path):
    summary = _w(tmp_path / "summary.md", "Revenue $5M.")
    body = _w(tmp_path / "body.md", "Revenue was $5M.")
    result = check_numbers(summary, body)
    rc = run_consistency([result], command_str="lens-kit consistency numbers")
    assert rc == EXIT_CLEAN
    ledger = Path.cwd() / "RUNS.md"        # conftest chdirs to a tmp cwd
    assert ledger.exists()
    text = ledger.read_text()
    assert "CONSISTENCY_OK" in text
    assert "numbers=ok" in text


def test_run_consistency_violation_exits_6_and_writes_fail_ledger(tmp_path):
    f = _w(tmp_path / "copy.md", "the lens gate scored it")
    result = check_leaks([f], deny=["lens gate"])
    rc = run_consistency([result], command_str="lens-kit consistency leaks")
    assert rc == EXIT_CONSISTENCY
    text = (Path.cwd() / "RUNS.md").read_text()
    assert "CONSISTENCY_FAIL" in text
    assert "leaks=FAIL" in text


def test_run_consistency_any_violation_trips_exit_6(tmp_path):
    clean = check_numbers(_w(tmp_path / "s.md", "$5M"),
                          _w(tmp_path / "b.md", "$5M total"))
    dirty = check_leaks([_w(tmp_path / "c.md", "COGS here")], deny=["cogs"])
    assert run_consistency([clean, dirty], write_ledger=False) == EXIT_CONSISTENCY


def test_run_consistency_no_score_overrides_a_leak(tmp_path):
    """Recovery 1.10: a leak is a violation regardless of anything else clean."""
    clean = check_markers(_w(tmp_path / "s.md", "[ESTIMATE] x"),
                          [_w(tmp_path / "r.md", "[ESTIMATE] x preserved")])
    leak = check_leaks([_w(tmp_path / "c.md", "internal cost is X")],
                       deny=["internal cost"])
    assert clean.violated is False
    assert run_consistency([clean, leak], write_ledger=False) == EXIT_CONSISTENCY
