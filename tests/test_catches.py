"""catches.jsonl institutional-memory loop — no network, no LLM.

Covers: add/append/round-trip, doctrine rejection (empty + routine-pass
heuristic), warn-on-missing-pattern, relevant filtering + ordering + the
paste-block format STABILITY, stats recurrence + threshold promotion line,
the --seed copy, the deny-list grep on the shipped seed (success criterion 2),
and the ledger redaction (S1 review WARN).

The conftest autouse fixture chdirs every test to a neutral tmp cwd, so the
default catches.jsonl / RUNS.md land in an isolated dir per test.
"""
import json
from pathlib import Path

import pytest

from lens_kit import catches as cat
from lens_kit.catches import (CATCHES_FILENAME, DEFAULT_PROMOTE_THRESHOLD,
                             SCHEMA_VERSION, CatchError, append_catch,
                             build_catch, catches_path, filter_records,
                             is_pass_like, load_catches, load_seed,
                             order_for_relevant, pattern_counts, promotable,
                             render_relevant_block, render_stats, seed_path)
from lens_kit.cli import main
from lens_kit.ledger import redact_command


# ── add / append / round-trip ────────────────────────────────────

def test_build_and_append_round_trip(tmp_path):
    path = tmp_path / CATCHES_FILENAME
    catch, warnings = build_catch(
        catch="dropped the [ESTIMATE] marker in the summary table",
        pattern="markers get lost when prose is summarized into a table",
        rule="diff marker counts source vs every render",
        domain="marketing", artifact_type="content-pack", date="2026-03-01")
    assert warnings == []
    append_catch(catch, path)
    records = load_catches(path)
    assert len(records) == 1
    r = records[0]
    assert r["catch"].startswith("dropped the [ESTIMATE]")
    assert r["domain"] == "marketing"
    assert r["artifact_type"] == "content-pack"
    assert r["self_catch"] is False
    assert r["schema_version"] == SCHEMA_VERSION
    assert r["date"] == "2026-03-01"
    # id is derived, stable, readable
    assert r["id"].startswith("2026-03-01-content-pack-")


def test_append_is_jsonl_one_object_per_line(tmp_path):
    path = tmp_path / CATCHES_FILENAME
    for i in range(3):
        c, _ = build_catch(catch=f"defect number {i}", pattern=f"p{i}",
                           rule="r", artifact_type="x")
        append_catch(c, path)
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    for l in lines:
        assert isinstance(json.loads(l), dict)   # each line parses alone


def test_default_date_is_today_when_absent():
    c, _ = build_catch(catch="a real defect", pattern="p", rule="r")
    import datetime
    assert c.date == datetime.date.today().isoformat()


def test_load_missing_file_is_empty(tmp_path):
    assert load_catches(tmp_path / "nope.jsonl") == []


def test_load_skips_blank_lines(tmp_path):
    path = tmp_path / CATCHES_FILENAME
    path.write_text('{"catch": "x", "pattern": "p"}\n\n\n', encoding="utf-8")
    assert len(load_catches(path)) == 1


def test_load_malformed_line_raises_with_lineno(tmp_path):
    path = tmp_path / CATCHES_FILENAME
    path.write_text('{"catch": "ok"}\nNOT JSON\n', encoding="utf-8")
    with pytest.raises(CatchError) as e:
        load_catches(path)
    assert ":2:" in str(e.value)


# ── doctrine: reject routine passes, warn on missing generalization ──

@pytest.mark.parametrize("passlike", ["", "  ", "pass", "PASS", "passed",
                                      "clean", "ok", "Looks good.", "LGTM",
                                      "no issues", "no violations", "n/a",
                                      "none", "-", "fine",
                                      # multi-word routine passes (closed-vocab rule)
                                      "all lenses passed", "no issues found",
                                      "all checks passed", "no problems found",
                                      "everything checks out", "verified clean",
                                      "all clear", "passed all 9 lenses",
                                      "nothing to flag", "zero violations",
                                      "all 9 lenses passed the gate", "ok?"])
def test_pass_like_catches_rejected(passlike):
    assert is_pass_like(passlike) is True
    with pytest.raises(CatchError) as e:
        build_catch(catch=passlike, pattern="p", rule="r")
    assert "NAME A DEFECT" in str(e.value)


@pytest.mark.parametrize("defect", [
    "the table dropped a marker",
    # mentions of passing INSIDE a defect description must survive:
    "the gate PASSED a fabricated stat",
    "all lenses passed but the deny scan was skipped",
    "passed stale data downstream",
    # short vague defects with a content word must survive:
    "no rollback plan exists",
    "the flags are wrong",
])
def test_real_defect_is_not_pass_like(defect):
    assert is_pass_like(defect) is False
    c, _ = build_catch(catch=defect, pattern="p", rule="r")
    assert c.catch == defect


def test_warn_on_missing_pattern_and_rule():
    _, warnings = build_catch(catch="a genuine defect", artifact_type="x")
    joined = " ".join(warnings)
    assert "no pattern" in joined
    assert "no rule" in joined


def test_no_warning_when_pattern_and_rule_present():
    _, warnings = build_catch(catch="a genuine defect", pattern="p", rule="r")
    assert warnings == []


# ── recurrence + promotion ───────────────────────────────────────

def _rec(catch, pattern, **kw):
    c, _ = build_catch(catch=catch, pattern=pattern, rule=kw.pop("rule", "r"), **kw)
    return c.to_record()


def test_pattern_counts_orders_most_recurrent_first():
    records = [_rec("bug a", "P common", artifact_type="x"),
               _rec("bug b", "P common", artifact_type="x"),
               _rec("bug c", "P rare", artifact_type="x")]
    counts, display = pattern_counts(records)
    first = next(iter(counts))
    assert counts[first] == 2
    assert display[first] == "P common"


def test_pattern_counts_case_and_whitespace_normalized():
    records = [_rec("bug a", "Same  Pattern"), _rec("bug b", "same pattern")]
    counts, _ = pattern_counts(records)
    assert len(counts) == 1
    assert next(iter(counts.values())) == 2


def test_records_without_pattern_excluded_from_recurrence():
    records = [{"catch": "x", "pattern": ""}, {"catch": "y", "pattern": "  "}]
    counts, _ = pattern_counts(records)
    assert counts == {}


def test_promotable_at_threshold():
    records = [_rec(f"c{i}", "recurring trap") for i in range(3)]
    assert promotable(records, threshold=3) == [("recurring trap", 3)]
    assert promotable(records, threshold=4) == []


def test_stats_shows_promote_line_at_threshold():
    records = [_rec(f"c{i}", "recurring trap") for i in range(3)]
    out = render_stats(records, threshold=3)
    assert "3x" in out
    assert "PROMOTE: this pattern has recurred 3x" in out
    assert "extension point: consistency.py" in out


def test_stats_no_promote_below_threshold():
    records = [_rec("bug a", "p"), _rec("bug b", "p")]
    out = render_stats(records, threshold=3)
    assert "PROMOTE" not in out
    assert "2x" in out


def test_stats_empty_is_clear():
    out = render_stats([], threshold=3)
    assert "0 catch(es)" in out


# ── relevant: filtering, ordering, paste-block stability ─────────

def test_filter_by_artifact_type():
    records = [_rec("bug a", "p1", artifact_type="landing-copy"),
               _rec("bug b", "p2", artifact_type="research-brief")]
    out = filter_records(records, artifact_type="landing-copy")
    assert len(out) == 1 and out[0]["artifact_type"] == "landing-copy"


def test_filter_by_domain():
    records = [_rec("bug a", "p1", domain="marketing"),
               _rec("bug b", "p2", domain="finance")]
    out = filter_records(records, domain="finance")
    assert len(out) == 1 and out[0]["domain"] == "finance"


def test_filter_artifact_type_is_case_insensitive():
    records = [_rec("bug a", "p1", artifact_type="Landing-Copy")]
    assert len(filter_records(records, artifact_type="landing-copy")) == 1


def test_order_for_relevant_most_recurrent_first():
    records = [_rec("rare", "P rare", artifact_type="x"),
               _rec("common1", "P common", artifact_type="x"),
               _rec("common2", "P common", artifact_type="x")]
    ordered = order_for_relevant(records)
    # the two P-common records (count 2) come before the rare one (count 1)
    assert ordered[0]["pattern"] == "P common"
    assert ordered[1]["pattern"] == "P common"
    assert ordered[2]["pattern"] == "P rare"


def test_relevant_block_format_is_stable():
    """The paste block is a documented contract — assert it byte-for-byte."""
    records = [_rec("a marker was dropped", "markers get lost in tables",
                    artifact_type="content-pack", rule="diff the counts")]
    block = render_relevant_block(records, artifact_type="content-pack")
    expected = (
        "## PRIOR CATCHES (institutional memory)\n"
        "scope: artifact_type=content-pack, domain=any; 1 catch(es)\n"
        "Read these before validating. Most-recurrent patterns first. A "
        "pattern marked PROMOTE has recurred enough to deserve a deterministic "
        "check.\n"
        "\n"
        "- CATCH: a marker was dropped\n"
        "  PATTERN: markers get lost in tables\n"
        "  RULE: diff the counts\n"
    )
    assert block == expected


def test_relevant_block_marks_self_catch():
    records = [_rec("anchored to a peer hash", "social hash anchoring",
                    self_catch=True, artifact_type="verdict")]
    block = render_relevant_block(records, artifact_type="verdict")
    assert "  (self-catch)" in block


def test_relevant_block_surfaces_promote_at_top():
    records = [_rec(f"c{i}", "recurring trap", artifact_type="x") for i in range(3)]
    block = render_relevant_block(records, artifact_type="x", threshold=3)
    body_idx = block.index("- CATCH:")
    promote_idx = block.index("[PROMOTE]")
    assert promote_idx < body_idx           # PROMOTE lines come before records
    assert "(3x)" in block


def test_relevant_block_empty_scope_message():
    block = render_relevant_block([], artifact_type="nothing-here")
    assert "(no prior catches recorded for this scope)" in block
    assert "0 catch(es)" in block


def test_relevant_block_none_for_missing_pattern_and_rule():
    records = [{"catch": "bare defect", "pattern": "", "rule": "",
                "self_catch": False}]
    block = render_relevant_block(records, artifact_type=None)
    assert "PATTERN: (none)" in block
    assert "RULE: (none)" in block


# ── seed: copy + the deny-list grep (success criterion 2) ────────

def test_seed_loads_six_records():
    seed = load_seed()
    assert len(seed) == 6
    classes = {r["id"] for r in seed}
    assert classes == {"seed-marker-loss", "seed-leak", "seed-allocation-mismatch",
                       "seed-arithmetic", "seed-self-catch", "seed-fabricated-anecdote"}
    # at least one self-catch is represented
    assert any(r["self_catch"] for r in seed)


def test_seed_records_are_well_formed_against_doctrine():
    for r in load_seed():
        # every seed record is a real defect (not pass-like) with a pattern+rule
        c, warnings = build_catch(
            catch=r["catch"], pattern=r["pattern"], rule=r["rule"],
            domain=r["domain"], artifact_type=r["artifact_type"],
            date=r["date"], self_catch=r["self_catch"], catch_id=r["id"])
        assert warnings == []
        assert not is_pass_like(c.catch)


# Success criterion 2: the shipped seed contains ZERO real third-party names
# and ZERO internal paths. This deny-list is the contract; if a future edit
# reintroduces any of these, this test fails the build.
#
# The terms are stored as salted hashes, not plaintext. That is deliberate and it
# matters: a plaintext deny-list of the real names you scrubbed IS the leak it was
# written to prevent — anyone reading the test learns exactly which third parties
# and individuals appear in the private catch history. Hashes keep the guard fully
# effective (the seed is tokenised and each token hashed) while publishing nothing.
#
# To add a term:  python -c "import hashlib;print(hashlib.sha256((_SALT+'term').encode()).hexdigest()[:16])"
# using the salt below. Keep any new term out of the source in plaintext.
_SALT = "lens-kit/seed-deny/v1:"

SEED_DENY_HASHES = {
    "02dd5afcdb5a859b", "3d640f4fb14d0572", "40682916459b7ed9",
    "d37aede5302cfa29", "985f3bc99d80e5e1", "4884605957b65305",
    "0d14aee38e2efc77", "af1e96c444b3f0d2", "db37cfd4f7010c6d",
    "a833ec27b98ca170", "dc8bea643cc883c7", "04e2f7a49538fa9c",
    "79d117dc67b7ede3", "da470cbf8a7aeb7c", "1be6ea11f1be01ce",
    "849d51b4ff731ef4", "f65cc8655ef3b548",
}


def _seed_tokens(text: str) -> set[str]:
    """Tokenise the seed the way a leaked name would appear: words and path fragments."""
    import re
    return {t for t in re.split(r"[^a-z0-9._/~-]+", text.lower()) if t}


def test_seed_has_no_real_names_or_internal_paths():
    import hashlib

    text = seed_path().read_text(encoding="utf-8").lower()
    tokens = _seed_tokens(text)
    # Hash every token and every substring-joined pair, so multi-word and
    # path-shaped terms are still caught without storing them in the clear.
    candidates = set(tokens)
    for tok in tokens:
        for frag in (tok.strip("./~-"), tok.split("/")[0], tok.split(".")[0]):
            if frag:
                candidates.add(frag)
    hits = sorted(
        h for h in (
            hashlib.sha256((_SALT + c).encode()).hexdigest()[:16] for c in candidates
        ) if h in SEED_DENY_HASHES
    )
    assert hits == [], (
        f"seed leaked {len(hits)} deny-listed term(s); hash prefix(es): {hits}. "
        "Look up the offending term locally — it is intentionally not printed."
    )


# ── CLI: add (flags / --from-json / --seed), relevant, stats ─────

def test_cli_add_then_relevant_then_stats_round_trip(capsys):
    # add (cwd is the conftest tmp dir)
    rc = main(["catches", "add",
               "--catch", "table dropped the [ESTIMATE] marker",
               "--pattern", "markers lost when summarizing into tables",
               "--rule", "diff counts source vs render",
               "--artifact-type", "content-pack", "--domain", "marketing"])
    assert rc == 0
    assert catches_path().exists()
    # relevant surfaces it
    capsys.readouterr()
    rc = main(["catches", "relevant", "content-pack"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "table dropped the [ESTIMATE] marker" in out
    assert "markers lost when summarizing into tables" in out
    # stats counts it
    rc = main(["catches", "stats"])
    assert rc == 0
    assert "1 catch(es)" in capsys.readouterr().out


def test_cli_add_pass_like_is_usage_error(capsys):
    rc = main(["catches", "add", "--catch", "looks good"])
    assert rc == 2
    assert "NAME A DEFECT" in capsys.readouterr().err


def test_cli_add_missing_catch_is_usage_error(capsys):
    rc = main(["catches", "add", "--pattern", "p"])
    assert rc == 2
    assert "needs --catch" in capsys.readouterr().err


def test_cli_add_warns_on_missing_pattern(capsys):
    rc = main(["catches", "add", "--catch", "a real new defect"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "no pattern" in err


def test_cli_add_from_json_object(tmp_path, capsys):
    jp = tmp_path / "one.json"
    jp.write_text(json.dumps({"catch": "leaked an internal phrase",
                              "pattern": "source-brief vocabulary leaks",
                              "rule": "scrub the source", "domain": "marketing",
                              "artifact_type": "landing-copy"}), encoding="utf-8")
    rc = main(["catches", "add", "--from-json", str(jp)])
    assert rc == 0
    records = load_catches(catches_path())
    assert len(records) == 1
    assert records[0]["catch"] == "leaked an internal phrase"


def test_cli_add_from_json_list(tmp_path):
    jp = tmp_path / "many.json"
    jp.write_text(json.dumps([
        {"catch": "defect one", "pattern": "p1", "rule": "r1", "artifact_type": "x"},
        {"catch": "defect two", "pattern": "p2", "rule": "r2", "artifact_type": "x"},
    ]), encoding="utf-8")
    rc = main(["catches", "add", "--from-json", str(jp)])
    assert rc == 0
    assert len(load_catches(catches_path())) == 2


def test_cli_add_from_json_rejects_pass_like(tmp_path, capsys):
    jp = tmp_path / "bad.json"
    jp.write_text(json.dumps({"catch": "clean", "pattern": "p"}), encoding="utf-8")
    rc = main(["catches", "add", "--from-json", str(jp)])
    assert rc == 2
    assert "NAME A DEFECT" in capsys.readouterr().err


def test_cli_add_from_json_batch_is_atomic(tmp_path, capsys):
    """A mid-batch doctrine rejection appends NOTHING (no partial write —
    a retried fixed batch must not duplicate the records that 'made it')."""
    jp = tmp_path / "mixed.json"
    jp.write_text(json.dumps([
        {"catch": "defect one", "pattern": "p1", "rule": "r1", "artifact_type": "x"},
        {"catch": "defect two", "pattern": "p2", "rule": "r2", "artifact_type": "x"},
        {"catch": "all lenses passed", "pattern": "p3"},      # rejected
    ]), encoding="utf-8")
    rc = main(["catches", "add", "--from-json", str(jp)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "record 3 of 3" in err
    assert "all-or-nothing" in err
    # the two valid records were NOT appended
    assert load_catches(catches_path()) == []


def test_cli_add_seed_copies_examples(capsys):
    rc = main(["catches", "add", "--seed"])
    assert rc == 0
    records = load_catches(catches_path())
    assert len(records) == 6
    assert "Seeded 6 example" in capsys.readouterr().out


def test_cli_seed_then_recurrence_promotes(capsys):
    main(["catches", "add", "--seed"])
    # add two more of the marker-loss pattern to reach threshold 3
    p = ("evidence markers get lost when a writer summarizes marked prose "
         "into a table or bullet list")
    main(["catches", "add", "--catch", "second drop", "--pattern", p,
          "--rule", "diff", "--artifact-type", "x"])
    main(["catches", "add", "--catch", "third drop", "--pattern", p,
          "--rule", "diff", "--artifact-type", "x"])
    capsys.readouterr()
    rc = main(["catches", "stats", "--threshold", "3"])
    assert rc == 0
    assert "PROMOTE: this pattern has recurred 3x" in capsys.readouterr().out


def test_cli_relevant_requires_artifact_type_or_all(capsys):
    rc = main(["catches", "relevant"])
    assert rc == 2
    assert "--all" in capsys.readouterr().err


def test_cli_relevant_all_and_json_format(capsys):
    main(["catches", "add", "--seed"])
    capsys.readouterr()
    rc = main(["catches", "relevant", "--all", "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list) and len(parsed) == 6


def test_cli_relevant_threshold_surfaces_promote(capsys):
    main(["catches", "add", "--seed"])
    p = ("evidence markers get lost when a writer summarizes marked prose "
         "into a table or bullet list")
    for i in range(2):
        main(["catches", "add", "--catch", f"drop {i}", "--pattern", p,
              "--rule", "diff", "--artifact-type", "x"])
    capsys.readouterr()
    main(["catches", "relevant", "--all", "--threshold", "3"])
    out = capsys.readouterr().out
    assert "[PROMOTE]" in out
    assert "(3x)" in out


# ── ledger redaction (S1 review WARN) ────────────────────────────

def test_redact_command_masks_deny_values():
    cmd = redact_command(["consistency", "leaks", "copy.md",
                          "--deny", "secret one", "secret two"])
    assert "secret one" not in cmd
    assert "secret two" not in cmd
    assert "--deny <redacted x2>" in cmd
    assert "consistency leaks copy.md" in cmd     # rest stays legible


def test_redact_command_single_value_singular_placeholder():
    cmd = redact_command(["leaks", "f.md", "--deny", "onlyone"])
    assert "--deny <redacted>" in cmd
    assert "onlyone" not in cmd


def test_redact_command_stops_at_next_flag():
    cmd = redact_command(["leaks", "f.md", "--deny", "x", "--profile", "p.yaml"])
    assert "--deny <redacted>" in cmd
    assert "--profile p.yaml" in cmd              # next flag untouched


def test_redact_command_passes_through_when_no_sensitive_flag():
    cmd = redact_command(["consistency", "numbers", "s.md", "b.md"])
    assert cmd == "consistency numbers s.md b.md"


def test_consistency_leaks_ledger_does_not_echo_deny_terms(capsys):
    copy = Path("copy.md")
    copy.write_text("clean customer copy with nothing forbidden\n", encoding="utf-8")
    rc = main(["consistency", "leaks", "copy.md",
               "--deny", "internal margin phrase", "another secret term"])
    assert rc == 0                               # clean copy -> exit 0
    runs = Path("RUNS.md").read_text()
    assert "internal margin phrase" not in runs
    assert "another secret term" not in runs
    assert "--deny <redacted x2>" in runs


def test_no_echo_args_drops_args_from_ledger(capsys):
    copy = Path("copy.md")
    copy.write_text("clean copy\n", encoding="utf-8")
    rc = main(["--no-echo-args", "consistency", "leaks", "copy.md",
               "--deny", "topsecret"])
    assert rc == 0
    runs = Path("RUNS.md").read_text()
    assert "topsecret" not in runs
    assert "--no-echo-args" not in runs          # the flag itself is never recorded
    assert "lens-kit consistency leaks copy.md [args redacted]" in runs


# ── item 5: optional snippet + lenses_failed (backward-compatible) ───

from lens_kit.catches import validate_lenses_failed  # noqa: E402


def test_build_with_snippet_and_lenses_round_trips(tmp_path):
    path = tmp_path / CATCHES_FILENAME
    c, _ = build_catch(
        catch="fabricated a 47% stat with no source",
        pattern="uncited statistic presented as fact",
        rule="every external number needs a [SOURCE] or [ESTIMATE]",
        domain="marketing", artifact_type="landing-copy",
        snippet="47% of teams ship faster with us.",
        lenses_failed="truth,extrapolation")
    append_catch(c, path)
    r = load_catches(path)[0]
    assert r["snippet"] == "47% of teams ship faster with us."
    assert r["lenses_failed"] == ["truth", "extrapolation"]


def test_snippet_preserved_verbatim_including_newlines(tmp_path):
    path = tmp_path / CATCHES_FILENAME
    raw = "Line one.\n  indented line two\nLine three.\n"
    c, _ = build_catch(catch="a real defect", pattern="p", rule="r",
                       snippet=raw, lenses_failed="truth")
    append_catch(c, path)
    assert load_catches(path)[0]["snippet"] == raw   # no strip, exact bytes


def test_lenses_failed_accepts_list_and_collapses_dupes():
    c, _ = build_catch(catch="d", pattern="p", rule="r",
                       lenses_failed=["truth", "truth", "rights"])
    assert c.lenses_failed == ["truth", "rights"]


def test_lenses_failed_unknown_key_rejected():
    with pytest.raises(CatchError) as e:
        build_catch(catch="d", pattern="p", rule="r",
                    lenses_failed="truth,boguslens")
    assert "boguslens" in str(e.value)


def test_validate_lenses_failed_canonical_set():
    # all 9 canonical keys validate; spelling matches the gate exactly
    from lens_kit.gate import CANONICAL_LENSES
    assert validate_lenses_failed(list(CANONICAL_LENSES)) == list(CANONICAL_LENSES)
    assert validate_lenses_failed(None) == []
    assert validate_lenses_failed("") == []


def test_optional_fields_omitted_when_absent(tmp_path):
    """Backward-compat: a catch with no snippet/lenses_failed serializes to
    EXACTLY the original 9 fields — no empty keys bloat the row."""
    path = tmp_path / CATCHES_FILENAME
    c, _ = build_catch(catch="a real defect", pattern="p", rule="r",
                       artifact_type="x", date="2026-03-01")
    append_catch(c, path)
    rec = json.loads(path.read_text().splitlines()[0])
    assert "snippet" not in rec
    assert "lenses_failed" not in rec
    assert set(rec) == {"id", "date", "domain", "artifact_type", "catch",
                        "pattern", "rule", "self_catch", "schema_version"}


def test_old_row_without_new_fields_loads_relevant_stats(capsys):
    """An OLD jsonl row (pre-extension shape) still loads / relevant / stats."""
    old = ('{"id": "x-1", "date": "2026-01-01", "domain": "marketing", '
           '"artifact_type": "landing-copy", "catch": "dropped a marker", '
           '"pattern": "markers lost in tables", "rule": "diff counts", '
           '"self_catch": false, "schema_version": 1}\n')
    Path(catches_path()).write_text(old, encoding="utf-8")
    recs = load_catches(catches_path())
    assert len(recs) == 1 and "snippet" not in recs[0]
    rc = main(["catches", "relevant", "landing-copy"])
    assert rc == 0
    assert "dropped a marker" in capsys.readouterr().out
    rc = main(["catches", "stats"])
    assert rc == 0
    assert "1 catch(es)" in capsys.readouterr().out


def test_cli_add_snippet_file_and_lenses_failed(tmp_path, capsys):
    snip = tmp_path / "snip.txt"
    snip.write_text("Save 80% today, limited spots!\n", encoding="utf-8")
    rc = main(["catches", "add",
               "--catch", "dark-pattern urgency with fabricated saving",
               "--pattern", "false-scarcity urgency",
               "--rule", "no urgency claims without a real deadline",
               "--artifact-type", "landing-copy", "--domain", "marketing",
               "--snippet-file", str(snip),
               "--lenses-failed", "truth,extrapolation"])
    assert rc == 0
    r = load_catches(catches_path())[0]
    assert r["snippet"] == "Save 80% today, limited spots!\n"
    assert r["lenses_failed"] == ["truth", "extrapolation"]


def test_cli_add_unknown_lens_is_usage_error(capsys):
    rc = main(["catches", "add", "--catch", "a real defect",
               "--pattern", "p", "--rule", "r",
               "--lenses-failed", "truth,notalens"])
    assert rc == 2
    assert "notalens" in capsys.readouterr().err


def test_cli_add_snippet_file_missing_is_usage_error(capsys):
    rc = main(["catches", "add", "--catch", "a real defect",
               "--pattern", "p", "--rule", "r",
               "--snippet-file", "does-not-exist.txt"])
    assert rc == 2
    assert "snippet-file not found" in capsys.readouterr().err


def test_cli_add_from_json_with_new_fields(tmp_path):
    jp = tmp_path / "one.json"
    jp.write_text(json.dumps({
        "catch": "fabricated stat", "pattern": "uncited number",
        "rule": "cite or mark", "domain": "marketing",
        "artifact_type": "landing-copy",
        "snippet": "92% of users agree.",
        "lenses_failed": ["truth"]}), encoding="utf-8")
    rc = main(["catches", "add", "--from-json", str(jp)])
    assert rc == 0
    r = load_catches(catches_path())[0]
    assert r["snippet"] == "92% of users agree."
    assert r["lenses_failed"] == ["truth"]


def test_cli_add_from_json_unknown_lens_rejected(tmp_path, capsys):
    jp = tmp_path / "bad.json"
    jp.write_text(json.dumps({"catch": "d", "pattern": "p", "rule": "r",
                              "lenses_failed": ["nope"]}), encoding="utf-8")
    rc = main(["catches", "add", "--from-json", str(jp)])
    assert rc == 2
    assert "nope" in capsys.readouterr().err


# ── item 6: catches export --as-labels ───────────────────────────

from lens_kit.catches import build_export_labels, holdout_contamination_dir  # noqa: E402


def _eligible(catch, snippet, lenses, **kw):
    c, _ = build_catch(catch=catch, pattern=kw.pop("pattern", "p"),
                       rule=kw.pop("rule", "r"), snippet=snippet,
                       lenses_failed=lenses, **kw)
    return c


def test_export_per_lens_inversion_named_false_rest_true():
    from lens_kit.gate import CANONICAL_LENSES
    recs = [_eligible("d", "47% of teams agree.", "truth,extrapolation",
                      domain="marketing").to_record()]
    labels, skipped = build_export_labels(recs)
    assert skipped == 0 and len(labels) == 1
    pl = labels[0]["per_lens"]
    assert pl["truth"] is False and pl["extrapolation"] is False
    assert all(pl[l] is True for l in CANONICAL_LENSES
               if l not in ("truth", "extrapolation"))
    # all canonical lenses present, no extras
    assert set(pl) == set(CANONICAL_LENSES)
    assert labels[0]["text"] == "47% of teams agree."
    assert labels[0]["source"] == "catches"
    assert labels[0]["text_length"] == len("47% of teams agree.")


def test_export_skips_catches_missing_snippet_or_lenses():
    recs = [
        _eligible("a real defect", "snippet here", "truth").to_record(),
        build_catch(catch="defect without a snippet recorded", pattern="p",
                    rule="r", lenses_failed="truth")[0].to_record(),  # no snippet
        build_catch(catch="defect without lenses recorded", pattern="p",
                    rule="r", snippet="some text")[0].to_record(),   # no lenses_failed
        build_catch(catch="bare defect, neither field",
                    pattern="p", rule="r")[0].to_record(),           # neither
    ]
    labels, skipped = build_export_labels(recs)
    assert len(labels) == 1
    assert skipped == 3


def test_export_dedupe_by_source_id():
    a = _eligible("d", "same snippet", "truth", catch_id="dup-1").to_record()
    b = _eligible("d2", "different snippet", "rights", catch_id="dup-1").to_record()
    labels, skipped = build_export_labels([a, b])
    assert len(labels) == 1                      # second dup-1 dropped
    assert labels[0]["text"] == "same snippet"   # first wins
    assert skipped == 0                          # a dup is not "skipped"


def test_export_domain_filter():
    recs = [_eligible("d", "mk snippet", "truth", domain="marketing").to_record(),
            _eligible("d", "fin snippet", "truth", domain="finance").to_record()]
    labels, _ = build_export_labels(recs, domain_filter="finance")
    assert len(labels) == 1 and labels[0]["domain"] == "finance"


def test_export_round_trips_load_holdout(tmp_path):
    """ACCEPTANCE: exported file loads through eval_harness.load_holdout
    unmodified and validate_lens_labels passes on every row."""
    from lens_kit.eval_harness import load_holdout
    recs = [
        _eligible("d1", "92% of users agree.", "truth").to_record(),
        _eligible("d2", "It will 10x revenue next quarter.",
                  "extrapolation,causality").to_record(),
    ]
    labels, _ = build_export_labels(recs)
    out = tmp_path / "labels.json"
    out.write_text(json.dumps(labels), encoding="utf-8")
    examples, meta = load_holdout(out)          # validates per_lens labels
    assert len(examples) == 2
    assert all("text" in e and "per_lens" in e for e in examples)


def test_export_load_training_examples_accepts_output(tmp_path):
    """The export also loads through the compile-harness training loader."""
    from lens_kit.compile_harness import load_training_examples
    recs = [_eligible("d", "47% of teams ship faster with our platform here.",
                      "truth").to_record()]
    labels, _ = build_export_labels(recs)
    out = tmp_path / "labels.json"
    out.write_text(json.dumps(labels), encoding="utf-8")
    examples = load_training_examples(out)
    assert len(examples) == 1


# contamination guard
def test_contamination_guard_fires_when_holdout_colocated(tmp_path):
    (tmp_path / "my-holdout.json").write_text("[]", encoding="utf-8")
    assert holdout_contamination_dir(tmp_path / "labels.json") is True


def test_contamination_guard_quiet_without_holdout(tmp_path):
    (tmp_path / "train-data.json").write_text("[]", encoding="utf-8")
    assert holdout_contamination_dir(tmp_path / "labels.json") is False


def test_contamination_guard_ignores_export_target_itself(tmp_path):
    # the target file may itself contain 'holdout' in its name — not a false trip
    assert holdout_contamination_dir(tmp_path / "labels.json") is False


# ── CLI: export ──────────────────────────────────────────────────

def test_cli_export_round_trip_and_skip_count(tmp_path, capsys):
    main(["catches", "add", "--catch", "fabricated stat",
          "--pattern", "uncited number", "--rule", "cite it",
          "--snippet-file", _write(tmp_path, "s1.txt", "92% of users agree."),
          "--lenses-failed", "truth", "--domain", "marketing"])
    main(["catches", "add", "--catch", "a defect with no snippet",
          "--pattern", "p", "--rule", "r"])     # ineligible
    capsys.readouterr()
    out = tmp_path / "labels.json"
    rc = main(["catches", "export", "--as-labels", str(out)])
    assert rc == 0
    msg = capsys.readouterr().out
    assert "1 catches exported, 1 skipped" in msg
    # round-trips
    from lens_kit.eval_harness import load_holdout
    examples, _ = load_holdout(out)
    assert len(examples) == 1
    assert examples[0]["per_lens"]["truth"] is False


def test_cli_export_empty_memory_exits_zero(tmp_path, capsys):
    out = tmp_path / "labels.json"
    rc = main(["catches", "export", "--as-labels", str(out)])
    assert rc == 0
    assert "0 catches exported, 0 skipped" in capsys.readouterr().out
    assert json.loads(out.read_text()) == []


def test_cli_export_appends_ledger_row(tmp_path, capsys):
    main(["catches", "add", "--catch", "fabricated stat", "--pattern", "p",
          "--rule", "r", "--snippet-file", _write(tmp_path, "s.txt", "92% agree."),
          "--lenses-failed", "truth"])
    capsys.readouterr()
    main(["catches", "export", "--as-labels", str(tmp_path / "labels.json")])
    runs = Path("RUNS.md").read_text()
    assert "catches-export" in runs
    assert "EXPORT" in runs


def test_cli_export_contamination_warning_fires(tmp_path, capsys):
    (tmp_path / "frozen-holdout.json").write_text("[]", encoding="utf-8")
    main(["catches", "add", "--catch", "fabricated stat", "--pattern", "p",
          "--rule", "r", "--snippet-file", _write(tmp_path, "s.txt", "92% agree."),
          "--lenses-failed", "truth"])
    capsys.readouterr()
    rc = main(["catches", "export", "--as-labels", str(tmp_path / "labels.json")])
    assert rc == 0
    assert "*holdout* file" in capsys.readouterr().err


def test_cli_export_no_contamination_warning_when_clean(tmp_path, capsys):
    main(["catches", "add", "--catch", "fabricated stat", "--pattern", "p",
          "--rule", "r", "--snippet-file", _write(tmp_path, "s.txt", "92% agree."),
          "--lenses-failed", "truth"])
    capsys.readouterr()
    rc = main(["catches", "export", "--as-labels", str(tmp_path / "labels.json")])
    assert rc == 0
    assert "*holdout*" not in capsys.readouterr().err


def _write(d, name, text):
    p = d / name
    p.write_text(text, encoding="utf-8")
    return str(p)
