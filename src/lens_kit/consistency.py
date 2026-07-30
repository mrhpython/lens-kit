"""Deterministic cross-artifact consistency checks (no LLM, no network).

These mechanize three recurring catch CLASSES from the validator's
institutional memory — the ones whose misses are all CROSS-something
(cross-file, cross-format, cross-section) and so slip past a single-document
LLM gate:

  1. markers  — evidence markers ([UNVERIFIED]/[ESTIMATE]/[PROJECTION], ...)
                counted in a SOURCE artifact must survive into every RENDERED
                output. Markers dropped in a summary table or a platform
                variant are how an unverified number gets laundered into a
                clean-looking claim.
  2. leaks    — a deny-list of strings that must NEVER reach customer-facing
                copy (internal lens names/counts, cost data, internal
                vocabulary). One hit is a violation that no score overrides
                (recovery 1.10: a leak is a leak regardless of how good the
                copy reads).
  3. numbers  — numeric literals asserted in a SUMMARY must appear in the BODY
                it summarizes. A summary that invents or mistypes a number the
                body never states is an exec-summary-vs-detail mismatch.

TRIPWIRE, NOT ORACLE (read this before trusting a result):
  Every check is deterministic and LITERAL. None of them understands meaning.
    - markers: a rendered file that legitimately covers a SUBSET of the source
      will under-count markers and be flagged — a false positive by design.
      The check can't know what a rendered file "includes"; it flags the
      under-count so a human/agent looks, it does not adjudicate.
    - numbers: literal matching only. No semantic math, no derived values:
      "13 of 17" in a summary will NOT be resolved to "76%" in the body and
      will false-positive. "$1.2M" matches "$1,200,000" only because they
      normalize identically — anything requiring arithmetic will not.
    - leaks: literal substring matching (case-insensitive). No regex, so no
      regex-injection risk and no fuzzy/semantic matching — a paraphrase of a
      denied phrase is not caught.
  A clean run means "no tripwire fired", not "the artifacts are consistent".
  A firing means "a human/agent must look here", not "the artifact is wrong".

Exit code 6 (consistency violation) is shared across all three checks.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

EXIT_CLEAN = 0
EXIT_USAGE = 2
EXIT_CONSISTENCY = 6

DEFAULT_MARKERS = ("[UNVERIFIED]", "[ESTIMATE]", "[PROJECTION]")


# ── Result shape ─────────────────────────────────────────────────

@dataclass
class CheckResult:
    """One consistency check's outcome. ``violations`` is human-readable lines."""
    check: str
    violated: bool
    violations: list = field(default_factory=list)
    summary: str = ""

    @property
    def exit_code(self) -> int:
        return EXIT_CONSISTENCY if self.violated else EXIT_CLEAN


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


# ── 1. Marker parity ─────────────────────────────────────────────

def count_marker(text: str, marker: str) -> int:
    """Literal, case-sensitive occurrences of ``marker`` in ``text``.

    Markers like ``[UNVERIFIED]`` are case-sensitive tokens by convention;
    overlapping matches are not possible for these bracketed literals, so a
    plain non-overlapping count is exact.
    """
    if not marker:
        return 0
    return text.count(marker)


def check_markers(source_path: str | Path, rendered_paths: list,
                  markers=DEFAULT_MARKERS) -> CheckResult:
    """Flag any rendered file whose count of any marker is LOWER than source.

    Contract (tripwire, not oracle): for each marker, count occurrences in the
    source and in each rendered file. A rendered file that carries FEWER of a
    marker than the source has dropped markers for whatever content it
    includes — OR it legitimately covers a subset of the source (a false
    positive). The check reports the under-count for human/agent review; it
    does not decide which case it is. Equal-or-greater counts pass.
    """
    markers = list(markers)
    source = _read(source_path)
    src_counts = {m: count_marker(source, m) for m in markers}

    violations = []
    for rp in rendered_paths:
        rendered = _read(rp)
        for m in markers:
            have = count_marker(rendered, m)
            need = src_counts[m]
            if have < need:
                violations.append(
                    f"{Path(rp).name}: {m} count {have} < source {need} "
                    f"(markers dropped in render, OR this output covers a subset "
                    f"of the source — review)")

    src_desc = ", ".join(f"{m}={src_counts[m]}" for m in markers)
    return CheckResult(
        check="markers",
        violated=bool(violations),
        violations=violations,
        summary=(f"source markers ({Path(source_path).name}): {src_desc}; "
                 f"{len(rendered_paths)} rendered file(s) checked"))


# ── 2. Forbidden-strings leak scan ───────────────────────────────

def scan_leaks(text: str, deny: list) -> list:
    """(line_number, line_text, matched_term) for every case-insensitive hit.

    Literal substring matching only — the deny entries are treated as plain
    text, never compiled as regex, so a customer's deny-list can contain any
    characters without injection risk or accidental wildcard behaviour.
    """
    deny_lower = [(d, d.lower()) for d in deny if d]
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line_lower = line.lower()
        for original, lower in deny_lower:
            if lower in line_lower:
                hits.append((lineno, line, original))
    return hits


def check_leaks(file_paths: list, deny: list) -> CheckResult:
    """ANY deny-list hit in ANY file is a consistency violation (recovery 1.10).

    Per-hit ``file:line`` output. No score, no quality, no length of the copy
    overrides a leak — internal vocabulary in customer-facing text is a defect
    by itself.
    """
    deny = [d for d in deny if d]
    violations = []
    if not deny:
        return CheckResult(
            check="leaks", violated=False, violations=[],
            summary="no deny-list configured (consistency.deny empty + no --deny) "
                    "— nothing to scan")

    for fp in file_paths:
        text = _read(fp)
        for lineno, line, term in scan_leaks(text, deny):
            snippet = line.strip()
            if len(snippet) > 100:
                snippet = snippet[:97] + "..."
            violations.append(f"{Path(fp).name}:{lineno}: forbidden '{term}' "
                              f"in: {snippet}")

    return CheckResult(
        check="leaks",
        violated=bool(violations),
        violations=violations,
        summary=f"{len(deny)} deny term(s) x {len(file_paths)} file(s) scanned")


# ── 3. Number parity ─────────────────────────────────────────────

# A numeric literal: optional currency symbol, digits with thousands
# separators (comma or space) or a decimal, optional %/k/M/bn suffix.
# Matched greedily so "$1,200,000" and "76%" come out whole.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                       # not mid-identifier / mid-decimal
    [\$£€¥]?                          # optional currency
    \d{1,3}(?:[,\s]\d{3})+           # grouped thousands: 1,200,000 / 1 200 000
        (?:\.\d+)?                    #   optional decimal
    | [\$£€¥]?\d+(?:\.\d+)?           # or a plain number: 1200000 / 4.2
    """,
    re.VERBOSE,
)
# Suffix attached to a matched number, e.g. the "%", "k", "M" in "76%", "1.2M".
# "%" needs no word boundary (it is itself non-word); the alphabetic suffixes
# do, so "5 million" isn't read as "5m" and "3xy" isn't read as "3x".
_SUFFIX_RE = re.compile(r"^(?:\s*(%)|(k|K|m|M|bn|B|x)\b)")

_SUFFIX_MULT = {
    "k": 1_000, "K": 1_000,
    "m": 1_000_000, "M": 1_000_000,
    "bn": 1_000_000_000, "B": 1_000_000_000,
}


def normalize_number(token: str, suffix: str = "") -> str:
    """Canonicalize a numeric literal for literal (post-normalization) matching.

    Normalizes: currency symbols stripped, thousands separators removed, a
    k/M/bn suffix multiplied out, trailing zeros after a decimal point dropped.
    '%' and 'x' are kept as a tag so "76%" and "76" do not collide.

    Examples:
      "$1,200,000" -> "1200000"      "1.2M" -> "1200000"
      "76%"        -> "76%"          "1 200 000" -> "1200000"
      "4.20"       -> "4.2"          "3x" -> "3x"
    """
    raw = token.strip().lstrip("$£€¥")
    raw = raw.replace(",", "").replace(" ", "")
    tag = ""
    sfx = suffix.strip().lower()
    if sfx == "%":
        tag = "%"
    elif sfx == "x":
        tag = "x"
    try:
        value = float(raw)
    except ValueError:
        return raw + tag
    if sfx in _SUFFIX_MULT:
        value *= _SUFFIX_MULT[sfx]
    if value == int(value):
        return f"{int(value)}{tag}"
    return f"{value:g}{tag}"


def extract_numbers(text: str) -> list:
    """Ordered (line_number, original_token, normalized) for every numeric literal."""
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _NUMBER_RE.finditer(line):
            token = m.group(0)
            tail = line[m.end():]
            sfx_m = _SUFFIX_RE.match(tail)
            suffix = (sfx_m.group(1) or sfx_m.group(2)) if sfx_m else ""
            display = token + (suffix if suffix else "")
            out.append((lineno, display, normalize_number(token, suffix)))
    return out


def check_numbers(summary_path: str | Path, body_path: str | Path) -> CheckResult:
    """Every numeric literal in the summary must appear (normalized) in the body.

    Contract (tripwire, not oracle): LITERAL matching after normalization
    (thousands separators, currency symbols, %, k/M/bn suffixes) only. There
    is NO semantic math and NO derived-value resolution — a summary's
    "13 of 17" will not be reconciled with a body's "76%", and "$1.2M" matches
    "$1,200,000" only because both normalize to "1200000". Orphans (summary
    numbers absent from the body) are reported with the summary line they
    appear on, for human/agent review.
    """
    summary = _read(summary_path)
    body = _read(body_path)
    body_norms = {norm for _, _, norm in extract_numbers(body)}

    violations = []
    seen = set()
    for lineno, display, norm in extract_numbers(summary):
        if norm in body_norms:
            continue
        key = (display, norm)
        if key in seen:
            continue
        seen.add(key)
        violations.append(
            f"{Path(summary_path).name}:{lineno}: '{display}' "
            f"(normalized '{norm}') not found in {Path(body_path).name}")

    return CheckResult(
        check="numbers",
        violated=bool(violations),
        violations=violations,
        summary=(f"{len(extract_numbers(summary))} number(s) in "
                 f"{Path(summary_path).name}; {len(body_norms)} distinct in "
                 f"{Path(body_path).name}"))


# ── Reporting + command entry ────────────────────────────────────

def print_result(result: CheckResult) -> None:
    head = "VIOLATION" if result.violated else "clean"
    print(f"[consistency:{result.check}] {head} — {result.summary}")
    for v in result.violations:
        print(f"  {v}")


def _ledger_metrics(results: list) -> str:
    parts = []
    for r in results:
        parts.append(f"{r.check}={'FAIL' if r.violated else 'ok'}"
                     + (f"({len(r.violations)})" if r.violated else ""))
    return "; ".join(parts)


def run_consistency(results: list, *, command_str: str = "lens-kit consistency",
                    write_ledger: bool = True) -> int:
    """Print every result, append ONE ledger row, return the CLI exit code.

    Exit 6 if ANY check violated (a leak/dropped-marker/orphan-number is a
    consistency violation no score overrides); else 0.
    """
    for r in results:
        print_result(r)
    violated = any(r.violated for r in results)

    if write_ledger:
        from . import ledger
        run_id = ledger.new_run_id("consistency")
        checks = ",".join(r.check for r in results)
        ledger.append_run(
            run_id=run_id, command=command_str, checkpoint_sha=None,
            data_file=checks, metrics=_ledger_metrics(results),
            status="CONSISTENCY_FAIL" if violated else "CONSISTENCY_OK")
        print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")

    return EXIT_CONSISTENCY if violated else EXIT_CLEAN
