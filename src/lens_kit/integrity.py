"""Gate-integrity trend monitor — the continuous version of mutation control.

Mutation control asks once: can the gate still FAIL a planted flaw? This module
asks it over TIME, from the receipts the kit already writes. A validator that
approves everything for run after run is not proven healthy by its green
scorecards — it is the exact shape of a rubber stamp. The mutation battery is a
point check; a gate can pass it on Monday and quietly drift into rubber-stamping
by Friday. This monitor reads the append-only `RUNS.md` ledger and the
`catches.jsonl` memory and surfaces three trend signals:

  (a) pass-rate over the last N eval runs, and its trend (rising / flat /
      falling) — a high AND rising pass-rate is the first half of the
      rubber-stamp shape;
  (b) runs and days since the last NAMED catch in catches.jsonl — a memory that
      has gone silent is the second half (a live validator keeps finding
      defects);
  (c) recency of the last mutation-control run (a `mutate-` ledger row) — the
      rubber-stamp detector itself can go stale.

TRIPWIRE, NOT ORACLE (read this before trusting a result):
  A flag means "look here", never "the gate is broken". The monitor cannot see
  WHY the pass-rate is high — a genuinely good gate on easy data looks identical
  to a rubber stamp at this altitude. That is why the headline tripwire requires
  the FULL rubber-stamp shape together: a high+rising pass-rate over enough runs
  AND zero catches in the window. Either signal alone is not enough.
  Crucially: the ABSENCE of a flag is NOT evidence of health. A quiet monitor
  means "no tripwire fired on the receipts you have", not "the gate is sound" —
  exactly the inversion this whole module exists to refuse.

  Where a signal is not identifiable from the ledger, it is surfaced as
  "unknown" honestly — never guessed. If the ledger records no mutation-control
  run, last_mutation is "unknown", not "never" and not a fabricated date.

Report-only by default (exit 0). `--strict` exits 6 (the integrity/consistency
exit family) when a threshold is breached, so a CI wrapper can gate on it.
No network, no LLM: pure parsing of two local files.
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

from .config import IntegrityConfig

EXIT_OK = 0
EXIT_USAGE = 2
# Shared with consistency/calibration's integrity family (exit-code map in
# cli.py): a breached integrity threshold is "look here", surfaced under the
# same code as a consistency violation. No NEW exit code is minted.
EXIT_INTEGRITY = 6


class IntegrityError(Exception):
    """Raised on a malformed/missing input. Maps to exit 2 (usage)."""


# ── Ledger parsing ───────────────────────────────────────────────

# A data row in RUNS.md: a markdown table row with 7 pipe-delimited cells.
# The header + separator rows are skipped (they contain "Run id" / "---").
_ACCURACY_RE = re.compile(r"overall_accuracy(?:_floor)?\s*=\s*([0-9]*\.?[0-9]+)")


def _run_kind(run_id: str) -> str:
    """The command family a run id belongs to (prefix before the timestamp).

    Run ids are ``<command>-<YYYYmmdd>-<HHMMSS>`` (see ledger.new_run_id), so the
    leading token is the command family: eval / compile / calib / mutate /
    consistency / integrity / label-audit / relabel. Unrecognized -> "other".
    """
    rid = (run_id or "").strip()
    for kind in ("eval", "compile", "calib", "mutate", "consistency",
                 "integrity", "label-audit", "relabel"):
        if rid.startswith(kind + "-"):
            return kind
    return "other"


def parse_ledger(path: str | Path) -> list[dict]:
    """Parse RUNS.md into typed rows (oldest first). Hard error on missing/empty.

    Each row dict: run_id, kind, date, command, checkpoint_sha, data_file,
    metrics (raw cell), status, accuracy (float | None, parsed from metrics).
    A ledger with a header but no data rows is an IntegrityError — there is
    nothing to monitor, and silently returning [] would let --strict pass for
    the wrong reason.
    """
    p = Path(path)
    if not p.exists():
        raise IntegrityError(f"ledger not found: {p}")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) != 7:
            continue
        run_id = cells[0]
        # Skip the header row and the |---|---| separator.
        if run_id.lower() in ("run id", "") or set(run_id) <= set("-: "):
            continue
        metrics = cells[5]
        m = _ACCURACY_RE.search(metrics)
        rows.append({
            "run_id": run_id,
            "kind": _run_kind(run_id),
            "date": cells[1],
            "command": cells[2],
            "checkpoint_sha": cells[3],
            "data_file": cells[4],
            "metrics": metrics,
            "status": cells[6],
            "accuracy": (float(m.group(1)) if m else None),
        })
    if not rows:
        raise IntegrityError(
            f"{p.name} has no data rows — nothing to monitor. Run some "
            f"evals/validations first, then check integrity.")
    return rows


def _load_catches(path: str | Path) -> list[dict]:
    """Load catches.jsonl. Missing file -> [] (a fresh project has no catches);
    malformed JSON -> IntegrityError (a memory you can't read can't be trusted).
    """
    import json
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise IntegrityError(f"{p.name}:{lineno}: malformed JSON ({e})") from e
        if not isinstance(rec, dict):
            raise IntegrityError(f"{p.name}:{lineno}: record is not a JSON object")
        out.append(rec)
    return out


# ── Trend math ───────────────────────────────────────────────────

def _trend(values: list[float]) -> str:
    """rising / flat / falling, by comparing the mean of the older half to the
    newer half. Documented method: split the ordered series in two; if the newer
    half's mean exceeds the older by more than a small epsilon it is rising,
    below by more than epsilon it is falling, else flat. Fewer than 2 points has
    no trend -> 'flat'.
    """
    if len(values) < 2:
        return "flat"
    mid = len(values) // 2
    older = values[:mid] or values[:1]
    newer = values[mid:]
    older_mean = sum(older) / len(older)
    newer_mean = sum(newer) / len(newer)
    eps = 0.01
    if newer_mean > older_mean + eps:
        return "rising"
    if newer_mean < older_mean - eps:
        return "falling"
    return "flat"


def _parse_date(s: str):
    """First YYYY-MM-DD in a string -> date, else None (never guess)."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s or "")
    if not m:
        return None
    try:
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _latest_catch_date(catches: list[dict]):
    dates = [d for d in (_parse_date(c.get("date", "")) for c in catches) if d]
    return max(dates) if dates else None


# ── Analysis ─────────────────────────────────────────────────────

def analyze_integrity(ledger_path, catches_path, *, last: int,
                      config: IntegrityConfig) -> dict:
    """Compute the trend signals + flags from the two files. Pure, no I/O beyond
    reading the two given paths.

    Returns a report dict (report-only; the exit code is decided by the caller):
      pass_rate            mean accuracy over the last <= `last` eval runs (None
                           if no eval run carried a parseable accuracy)
      trend                rising / flat / falling over that window
      eval_runs_in_window  count of eval runs whose accuracy was usable
      catches_total        records in catches.jsonl
      runs_since_last_catch eval/validate runs appended AFTER the last catch's
                           date (None if there is no catch at all)
      days_since_last_catch days since the last catch (None if none)
      last_mutation        "unknown" if no mutate- row, else the most recent
                           mutate- row's date string
      runs_since_mutation  eval runs since the last mutation run (None/unknown)
      flags                human-readable strings; empty == no tripwire fired
      flagged              bool(flags)
    """
    rows = parse_ledger(ledger_path)        # oldest first; raises on empty
    catches = _load_catches(catches_path)

    eval_rows = [r for r in rows if r["kind"] == "eval"]
    window = eval_rows[-last:] if last and last > 0 else eval_rows
    accs = [r["accuracy"] for r in window if r["accuracy"] is not None]
    pass_rate = (sum(accs) / len(accs)) if accs else None
    trend = _trend(accs)

    latest_catch = _latest_catch_date(catches)
    # Runs since the last catch: eval rows dated strictly after the catch date.
    runs_since_last_catch = None
    days_since_last_catch = None
    if latest_catch is not None:
        runs_since_last_catch = sum(
            1 for r in eval_rows
            if (_parse_date(r["date"]) or datetime.date.min) > latest_catch)
        latest_run_date = max(
            (d for d in (_parse_date(r["date"]) for r in rows) if d),
            default=None)
        if latest_run_date is not None:
            days_since_last_catch = (latest_run_date - latest_catch).days

    # Mutation-control recency. No mutate- row -> "unknown", honestly.
    mutate_rows = [r for r in rows if r["kind"] == "mutate"]
    if mutate_rows:
        last_mutation = mutate_rows[-1]["date"]
        last_mut_date = _parse_date(mutate_rows[-1]["date"])
        runs_since_mutation = (
            sum(1 for r in eval_rows
                if (_parse_date(r["date"]) or datetime.date.min)
                > (last_mut_date or datetime.date.min))
            if last_mut_date else None)
    else:
        last_mutation = "unknown"
        runs_since_mutation = None

    catches_in_window = _catches_in_window(rows, window, catches)

    flags = _flags(config, pass_rate=pass_rate, trend=trend,
                   eval_runs_in_window=len(accs),
                   catches_in_window=catches_in_window,
                   runs_since_last_catch=runs_since_last_catch,
                   days_since_last_catch=days_since_last_catch,
                   last_mutation=last_mutation,
                   runs_since_mutation=runs_since_mutation)

    return {
        "last": last,
        "pass_rate": pass_rate,
        "trend": trend,
        "eval_runs_total": len(eval_rows),
        "eval_runs_in_window": len(accs),
        "catches_total": len(catches),
        "catches_in_window": catches_in_window,
        "runs_since_last_catch": runs_since_last_catch,
        "days_since_last_catch": days_since_last_catch,
        "last_mutation": last_mutation,
        "runs_since_mutation": runs_since_mutation,
        "flags": flags,
        "flagged": bool(flags),
    }


def _catches_in_window(rows: list[dict], window: list[dict],
                       catches: list[dict]) -> int:
    """Count catches dated on/after the window's earliest run date.

    The window is the last-N eval runs; a catch recorded during that span is
    evidence the validator is still finding defects. If the window has no
    parseable date, fall back to the total catch count (conservative: assume
    they are in window so the rubber-stamp flag does not fire on a date gap).
    """
    win_dates = [d for d in (_parse_date(r["date"]) for r in window) if d]
    if not win_dates:
        return len(catches)
    earliest = min(win_dates)
    return sum(1 for c in catches
               if (_parse_date(c.get("date", "")) or datetime.date.min) >= earliest)


def _flags(config: IntegrityConfig, *, pass_rate, trend, eval_runs_in_window,
           catches_in_window, runs_since_last_catch, days_since_last_catch,
           last_mutation, runs_since_mutation) -> list[str]:
    """Build the human-readable flag list. Each flag is a tripwire, not a verdict.

    Headline tripwire (the rubber-stamp shape): a high (>= pass_rate_warn) AND
    non-falling pass-rate over >= min_runs eval runs, WITH zero catches in the
    window. Both halves required — a high pass-rate alone is not flagged.
    """
    flags = []

    if (pass_rate is not None
            and eval_runs_in_window >= config.min_runs
            and pass_rate >= config.pass_rate_warn
            and trend in ("rising", "flat")
            and catches_in_window == 0):
        flags.append(
            f"RUBBER-STAMP SHAPE: pass-rate {pass_rate:.3f} (>= "
            f"{config.pass_rate_warn:.3f}), trend {trend}, over "
            f"{eval_runs_in_window} runs, with ZERO catches in the window — a "
            f"validator that approves everything is surfaced, not trusted. "
            f"Look here: run mutation control and re-audit a sample by hand.")

    if (runs_since_last_catch is not None
            and runs_since_last_catch >= config.stale_catch_runs):
        flags.append(
            f"STALE MEMORY: {runs_since_last_catch} runs since the last named "
            f"catch (>= {config.stale_catch_runs}) — the catches memory has "
            f"gone quiet; a live validator keeps finding defects.")
    if (days_since_last_catch is not None
            and days_since_last_catch >= config.stale_catch_days):
        flags.append(
            f"STALE MEMORY: {days_since_last_catch} days since the last named "
            f"catch (>= {config.stale_catch_days}).")

    if last_mutation == "unknown":
        # Honest unknown is not itself a flag (absence of a mutation row is
        # common in a fresh project); only surface staleness when we CAN see it.
        pass
    elif (runs_since_mutation is not None
          and runs_since_mutation >= config.stale_mutation_runs):
        flags.append(
            f"STALE MUTATION CONTROL: {runs_since_mutation} eval runs since the "
            f"last mutation-control run (>= {config.stale_mutation_runs}) — the "
            f"rubber-stamp detector itself is overdue.")

    return flags


# ── Reporting + command entry ────────────────────────────────────

def render_report(report: dict) -> str:
    pr = ("n/a" if report["pass_rate"] is None
          else f"{report['pass_rate']:.3f}")
    rslc = ("never caught" if report["runs_since_last_catch"] is None
            else str(report["runs_since_last_catch"]))
    lines = [
        "[integrity] gate-integrity trend monitor (tripwire, not oracle)",
        f"  pass-rate (last {report['last']} eval runs): {pr}  "
        f"trend: {report['trend']}  (n={report['eval_runs_in_window']})",
        f"  catches: {report['catches_total']} total, "
        f"{report['catches_in_window']} in window; "
        f"runs since last catch: {rslc}",
        f"  last mutation-control run: {report['last_mutation']}",
    ]
    if report["flagged"]:
        lines.append(f"  FLAGS ({len(report['flags'])}):")
        for f in report["flags"]:
            lines.append(f"    - {f}")
    else:
        lines.append("  no tripwire fired — NOTE: absence of a flag is NOT "
                     "evidence of health (see the docstring).")
    return "\n".join(lines) + "\n"


def run_integrity(ledger_path, catches_path, *, last: int, strict: bool,
                  config: IntegrityConfig,
                  command_str: str = "lens-kit integrity",
                  write_ledger: bool = True) -> int:
    """Print the report, append a ledger row, return the exit code.

    Report-only by default: exit 0 even when flagged (a flag is "look here", not
    a gate). `strict=True`: exit 6 when any flag is present. Malformed/missing
    inputs (IntegrityError) -> exit 2 (usage), never a crash.
    """
    try:
        report = analyze_integrity(ledger_path, catches_path, last=last,
                                   config=config)
    except IntegrityError as e:
        import sys
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE

    print(render_report(report), end="")

    if write_ledger:
        from . import ledger
        run_id = ledger.new_run_id("integrity")
        pr = ("n/a" if report["pass_rate"] is None
              else f"{report['pass_rate']:.3f}")
        metrics = (f"pass_rate={pr}; trend={report['trend']}; "
                   f"flags={len(report['flags'])}")
        status = ("INTEGRITY_FLAG" if report["flagged"] else "INTEGRITY_OK")
        ledger.append_run(
            run_id=run_id, command=command_str, checkpoint_sha=None,
            data_file=Path(ledger_path).name, metrics=metrics, status=status,
            ledger_path=ledger_path)
        print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")

    if strict and report["flagged"]:
        return EXIT_INTEGRITY
    return EXIT_OK
