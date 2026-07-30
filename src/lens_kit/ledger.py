"""RUNS.md — the append-only run ledger.

Both `lens-kit compile` and `lens-kit eval` append one row per run to a
RUNS.md in the working directory (created with header on first use).
DISCARD/KILLED/STOPPED_COST rows are kept forever — "tried and didn't
work" is evidence, not clutter.

Run-designation rule (printed in the header): any edit to labels, fixtures,
or training/holdout data files forces a NEW run designation. Never extend or
compare metrics across a label/fixture edit under the same run id.
"""
import time
from pathlib import Path

LEDGER_FILENAME = "RUNS.md"

# Flags whose VALUES are sensitive: the ledger echoes the full command for
# provenance, but a deny-list term (an internal phrase you are scanning FOR)
# must not be written into RUNS.md in the clear. S1 review WARN: append_run
# echoed `--deny <internal term>` verbatim. ``redact_command`` replaces each
# such flag's value(s) with a placeholder, preserving the shape of the command.
SENSITIVE_VALUE_FLAGS = ("--deny",)


def redact_command(parts, sensitive_flags=SENSITIVE_VALUE_FLAGS) -> str:
    """Join ``parts`` into a command string, masking sensitive flag VALUES.

    For each flag in ``sensitive_flags`` (e.g. ``--deny``) every following
    value token (up to the next ``-``-prefixed token or end) is replaced with
    ``<redacted>`` and the count summarized: ``--deny <redacted x3>``. The flag
    itself stays visible so the command remains legible; only the secret terms
    are masked. Non-sensitive tokens pass through unchanged.
    """
    parts = [str(p) for p in parts]
    flags = set(sensitive_flags)
    out = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok in flags:
            out.append(tok)
            j = i + 1
            n = 0
            while j < len(parts) and not parts[j].startswith("-"):
                n += 1
                j += 1
            out.append(f"<redacted x{n}>" if n != 1 else "<redacted>")
            i = j
            continue
        out.append(tok)
        i += 1
    return " ".join(out)

HEADER = """\
# RUNS — lens-kit run ledger

RULE: label/fixture/data-file edits force a NEW run designation. Never extend or
compare metrics across a label or fixture edit under the same run id.
Rows are append-only; DISCARD/KILLED/STOPPED_COST rows stay (negative results are evidence).

| Run id | Date | Command | Checkpoint sha | Labels/data file | Key metrics | Status |
|---|---|---|---|---|---|---|
"""


def new_run_id(command: str) -> str:
    return f"{command}-{time.strftime('%Y%m%d-%H%M%S')}"


def _cell(value) -> str:
    """One markdown table cell: no pipes, no newlines, never empty."""
    text = str(value if value not in (None, "") else "-")
    return text.replace("|", "\\|").replace("\n", " ")


def append_run(*, run_id: str, command: str, checkpoint_sha: str | None,
               data_file: str | None, metrics: str, status: str,
               ledger_path: str | Path | None = None) -> Path:
    """Append one row; create the ledger with header if absent. Returns path."""
    path = Path(ledger_path) if ledger_path else Path.cwd() / LEDGER_FILENAME
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    sha_cell = _cell(checkpoint_sha[:12] if checkpoint_sha else None)
    row = (f"| {_cell(run_id)} | {time.strftime('%Y-%m-%d %H:%M')} | {_cell(command)} "
           f"| {sha_cell} | {_cell(data_file)} | {_cell(metrics)} | {_cell(status)} |\n")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(row)
    return path
