"""Label-audit workflow: audit the GOLD LABELS before you blame the model.

WHY (the argument a customer reads in the README): when a gate disagrees with a
gold label, two things are possible — the gate is wrong, or the LABEL is wrong.
A mislabeled example penalizes a correct gate and rewards a broken one; if you
tune against it you are optimizing toward a bug. Before you accept an eval's
FNs/FPs as the gate's fault, audit the examples the gate disagreed with and
decide, per example, whether the label or the model is at fault. This workflow
produces the audit SCAFFOLD — a Confirmed / Borderline / False-Alarms markdown
report seeded with the suspicious examples and the impact math (how many FNs/FPs
would flip if you relabeled) — and enforces the discipline mechanically: it
refuses to overwrite a dataset file, requires a dated --backup before any edit,
and prints the new-run-designation rule (a label edit forces a NEW run id; never
compare metrics across a relabel).

Pure + offline: reads gold labels + an eval-results JSON, writes a markdown
report. Makes no edits to the dataset itself — the human decides each call from
the scaffold, then re-runs eval under a new run designation.
"""
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

from .checkpoint import sha256_file

LABEL_AUDIT_SCHEMA_VERSION = 1


class BackupRequired(Exception):
    """Raised when a label edit would overwrite a dataset without --backup."""


# ── Suspicion detection ──────────────────────────────────────────

def _load_gold(dataset_path: Path) -> list[dict]:
    doc = json.loads(dataset_path.read_text())
    if isinstance(doc, dict):
        return doc.get("examples", [])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Dataset must be a JSON list or {{'examples': [...]}}: {dataset_path}")


def _load_eval_per_example(eval_results_path: Path) -> list[dict]:
    doc = json.loads(eval_results_path.read_text())
    if isinstance(doc, dict) and "per_example" in doc:
        return doc["per_example"]
    raise ValueError(
        f"Eval-results file {eval_results_path.name} has no 'per_example' block "
        f"— pass a single-run eval doc (lens-kit eval writes one), not an "
        f"envelope.")


def find_suspicious(gold: list[dict], per_example: list[dict]) -> list[dict]:
    """Examples where the gate disagrees with gold on >=1 lens.

    For each example we read the eval's per-lens confusion (fp/fn = disagreement,
    tp/tn = agreement). An example with any fp or fn is a candidate for the
    audit; the more disagreements, the higher up the report. Returns records
    sorted by disagreement count (desc), each carrying the disagreeing lenses
    split into FNs (model passed, gold says FAIL) and FPs (model failed, gold
    says PASS) — the two relabel directions.
    """
    by_index = {row.get("index"): row for row in per_example}
    suspicious: list[dict] = []
    for i, ex in enumerate(gold):
        row = by_index.get(i)
        if row is None:
            continue
        confusion = row.get("lens_confusion") or {}
        fns = [lens for lens, c in confusion.items() if c == "fn"]
        fps = [lens for lens, c in confusion.items() if c == "fp"]
        if not fns and not fps:
            continue
        suspicious.append({
            "index": i,
            "source": ex.get("source", row.get("source", "unknown")),
            "source_id": ex.get("source_id", row.get("source_id", "")),
            "text": str(ex.get("text", "")),
            "gold_per_lens": ex.get("per_lens", {}),
            "fn_lenses": sorted(fns),   # model PASSED, gold says FAIL
            "fp_lenses": sorted(fps),   # model FAILED, gold says PASS
            "disagreements": len(fns) + len(fps),
            "score": row.get("score"),
        })
    suspicious.sort(key=lambda r: r["disagreements"], reverse=True)
    return suspicious


def impact_math(suspicious: list[dict]) -> dict:
    """How many FNs/FPs would flip if every suspicious label were corrected.

    This is the UPPER BOUND of label-attributable error: it assumes every
    disagreement is a label error (it is not — that is what the human decides
    per example). It quantifies the question "how much of my eval's apparent
    error could be phantom?". Per-lens breakdown mirrors the audit report's
    impact section.
    """
    total_fn = sum(len(r["fn_lenses"]) for r in suspicious)
    total_fp = sum(len(r["fp_lenses"]) for r in suspicious)
    per_lens: dict[str, dict[str, int]] = {}
    for r in suspicious:
        for lens in r["fn_lenses"]:
            per_lens.setdefault(lens, {"fn": 0, "fp": 0})["fn"] += 1
        for lens in r["fp_lenses"]:
            per_lens.setdefault(lens, {"fn": 0, "fp": 0})["fp"] += 1
    return {
        "suspicious_examples": len(suspicious),
        "max_fn_flips": total_fn,
        "max_fp_flips": total_fp,
        "per_lens": per_lens,
    }


# ── Report rendering ─────────────────────────────────────────────

def _truncate(text: str, n: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def render_report(suspicious: list[dict], impact: dict, *, dataset_name: str,
                  eval_name: str, new_run_id: str) -> str:
    """Markdown audit scaffold: Confirmed / Borderline / False-Alarms + impact.

    The three sections start as a SCAFFOLD — every suspicious example is seeded
    into 'Borderline / Worth Reviewing' with both relabel directions spelled out;
    the human moves each row to Confirmed or False-Alarms after deciding. This
    mirrors the worked audit format (Confirmed Label Errors / Borderline /
    False Alarms / Impact on Eval Scores / Applied).
    """
    date = datetime.date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# Hold-Out Label Audit — {date}")
    lines.append("")
    lines.append(f"Dataset: `{dataset_name}`  ·  Eval: `{eval_name}`")
    lines.append(f"New run designation (use this id after any relabel): **{new_run_id}**")
    lines.append("")
    lines.append("> SCAFFOLD. Every disagreement is seeded into *Borderline* below. "
                 "Decide each one, then MOVE the row to *Confirmed Label Errors* "
                 "(the label is wrong — relabel) or *False Alarms* (the label is "
                 "right — the model erred). Do not relabel from this file without "
                 "a `--backup`. A label edit forces the new run designation above.")
    lines.append("")

    lines.append("## Confirmed Label Errors")
    lines.append("")
    lines.append("| # | ID | Lens | Current | Correct | Reason |")
    lines.append("|---|-----|------|---------|---------|--------|")
    lines.append("| _move confirmed rows here from Borderline_ | | | | | |")
    lines.append("")

    lines.append("## Borderline / Worth Reviewing")
    lines.append("")
    lines.append("| # | ID | Disagree | FN lenses (model PASS, gold FAIL) | "
                 "FP lenses (model FAIL, gold PASS) | Text |")
    lines.append("|---|-----|----------|-----------------------------------|"
                 "-----------------------------------|------|")
    for r in suspicious:
        ident = r["source_id"] or r["source"]
        fn = ", ".join(r["fn_lenses"]) or "-"
        fp = ", ".join(r["fp_lenses"]) or "-"
        lines.append(f"| {r['index']} | {ident} | {r['disagreements']} | {fn} "
                     f"| {fp} | {_truncate(r['text'])} |")
    if not suspicious:
        lines.append("| _no disagreements — gate and gold agree on every lens_ | | | | | |")
    lines.append("")

    lines.append("## False Alarms (NOT label errors)")
    lines.append("")
    lines.append("_Move rows here when the gold label is correct and the gate "
                 "erred (a model bug, not a data bug)._")
    lines.append("")

    lines.append("## Impact on Eval Scores")
    lines.append("")
    lines.append(f"- Suspicious examples (>=1 lens disagreement): "
                 f"**{impact['suspicious_examples']}**")
    lines.append(f"- Upper bound if EVERY suspicious label were a label error "
                 f"(it is not — decide per example): up to **{impact['max_fn_flips']} "
                 f"FN** and **{impact['max_fp_flips']} FP** would flip.")
    if impact["per_lens"]:
        lines.append("- Per-lens (max label-attributable, upper bound):")
        for lens in sorted(impact["per_lens"]):
            c = impact["per_lens"][lens]
            lines.append(f"  - `{lens}`: up to {c['fn']} FN, {c['fp']} FP phantom")
    lines.append("")
    lines.append("> These are UPPER BOUNDS. A disagreement is only phantom error "
                 "if the LABEL is wrong; the audit decides that per example. "
                 "Relabeling a model error would hide a real gate weakness.")
    lines.append("")

    lines.append("## Applied")
    lines.append("")
    lines.append("- Backup: `<dataset>.backup.<date>.json` (created with --backup)")
    lines.append("- Fixes applied: _date_")
    lines.append(f"- New run designation: **{new_run_id}** "
                 "(metrics from the OLD run id reflect the OLD labels — never "
                 "compare across this line)")
    return "\n".join(lines) + "\n"


# ── Backup discipline (mechanical enforcement) ───────────────────

def backup_dataset(dataset_path: str | Path) -> Path:
    """Make a dated backup copy: <dataset>.backup.<YYYYMMDD>.json. Never overwrites.

    A label edit must be preceded by a backup (the audit-report discipline). This
    is the only sanctioned way to make the dataset writable for relabeling.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    date = datetime.date.today().strftime("%Y%m%d")
    base = dataset_path.with_suffix("")
    suffix = dataset_path.suffix
    backup = Path(f"{base}.backup.{date}{suffix}")
    k = 2
    while backup.exists():
        backup = Path(f"{base}.backup.{date}-{k}{suffix}")
        k += 1
    shutil.copy2(dataset_path, backup)
    return backup


def run_label_audit(dataset_path: str | Path, eval_results_path: str | Path, *,
                    backup: bool = False, output_path: str | Path | None = None,
                    command_str: str = "lens-kit label-audit") -> int:
    """Build the audit scaffold + enforce backup discipline. Returns exit code.

        0 — audit report written (with a backup when --backup given)
        1 — backup requested but the dataset is missing (cannot back up)

    Without --backup the report is still written (auditing is read-only); the
    NEW-RUN-DESIGNATION rule is always printed so a relabel can't silently reuse
    a run id. --backup creates the dated copy that makes a relabel sanctioned.
    """
    dataset_path = Path(dataset_path)
    eval_results_path = Path(eval_results_path)

    gold = _load_gold(dataset_path)
    per_example = _load_eval_per_example(eval_results_path)
    suspicious = find_suspicious(gold, per_example)
    impact = impact_math(suspicious)

    from . import ledger
    new_run_id = ledger.new_run_id("relabel")

    backup_path = None
    if backup:
        backup_path = backup_dataset(dataset_path)
        print(f"Backup created (relabel now sanctioned): {backup_path}")
    else:
        print("NEW-RUN-DESIGNATION RULE: a label edit forces a new run id. "
              f"Use '{new_run_id}' after relabeling; never compare metrics "
              "across a label edit. Re-run with --backup before editing labels.")

    report = render_report(suspicious, impact, dataset_name=dataset_path.name,
                           eval_name=eval_results_path.name, new_run_id=new_run_id)
    if output_path is None:
        date = datetime.date.today().isoformat()
        output_path = Path.cwd() / f"label-audit-{dataset_path.stem}-{date}.md"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nAudit scaffold written: {output_path}")
    print(f"  suspicious examples: {impact['suspicious_examples']}  "
          f"(max {impact['max_fn_flips']} FN / {impact['max_fp_flips']} FP phantom)")

    ledger.append_run(
        run_id=ledger.new_run_id("label-audit"), command=command_str,
        checkpoint_sha=sha256_file(eval_results_path),
        data_file=dataset_path.name,
        metrics=f"suspicious={impact['suspicious_examples']}; "
                f"max_fn={impact['max_fn_flips']}; max_fp={impact['max_fp_flips']}; "
                f"next_run={new_run_id}",
        status="LABEL_AUDIT" + (" (backed up)" if backup_path else ""))
    print(f"Ledger row appended: {ledger.LEDGER_FILENAME} (LABEL_AUDIT)")
    return 0
