"""HTML generation for the review surface.

``generate_html`` embeds the parsed records + summaries (+ optional previous
feedback) into the self-contained viewer template by replacing the
``/*__EMBEDDED_DATA__*/`` marker — the same single-injection-point pattern as
the upstream eval-viewer, so the resulting page works offline (and the only
runtime fetch is the optional localhost feedback endpoint).
"""
from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = Path(__file__).with_name("viewer.html")
_MARKER = "/*__EMBEDDED_DATA__*/"


def generate_html(
    records: list[dict],
    summaries: list[dict],
    *,
    label: str = "",
    previous_feedback: dict[str, str] | None = None,
) -> str:
    """Render the standalone review page with all data embedded.

    ``previous_feedback`` maps record_id -> prior feedback text; non-empty
    entries are shown as diff context next to the current record (and suppress
    pre-fill from a stale on-disk feedback.json, matching the upstream rule).
    """
    template = _TEMPLATE.read_text(encoding="utf-8")
    embedded = {
        "label": label,
        "records": records,
        "summaries": summaries,
        "previous_feedback": previous_feedback or {},
    }
    data_json = json.dumps(embedded)
    # Artifact strings (mutant text, markers, mismatches) can contain markup:
    # "</script>" would terminate the inline script early, and "<!--" followed
    # by "<script" puts an HTML tokenizer into script-data-double-escaped
    # state, swallowing the real terminator (blank page). Escape EVERY "<"
    # as the six-char sequence backslash-u003c — a valid JSON escape AND a
    # valid JS one, so string values are byte-identical after parsing while
    # the raw HTML carries no "<" inside the embedded data at all.
    data_json = data_json.replace("<", "\\u003c")
    return template.replace(_MARKER, f"const EMBEDDED_DATA = {data_json};")


def load_previous_feedback(path: str | Path) -> dict[str, str]:
    """Load a previous feedback.json into a record_id -> feedback map.

    Tolerant of the upstream key name: rows use ``record_id`` here, but a
    file written by the original eval-viewer uses ``run_id``. Empty feedback
    strings are dropped so they don't show as spurious diff context.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, str] = {}
    for r in data.get("reviews", []) if isinstance(data, dict) else []:
        if not isinstance(r, dict):
            continue
        rid = r.get("record_id") or r.get("run_id")
        fb = r.get("feedback", "")
        if rid and isinstance(fb, str) and fb.strip():
            out[rid] = fb
    return out
