# lens_stop_hook/transcript.py
"""Extract the text of the final assistant turn from a Claude Code transcript.

Walks the JSONL from the end, collecting text blocks from the trailing run of
`assistant` lines, stopping at the first `user` line (the turn boundary).
Tool-use blocks are ignored. Any parse problem fails soft to "".
"""
from __future__ import annotations

import json


def final_assistant_text(transcript_path: str) -> str:
    try:
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return ""

    collected: list[str] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        t = obj.get("type")
        if t == "user":
            break  # turn boundary
        if t != "assistant":
            continue
        content = (obj.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        texts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
        if texts:
            collected.append("\n".join(texts))

    collected.reverse()  # we walked backwards; restore document order
    return "\n".join(collected)
