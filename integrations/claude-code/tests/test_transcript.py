# tests/test_transcript.py
import json
from lens_stop_hook.transcript import final_assistant_text


def _write(tmp_path, rows):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def _asst(text=None, tool=False):
    content = []
    if text is not None:
        content.append({"type": "text", "text": text})
    if tool:
        content.append({"type": "tool_use", "id": "x", "name": "n", "input": {}})
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


def test_returns_last_assistant_turn_text(tmp_path):
    path = _write(tmp_path, [
        {"type": "user", "message": {"role": "user", "content": "q1"}},
        _asst("first answer"),
        {"type": "user", "message": {"role": "user", "content": "q2"}},
        _asst("looking...", tool=True),
        _asst("final answer"),
    ])
    assert final_assistant_text(path) == "looking...\nfinal answer"


def test_tool_only_last_entry_still_finds_turn_text(tmp_path):
    path = _write(tmp_path, [
        {"type": "user", "message": {"role": "user", "content": "q"}},
        _asst("the answer"),
        _asst(tool=True),  # trailing tool_use entry, no text
    ])
    assert final_assistant_text(path) == "the answer"


def test_no_assistant_text_returns_empty(tmp_path):
    path = _write(tmp_path, [
        {"type": "user", "message": {"role": "user", "content": "q"}},
        _asst(tool=True),
    ])
    assert final_assistant_text(path) == ""


def test_missing_file_returns_empty():
    assert final_assistant_text("/no/such/file.jsonl") == ""
