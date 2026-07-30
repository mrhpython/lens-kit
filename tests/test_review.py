"""Review surface — no network beyond localhost; no fixed ports.

Covers the four required areas from stage C5:
  1. HTML generation embeds the right records for each of the 3 artifact types
     (parse the generated HTML for expected ids/markers — no browser).
  2. feedback.json round-trip via the HTTP handler (POST then read).
  3. previous-feedback diff annotation.
  4. malformed-results error path (exit 2).
Plus artifact-type detection and CLI wiring.
"""
import json
import threading
import urllib.request
from functools import partial
from http.server import HTTPServer

import pytest

from lens_kit.cli import main
from lens_kit.review import (
    build_records, detect_artifact_type, generate_html, load_artifacts,
)
from lens_kit.review.parser import (
    ArtifactError, KIND_CALIBRATION, KIND_EVAL, KIND_MUTATION,
)
from lens_kit.review.server import ReviewHandler, run_review
from lens_kit.review.viewer import load_previous_feedback


# ── Fixtures: one artifact of each kind ───────────────────────────

def _eval_doc():
    return {
        "eval_schema_version": 1, "checkpoint": "baseline", "model": "m",
        "profile": "p", "total_examples": 2, "errors": 0,
        "summary": {"overall_accuracy": 0.9, "catch_rate": 0.8, "fp_rate": 0.1},
        "per_lens": {},
        "per_example": [
            {"index": 0, "source": "s", "source_id": "ex0", "domain": "general",
             "score": 1.0, "mismatches": [],
             "lens_confusion": {"truth": "tn", "rights": "tp"}, "elapsed": 0.2},
            {"index": 1, "source": "s", "source_id": "ex1", "domain": "general",
             "score": 0.5, "mismatches": ["truth: FN (expected FAIL, got PASS)"],
             "lens_confusion": {"truth": "fn"}, "elapsed": 0.3},
        ],
    }


def _calib_doc():
    return {
        "calib_schema_version": 1, "model": "m", "profile": "p", "battery_dir": "b",
        "summary": {"acceptable_rate": 1.0, "on_target_rate": 0.75,
                    "fp_rate": 0.0, "fn_rate": 0.0, "unknown_count": 0},
        "results": [
            {"fixture": "clean-00", "class": "clean", "verdict": "SHIP",
             "acceptable": True, "on_target": True, "halted": False,
             "lens_confusion": {"truth": "tn"}, "mismatches": []},
            {"fixture": "contradiction-00", "class": "contradiction",
             "verdict": "HOLD", "acceptable": True, "on_target": True,
             "halted": False, "lens_confusion": {"contradiction": "tp"},
             "mismatches": []},
        ],
    }


def _mutation_doc():
    return {
        "mutation_schema_version": 1, "model": "m", "profile": "p",
        "holdout": "h", "seed": 0, "total": 2, "missed": 1,
        "results": [
            {"id": "contradiction-01", "mutation_type": "contradiction",
             "verdict": "HOLD", "caught": True, "expected_one_of": ["HOLD"],
             "marker": "planted: X and not-X simultaneously"},
            {"id": "unsourced-01", "mutation_type": "unsourced",
             "verdict": "SHIP", "caught": False,
             "expected_one_of": ["HOLD", "NEEDS_SOURCES"],
             "marker": "planted: 47.3% uncited statistic"},
        ],
    }


@pytest.fixture
def eval_file(tmp_path):
    p = tmp_path / "eval-2026.json"
    p.write_text(json.dumps(_eval_doc()))
    return p


@pytest.fixture
def calib_file(tmp_path):
    p = tmp_path / "calib-2026.json"
    p.write_text(json.dumps(_calib_doc()))
    return p


@pytest.fixture
def mutation_file(tmp_path):
    p = tmp_path / "mutation-2026.json"
    p.write_text(json.dumps(_mutation_doc()))
    return p


# ── Artifact-type detection ───────────────────────────────────────

def test_detect_by_schema_version():
    assert detect_artifact_type(_eval_doc()) == KIND_EVAL
    assert detect_artifact_type(_calib_doc()) == KIND_CALIBRATION
    assert detect_artifact_type(_mutation_doc()) == KIND_MUTATION


def test_detect_structural_fallback_no_schema_version():
    assert detect_artifact_type({"results": [{"caught": True, "marker": "m"}]}) == KIND_MUTATION
    assert detect_artifact_type({"results": [{"acceptable": True, "class": "clean"}]}) == KIND_CALIBRATION
    # per_example alone classifies as eval even without the schema key.
    assert detect_artifact_type({"per_example": []}) == KIND_EVAL


def test_detect_rejects_unknown_and_non_dict():
    assert detect_artifact_type({"hello": "world"}) is None
    assert detect_artifact_type([1, 2, 3]) is None
    assert detect_artifact_type("nope") is None


# ── 1. HTML generation embeds the right records per artifact type ──

def test_html_embeds_eval_records_and_summary(eval_file):
    loaded = load_artifacts(eval_file)
    recs, summ = build_records(loaded)
    html = generate_html(recs, summ, label="t")
    # Per-example record ids present.
    assert "eval:eval-2026#0" in html
    assert "eval:eval-2026#1" in html
    # Mismatch string + per-lens confusion verdicts embedded.
    assert "truth: FN (expected FAIL, got PASS)" in html
    assert '"truth": "fn"' in html
    # Summary metrics embedded.
    assert '"overall_accuracy": 0.9' in html
    assert '"catch_rate": 0.8' in html
    assert "EMBEDDED_DATA" in html


def test_html_embeds_calibration_records(calib_file):
    loaded = load_artifacts(calib_file)
    recs, summ = build_records(loaded)
    html = generate_html(recs, summ, label="t")
    assert "calib:calib-2026#clean-00" in html
    assert "calib:calib-2026#contradiction-00" in html
    # verdict vs acceptable/target rendered as data.
    assert '"verdict": "SHIP"' in html
    assert '"acceptable": true' in html
    assert '"on_target": true' in html


def test_html_embeds_mutation_marker_and_caught(mutation_file):
    loaded = load_artifacts(mutation_file)
    recs, summ = build_records(loaded)
    html = generate_html(recs, summ, label="t")
    assert "mut:mutation-2026#contradiction-01" in html
    assert "mut:mutation-2026#unsourced-01" in html
    # Planted-defect marker embedded verbatim.
    assert "planted: X and not-X simultaneously" in html
    assert "planted: 47.3% uncited statistic" in html
    # caught / missed flags embedded.
    assert '"caught": true' in html
    assert '"caught": false' in html


def test_html_directory_loads_all_three_kinds(eval_file, calib_file, mutation_file):
    d = eval_file.parent
    loaded = load_artifacts(d)
    recs, summ = build_records(loaded)
    kinds = {s["kind"] for s in summ}
    assert kinds == {KIND_EVAL, KIND_CALIBRATION, KIND_MUTATION}
    html = generate_html(recs, summ, label=d.name)
    assert "eval:eval-2026#0" in html
    assert "calib:calib-2026#clean-00" in html
    assert "mut:mutation-2026#unsourced-01" in html


def test_html_escapes_script_close_in_embedded_data(tmp_path):
    """A mutant marker containing </script> must not break the inline script."""
    doc = _mutation_doc()
    doc["results"][0]["marker"] = "evil </script><img> payload"
    p = tmp_path / "mutation-evil.json"
    p.write_text(json.dumps(doc))
    recs, summ = build_records(load_artifacts(p))
    html = generate_html(recs, summ, label="t")
    assert "</script><img>" not in html                    # not raw
    assert "\\u003c/script>\\u003cimg> payload" in html    # escaped form survives


def test_html_escapes_comment_open_script_double_escape(tmp_path):
    """'<!--' + '<script' in artifact content must not reach the tokenizer raw.

    That pair puts an HTML parser into script-data-double-escaped state, where
    the page's real </script> terminator is swallowed and the page goes blank
    (availability, not XSS). Every '<' in the embedded data is escaped, so
    neither sequence can appear raw.
    """
    doc = _mutation_doc()
    doc["results"][0]["marker"] = "quoting html: <!-- <script> oops"
    p = tmp_path / "mutation-comment.json"
    p.write_text(json.dumps(doc))
    recs, summ = build_records(load_artifacts(p))
    html = generate_html(recs, summ, label="t")
    _, _, tail = html.partition("const EMBEDDED_DATA")
    embedded_segment = tail.partition("</script>")[0]
    assert "<!--" not in embedded_segment      # no raw comment-open
    assert "<script" not in embedded_segment   # no raw script-open
    assert "\\u003c!-- \\u003cscript> oops" in html   # value preserved


def test_mutation_text_persisted_and_rendered(tmp_path, stub_gate_factory):
    """run_mutant persists the mutated body; the review embeds it (FIX-1)."""
    from lens_kit.mutation import run_mutant

    mutant = {
        "id": "fabricated_stat-01", "mutation_type": "fabricated_stat",
        "caught_verdicts": ["HOLD", "NEEDS_SOURCES"], "domain": "general",
        "context": "", "marker": "planted: 12.34% fabricated",
        "text": "The mutated body the human must be able to read.",
    }
    row = run_mutant(mutant, stub_gate_factory())
    assert row["text"] == mutant["text"]

    doc = _mutation_doc()
    doc["results"][0]["text"] = "mutant body in artifact"
    p = tmp_path / "mutation-text.json"
    p.write_text(json.dumps(doc))
    recs, summ = build_records(load_artifacts(p))
    mut_rec = next(r for r in recs if r["id"].endswith("contradiction-01"))
    assert mut_rec["data"]["text"] == "mutant body in artifact"
    html = generate_html(recs, summ, label="t")
    assert "mutant body in artifact" in html


def test_envelope_block_surfaced_when_present(tmp_path):
    env = {
        "eval_schema_version": 1, "date": "2026-06-12", "checkpoint": "baseline",
        "reruns": 3, "fresh_cache": True, "cache_caveat": None,
        "run_files": [], "run_summaries": [],
        "metrics": {"overall_accuracy": {"mean": 0.88, "stddev": 0.01,
                                         "min": 0.87, "max": 0.89, "worst_of_n": 0.87}},
    }
    p = tmp_path / "eval-2026-envelope.json"
    p.write_text(json.dumps(env))
    recs, summ = build_records(load_artifacts(p))
    assert summ[0]["envelope"] is not None
    assert summ[0]["envelope"]["reruns"] == 3
    html = generate_html(recs, summ, label="t")
    assert '"worst_of_n": 0.87' in html


# ── 2. feedback.json round-trip via the HTTP handler (POST then read) ──

def _serve_one(html_bytes, feedback_path):
    """Bind an ephemeral port, handle exactly one request, return the port."""
    handler = partial(ReviewHandler, html_bytes, feedback_path)
    srv = HTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.handle_request, daemon=True)
    t.start()
    return srv, port, t


def test_feedback_post_then_read_roundtrip(eval_file):
    recs, summ = build_records(load_artifacts(eval_file))
    html = generate_html(recs, summ, label="t").encode()
    feedback_path = eval_file.parent / "feedback.json"

    # POST a review (one request).
    srv, port, t = _serve_one(html, feedback_path)
    payload = json.dumps({
        "reviews": [{"record_id": "eval:eval-2026#1", "kind": "eval",
                     "feedback": "gate is wrong here"}],
        "status": "in_progress",
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/feedback", data=payload,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=5)
    assert json.loads(resp.read()) == {"ok": True}
    t.join(timeout=5)
    srv.server_close()

    # File written with the review.
    saved = json.loads(feedback_path.read_text())
    assert saved["reviews"][0]["record_id"] == "eval:eval-2026#1"
    assert saved["reviews"][0]["feedback"] == "gate is wrong here"

    # GET reads it back over a fresh ephemeral-port request.
    srv, port, t = _serve_one(html, feedback_path)
    got = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/api/feedback", timeout=5).read()
    t.join(timeout=5)
    srv.server_close()
    assert json.loads(got)["reviews"][0]["record_id"] == "eval:eval-2026#1"


def test_feedback_post_rejects_malformed_body(eval_file):
    feedback_path = eval_file.parent / "feedback.json"
    srv, port, t = _serve_one(b"<html></html>", feedback_path)
    # 'reviews' key missing -> 500, file not written.
    bad = json.dumps({"notreviews": 1}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/feedback", data=bad,
        headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 500
    t.join(timeout=5)
    srv.server_close()
    assert not feedback_path.exists()


def test_get_serves_html_at_root(eval_file):
    feedback_path = eval_file.parent / "feedback.json"
    srv, port, t = _serve_one(b"<html>hello-review</html>", feedback_path)
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
    t.join(timeout=5)
    srv.server_close()
    assert b"hello-review" in body


# ── 3. previous-feedback diff annotation ──────────────────────────

def test_previous_feedback_loaded_and_annotated(eval_file, tmp_path):
    prev = tmp_path / "prev-feedback.json"
    prev.write_text(json.dumps({"reviews": [
        {"record_id": "eval:eval-2026#0", "feedback": "looked fine last time"},
        {"run_id": "legacy-id", "feedback": "legacy key still read"},
        {"record_id": "blank", "feedback": "   "},  # dropped
    ]}))
    pm = load_previous_feedback(prev)
    assert pm["eval:eval-2026#0"] == "looked fine last time"
    assert pm["legacy-id"] == "legacy key still read"   # run_id fallback
    assert "blank" not in pm                              # empty dropped

    recs, summ = build_records(load_artifacts(eval_file))
    html = generate_html(recs, summ, label="t", previous_feedback=pm)
    assert "looked fine last time" in html
    # The diff-context section + suppress-prefill logic exist in the template.
    assert "previous_feedback" in html
    assert "Previous feedback" in html


def test_previous_feedback_missing_file_is_empty(tmp_path):
    assert load_previous_feedback(tmp_path / "nope.json") == {}


# ── 4. malformed-results error path (exit 2) ──────────────────────

def test_malformed_json_file_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    rc = main(["review", str(bad), "--static", str(tmp_path / "out.html")])
    assert rc == 2
    assert "could not load artifacts" in capsys.readouterr().err


def test_unrecognised_shape_exits_2(tmp_path, capsys):
    unk = tmp_path / "unk.json"
    unk.write_text(json.dumps({"hello": "world"}))
    rc = main(["review", str(unk)])
    assert rc == 2
    assert "not a recognised lens-kit artifact" in capsys.readouterr().err


def test_missing_path_exits_2(tmp_path, capsys):
    rc = main(["review", str(tmp_path / "absent.json")])
    assert rc == 2
    assert "results path not found" in capsys.readouterr().err


def test_empty_directory_exits_2(tmp_path, capsys):
    rc = main(["review", str(tmp_path)])
    assert rc == 2
    assert "no .json artifacts" in capsys.readouterr().err


def test_load_artifacts_raises_artifacterror_on_bad_input(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")
    with pytest.raises(ArtifactError):
        load_artifacts(bad)


# ── CLI wiring + static export (no port bound) ────────────────────

def test_review_static_export_writes_html(eval_file, calib_file, mutation_file, capsys):
    d = eval_file.parent
    out = d / "review.html"
    rc = main(["review", str(d), "--static", str(out)])
    assert rc == 0
    assert out.exists()
    html = out.read_text()
    assert "eval:eval-2026#0" in html
    assert "calib:calib-2026#clean-00" in html
    assert "mut:mutation-2026#contradiction-01" in html
    assert "Static review page written" in capsys.readouterr().out


def test_run_review_static_returns_zero(eval_file, tmp_path):
    rc = run_review(eval_file, static=tmp_path / "o.html")
    assert rc == 0
    assert (tmp_path / "o.html").exists()


def test_feedback_json_excluded_from_artifact_scan(eval_file):
    # A feedback.json in the dir must not be mistaken for an artifact.
    (eval_file.parent / "feedback.json").write_text(json.dumps({"reviews": []}))
    loaded = load_artifacts(eval_file.parent)
    sources = {f.name for f, _, _ in loaded}
    assert "feedback.json" not in sources
    assert "eval-2026.json" in sources
