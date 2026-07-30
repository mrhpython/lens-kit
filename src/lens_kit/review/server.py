"""Tiny stdlib HTTP server for the review surface.

Serves the embedded-data HTML at ``/`` and round-trips human verdicts at
``/api/feedback`` (GET reads the saved file, POST validates + writes it).
No third-party deps; the only network surface is this localhost server.

Kept from the upstream eval-viewer: the GET/POST feedback handler, the
"validate {reviews: [...]} then write" contract, the static (no-server) HTML
export, and ``--port`` / ``--previous-feedback`` flags. Dropped: the
``find_runs`` workspace scan + ``lsof``-based port kill (we re-scan nothing
on each GET — records are fixed at startup — and bind an OS-chosen port via
``--port 0`` rather than killing whoever holds a fixed one).
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .parser import ArtifactError, build_records, load_artifacts
from .viewer import generate_html, load_previous_feedback


class ReviewHandler(BaseHTTPRequestHandler):
    """Serves the prebuilt review HTML and persists feedback.json."""

    def __init__(self, html: bytes, feedback_path: Path, *args, **kwargs):
        self._html = html
        self._feedback_path = feedback_path
        super().__init__(*args, **kwargs)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, self._html, "text/html; charset=utf-8")
        elif self.path == "/api/feedback":
            data = b"{}"
            if self._feedback_path.exists():
                data = self._feedback_path.read_bytes()
            self._send(200, data, "application/json")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/feedback":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            if not isinstance(data, dict) or "reviews" not in data:
                raise ValueError("expected a JSON object with a 'reviews' key")
            if not isinstance(data["reviews"], list):
                raise ValueError("'reviews' must be a list")
            self._feedback_path.write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8")
            self._send(200, b'{"ok":true}', "application/json")
        except (json.JSONDecodeError, OSError, ValueError) as e:
            self._send(500, json.dumps({"error": str(e)}).encode(),
                       "application/json")

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # keep the terminal clean


def _resolve_feedback_path(path: Path) -> Path:
    """Where feedback.json lives: in the dir, or beside a single file."""
    return (path if path.is_dir() else path.parent) / "feedback.json"


def run_review(
    results_path: str | Path,
    *,
    port: int = 0,
    previous_feedback: str | Path | None = None,
    static: str | Path | None = None,
    open_browser: bool = False,
) -> int:
    """Build the review page and either write it (static) or serve it.

    Returns a CLI exit code: 0 = served/written ok, 2 = the input could not
    be parsed into any record (ArtifactError).

    ``port=0`` binds an OS-chosen ephemeral port (the default, and what the
    tests rely on — no fixed-port binding). ``static`` writes the standalone
    HTML and returns without starting a server.
    """
    try:
        loaded = load_artifacts(results_path)
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    records, summaries = build_records(loaded)
    prev = load_previous_feedback(previous_feedback) if previous_feedback else {}
    label = Path(results_path).name
    html = generate_html(records, summaries, label=label, previous_feedback=prev)

    if static is not None:
        out = Path(static)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"Static review page written: {out}")
        return 0

    feedback_path = _resolve_feedback_path(Path(results_path))
    handler = partial(ReviewHandler, html.encode("utf-8"), feedback_path)
    server = HTTPServer(("127.0.0.1", port), handler)
    bound_port = server.server_address[1]
    url = f"http://127.0.0.1:{bound_port}"
    print("\n  lens-kit review")
    print("  ---------------------------------")
    print(f"  URL:       {url}")
    print(f"  Records:   {len(records)} across {len(summaries)} artifact(s)")
    print(f"  Feedback:  {feedback_path}")
    if prev:
        print(f"  Previous:  {previous_feedback} ({len(prev)} entries)")
    print("\n  Press Ctrl+C to stop.\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lens-kit review",
        description="Human review surface for lens-kit eval / calibration / "
                    "mutation artifacts (self-contained HTML + feedback.json).")
    p.add_argument("results", help="A results JSON file or a directory of them")
    p.add_argument("--port", type=int, default=0,
                   help="Server port (default: 0 = OS-chosen ephemeral)")
    p.add_argument("--previous-feedback", default=None,
                   help="A prior feedback.json to show as diff context")
    p.add_argument("--static", default=None,
                   help="Write standalone HTML to this path instead of serving")
    p.add_argument("--open", action="store_true",
                   help="Open the page in a browser (server mode only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    return run_review(
        args.results, port=args.port,
        previous_feedback=args.previous_feedback,
        static=args.static, open_browser=args.open,
    )


if __name__ == "__main__":
    sys.exit(main())
