#!/usr/bin/env python3
"""Serve a local UI for browsing structured terminal-bench results."""

from __future__ import annotations

import argparse
import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from terminalbench_results import delete_run, load_run, load_run_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(".benchmarks/terminalbench"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


class BenchUiHandler(BaseHTTPRequestHandler):
    server_version = "LashBenchUI/1.0"

    @property
    def results_dir(self) -> Path:
        return self.server.results_dir  # type: ignore[attr-defined]

    @property
    def html_path(self) -> Path:
        return self.server.html_path  # type: ignore[attr-defined]

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: object) -> None:
        self._send_bytes(status, "application/json; charset=utf-8", json.dumps(payload).encode())

    def _send_text(self, status: int, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
        self._send_bytes(status, content_type, body.encode())

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode() or "{}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(HTTPStatus.OK, self.html_path.read_text(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/runs":
            self._send_json(HTTPStatus.OK, {"runs": load_run_summaries(self.results_dir)})
            return
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/artifact"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 4:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "invalid artifact path"})
                return
            run_id = parts[2]
            query = parse_qs(parsed.query)
            relative_path = (query.get("path") or [None])[0]
            if not relative_path:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing artifact path"})
                return
            run_dir = (self.results_dir / "runs" / run_id).resolve()
            artifact_path = (run_dir / relative_path).resolve()
            if run_dir not in artifact_path.parents or not artifact_path.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "artifact not found"})
                return
            self._send_text(HTTPStatus.OK, artifact_path.read_text(errors="replace"))
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.removeprefix("/api/runs/").strip("/")
            run_dir = self.results_dir / "runs" / run_id
            if not run_dir.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
                return
            self._send_json(HTTPStatus.OK, load_run(run_dir))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/runs/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        run_id = self.path.removeprefix("/api/runs/").strip("/")
        if delete_run(self.results_dir, run_id):
            self._send_json(HTTPStatus.OK, {"deleted": [run_id]})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "run not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/runs/delete":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        body = self._read_json_body()
        run_ids = body.get("run_ids") or []
        if not isinstance(run_ids, list):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "run_ids must be a list"})
            return
        deleted = []
        for run_id in run_ids:
            if isinstance(run_id, str) and delete_run(self.results_dir, run_id):
                deleted.append(run_id)
        self._send_json(HTTPStatus.OK, {"deleted": deleted})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> int:
    ns = parse_args()
    ns.results_dir.mkdir(parents=True, exist_ok=True)
    html_path = Path(__file__).with_name("terminalbench_ui.html")
    handler = BenchUiHandler
    server = ThreadingHTTPServer((ns.host, ns.port), handler)
    server.results_dir = ns.results_dir.resolve()  # type: ignore[attr-defined]
    server.html_path = html_path  # type: ignore[attr-defined]
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}"
    print(f"Serving benchmark UI at {url}")
    if ns.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
