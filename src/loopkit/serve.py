"""A local page for trying one Loop. Programs call Loop.ask instead."""

from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from loopkit.errors import LoopError
from loopkit.loop import Loop

MAX_BODY_BYTES = 1_000_000

PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>loop-kit</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: #f4f1ea;
    color: #1d1a16;
    font-family: "Iowan Old Style", Palatino, Georgia, serif;
  }
  main { max-width: 40rem; margin: 0 auto; padding: 3.5rem 1.25rem 5rem; }
  h1 { font-weight: 500; font-size: 1.8rem; letter-spacing: -0.03em; margin: 0; }
  .lede { color: #5c564c; margin: 0.6rem 0 1.6rem; line-height: 1.45; }
  label { display: block; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.8rem; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.4rem; }
  textarea {
    width: 100%;
    min-height: 7.5rem;
    resize: vertical;
    font: 1rem/1.5 ui-sans-serif, system-ui, sans-serif;
    padding: 0.8rem 0.85rem;
    border: 1px solid #d4cbbd;
    border-radius: 8px;
    background: #fffdf8;
    color: inherit;
  }
  button {
    margin-top: 0.75rem;
    font: 600 0.95rem/1 ui-sans-serif, system-ui, sans-serif;
    background: #1d1a16;
    color: #f4f1ea;
    border: 0;
    border-radius: 8px;
    padding: 0.75rem 1.05rem;
    cursor: pointer;
  }
  button:disabled { opacity: 0.55; cursor: progress; }
  textarea:focus, button:focus-visible { outline: 2px solid #1d1a16; outline-offset: 2px; }
  #error { color: #8a2b2b; font-family: ui-sans-serif, system-ui, sans-serif; }
  article { margin-top: 2rem; font-size: 1.12rem; line-height: 1.55; white-space: pre-wrap; }
  .meta { color: #5c564c; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.85rem; }
  ul { padding-left: 1.1rem; margin: 0.4rem 0 0; }
  li { margin: 0.35rem 0; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.9rem; }
</style>
</head>
<body>
<main>
  <h1>loop-kit</h1>
  <p class="lede">Try the loop loaded in this process. In a program, call Loop.ask.</p>
  <form id="ask">
    <label for="task">Question</label>
    <textarea id="task" name="task" placeholder="What is the refund window?" required></textarea>
    <button type="submit">Ask</button>
  </form>
  <p id="error" hidden role="alert"></p>
  <article id="answer" hidden aria-live="polite"></article>
  <p id="meta" class="meta" hidden></p>
  <ul id="sources"></ul>
  <noscript><p>This page needs JavaScript. You can also POST JSON to /ask.</p></noscript>
</main>
<script>
const form = document.querySelector("#ask");
const task = document.querySelector("#task");
const button = form.querySelector("button");
const error = document.querySelector("#error");
const answer = document.querySelector("#answer");
const meta = document.querySelector("#meta");
const sources = document.querySelector("#sources");

task.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.hidden = true;
  button.disabled = true;
  button.textContent = "Drafting and checking…";
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: task.value }),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error || "The request failed.");
    }
    answer.hidden = false;
    answer.textContent = body.text;
    const turns = body.turns === 1 ? "1 turn" : body.turns + " turns";
    meta.hidden = false;
    meta.textContent = body.stopped_reason === "passed"
      ? "Checked in " + turns + "."
      : "Stopped after " + turns + ".";
    sources.replaceChildren();
    for (const citation of body.citations) {
      const item = document.createElement("li");
      const label = citation.heading ? citation.source + " — " + citation.heading : citation.source;
      item.textContent = "[" + citation.id + "] " + label;
      sources.appendChild(item);
    }
  } catch (err) {
    error.hidden = false;
    error.textContent = err && err.message ? err.message : "The request failed.";
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
});
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    loop: Loop
    server_version = "loopkit"

    def log_message(self, format: str, *args: object) -> None:
        if self.loop.trace:
            super().log_message(format, *args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"ok": True})
            return
        if path in {"/", "/index.html"}:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            return
        self._send(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/ask":
            self._send(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "Content-Length must be an integer."})
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._send(413, {"error": "Request body is too large."})
            return
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": "Send a JSON object with a task string."})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("task"), str):
            self._send(400, {"error": "Send a JSON object with a task string."})
            return
        try:
            answer = self.loop.ask(payload["task"])
        except LoopError as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception:  # noqa: BLE001 - a request handler must never take the server down
            # Loop already wraps model failures in LoopError (handled above). Anything else
            # reaching here is a bug; log the traceback and return a generic 500 rather than
            # leaking internals to the client or killing the serving thread.
            traceback.print_exc()
            self._send(500, {"error": "The model call failed."})
            return
        self._send(200, answer.to_dict())

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def make_handler(loop: Loop) -> type[_Handler]:
    class Handler(_Handler):
        pass

    Handler.loop = loop
    return Handler


def make_server(loop: Loop, host: str, port: int) -> HTTPServer:
    try:
        return HTTPServer((host, port), make_handler(loop))
    except OSError as exc:
        raise LoopError(f"Could not listen on {host}:{port}. {exc}") from exc


def server_url(server: HTTPServer) -> str:
    """The ``http://host:port`` the server is actually bound to (port 0 picks a free one)."""

    bound_host, bound_port = server.server_address[:2]
    host_text = bound_host.decode() if isinstance(bound_host, bytes) else str(bound_host)
    return f"http://{host_text}:{bound_port}"


def serve(loop: Loop, host: str = "127.0.0.1", port: int = 8080) -> None:
    server = make_server(loop, host, port)
    print(f"loopkit listening on {server_url(server)}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
