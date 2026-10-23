"""A local page for trying one Loop. Programs call Loop.ask instead.

The page is ``ui/index.html`` next to this module. It talks to three routes:

* ``GET /config`` says what this process loaded (provider, model, mode, the
  instructions, the knowledge base) and which provider keys are set in the
  environment. Key values are never sent, only whether each one is present.
* ``POST /ask`` takes ``{"task": "..."}`` and, optionally, ``"history"`` and
  ``"options"``. ``history`` is the earlier turns of the chat as
  ``[[ask, answer], ...]``, oldest first; :func:`compose_task` folds them
  into the one task string the loop receives, and the response says what
  that string was under ``"task"``. ``options`` are the :data:`OPTION_KEYS`
  below: they build a :class:`Loop` for that one request on top of the
  knowledge base that is already in memory, so the page can try the loop's
  settings without restarting the server.
* ``GET /session`` and ``PUT /session`` read and write the page's state
  (tabs, their turns, their options) when the server was started with a
  session file (``loopkit serve --session``). The page loads it first on a
  refresh, so a chat survives closing the browser. :class:`SessionStore`
  checks the shape and writes the file atomically.
* ``GET /health`` returns ``{"ok": true}``.

The loop itself has no memory, and this server keeps none in the process
between requests: the page holds the turns of each chat and sends them with
the next ask. A session file is the only thing that outlives a request.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse

from loopkit import __version__
from loopkit.client import _KEY_ENV, DEFAULT_MODELS
from loopkit.config import CHECK_MODES, RESPONSES
from loopkit.errors import LoopError, LoopWarning
from loopkit.knowledge import Knowledge
from loopkit.loop import Loop, default_instructions

MAX_BODY_BYTES = 1_000_000
MAX_SESSION_BYTES = 50_000_000
"""A session holds every answer of every tab; book-length answers add up."""

DEFAULT_SESSION_FILE = "loopkit-session.json"
SESSION_VERSION = 1
MESSAGE_ROLES = ("user", "assistant", "error")

PAGE = files("loopkit").joinpath("ui/index.html").read_text(encoding="utf-8")

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
    const tokens = body.tokens && body.tokens.total
      ? " " + body.tokens.total.toLocaleString() + " tokens."
      : "";
    meta.textContent = (body.stopped_reason === "passed"
      ? "Checked in " + turns + "."
      : body.stopped_reason === "token_budget"
        ? "Stopped after " + turns + ": token budget reached."
        : "Stopped after " + turns + ".") + tokens;
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

KNOWLEDGE_SOURCES = ("project", "none", "text")
"""Where a request's passages come from.

``project`` is the knowledge base this server loaded. ``none`` runs the loop
in open mode. ``text`` reads ``knowledge_text`` from the request as the one
document.
"""

OPTION_KEYS = frozenset(
    {
        "knowledge_source",
        "knowledge_text",
        "instructions",
        "provider",
        "model",
        "base_url",
        "max_turns",
        "top_k",
        "context_budget",
        "check_mode",
        "response",
        "search",
        "auto_optimize",
    }
)
"""Settings ``POST /ask`` accepts under ``options``. Anything else is a 400."""


def compose_task(task: str, history: Sequence[tuple[str, str]] | None) -> str:
    """The one task string the loop receives for a turn of a chat.

    The first ask goes through as typed. A follow-up quotes the earlier
    turns, most recent first, each as a ``(ask, final answer)`` pair::

        User asked: <task>; Previous conversation: [("<ask n-1>", "<answer n-1>"), ("<ask n-2>", "<answer n-2>"), ...]

    Asks and answers are JSON-quoted, so commas, quotes, and line breaks
    inside them cannot be mistaken for the list's own punctuation.
    """

    if not history:
        return task
    pairs = ", ".join(
        f"({json.dumps(ask, ensure_ascii=False)}, {json.dumps(answer, ensure_ascii=False)})"
        for ask, answer in reversed(history)
    )
    return f"User asked: {task}; Previous conversation: [{pairs}]"


def parse_history(value: object) -> list[tuple[str, str]]:
    """``history`` from a request body: a list of ``[ask, answer]`` string pairs."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise LoopError("history must be a list of [ask, answer] pairs.")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
        ):
            raise LoopError("history must be a list of [ask, answer] pairs of strings.")
        pairs.append((item[0], item[1]))
    return pairs


class SessionStore:
    """The page's state in one JSON file, so a refresh shows the same chats.

    The page is the source of truth while it is open: after each change it
    sends its whole state, and :meth:`save` checks the shape and replaces the
    file in one step (write to a temporary file, then rename), so a crash
    mid-write cannot leave a half-written session. :meth:`load` returns
    ``None`` when there is no file yet.

    What is stored is what the page needs to redraw: each tab's label, title,
    options, messages, and the ``[ask, answer]`` pairs it sends as
    ``history``. Answers are kept as the ``/ask`` response bodies, so the
    audit trail (role, citations, verdicts) survives too.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def load(self) -> dict[str, object] | None:
        if not self.path.is_file():
            return None
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LoopError(f"Could not read the session file {self.path}: {exc}") from exc
        if not raw.strip():
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LoopError(f"The session file {self.path} is not valid JSON: {exc}") from exc
        return validate_session(data)

    def save(self, state: object) -> dict[str, object]:
        checked = validate_session(state)
        checked["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        body = json.dumps(checked, ensure_ascii=False, indent=1)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(
                prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    file.write(body)
                os.replace(temp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temp)
                raise
        except OSError as exc:
            raise LoopError(f"Could not write the session file {self.path}: {exc}") from exc
        return checked


def validate_session(data: object) -> dict[str, object]:
    """The page's state, checked for shape. Unknown fields inside a tab are kept.

    Raises :class:`LoopError` naming the first problem, so ``PUT /session``
    can answer 400 and a corrupt file is reported, not silently dropped.
    """

    if not isinstance(data, dict):
        raise LoopError("A session must be a JSON object.")
    version = data.get("version", SESSION_VERSION)
    if version != SESSION_VERSION:
        raise LoopError(
            f"Unsupported session version {version!r}; this server writes version {SESSION_VERSION}."
        )
    tabs = data.get("tabs")
    if not isinstance(tabs, list):
        raise LoopError("A session needs a tabs list.")
    ids: set[int] = set()
    checked_tabs: list[dict[str, object]] = []
    for index, tab in enumerate(tabs, start=1):
        if not isinstance(tab, dict):
            raise LoopError(f"Tab {index} must be an object.")
        tab_id = tab.get("id")
        if isinstance(tab_id, bool) or not isinstance(tab_id, int):
            raise LoopError(f"Tab {index} needs an integer id.")
        if tab_id in ids:
            raise LoopError(f"Tab id {tab_id} is used twice.")
        ids.add(tab_id)
        for key in ("kind", "label", "title"):
            if not isinstance(tab.get(key), str):
                raise LoopError(f"Tab {index} needs a string {key}.")
        if not isinstance(tab.get("options"), dict):
            raise LoopError(f"Tab {index} needs an options object.")
        messages = tab.get("messages")
        if not isinstance(messages, list):
            raise LoopError(f"Tab {index} needs a messages list.")
        for position, message in enumerate(messages, start=1):
            if not isinstance(message, dict) or message.get("role") not in MESSAGE_ROLES:
                roles = ", ".join(MESSAGE_ROLES)
                raise LoopError(f"Message {position} of tab {index} needs a role: {roles}.")
            if message["role"] == "assistant" and not isinstance(message.get("answer"), dict):
                raise LoopError(
                    f"Message {position} of tab {index} is an answer without an answer object."
                )
            if message["role"] != "assistant" and not isinstance(message.get("text"), str):
                raise LoopError(f"Message {position} of tab {index} needs a text string.")
        try:
            history = parse_history(tab.get("history", []))
        except LoopError as exc:
            raise LoopError(f"Tab {index}: {exc}") from exc
        checked_tabs.append({**tab, "history": [list(pair) for pair in history]})
    active = data.get("active_id")
    if active is not None and (isinstance(active, bool) or not isinstance(active, int)):
        raise LoopError("active_id must be an integer tab id or null.")
    if active is not None and active not in ids:
        active = checked_tabs[0]["id"] if checked_tabs else None
    return {"version": SESSION_VERSION, "tabs": checked_tabs, "active_id": active}


def settings(loop: Loop) -> dict[str, object]:
    """The settings a loop is running with, in the shape ``/config`` and ``/ask`` report."""

    return {
        "provider": loop.provider,
        "model": loop.model,
        "mode": loop.mode,
        "instructions": list(loop.instructions),
        "check_mode": loop.check_mode,
        "response": loop.response,
        "max_turns": loop.max_turns,
        "top_k": loop.top_k,
        "context_budget": loop.context_budget,
        "search": loop.search,
        "auto_optimize": loop.auto_optimize,
    }


def knowledge_summary(knowledge: Knowledge) -> dict[str, object]:
    sources: list[str] = []
    for passage in knowledge.passages:
        if passage.source not in sources:
            sources.append(passage.source)
    return {
        "passages": len(knowledge.passages),
        "sources": sources,
        "warnings": list(knowledge.warnings),
    }


def provider_status() -> dict[str, dict[str, object]]:
    """Which providers this process could call. Reports whether a key is set, never the key."""

    status: dict[str, dict[str, object]] = {}
    for name in ("xai", "openai", "anthropic"):
        variable = _KEY_ENV[name]
        status[name] = {
            "env": variable,
            "key_set": bool(os.environ.get(variable, "").strip()),
            "default_model": DEFAULT_MODELS[name],
        }
    base_url = os.environ.get("LOOPKIT_BASE_URL", "").strip() or None
    status["compatible"] = {
        "env": "LOOPKIT_API_KEY",
        "key_set": any(
            os.environ.get(v, "").strip() for v in ("LOOPKIT_API_KEY", "OPENAI_API_KEY")
        ),
        "default_model": os.environ.get("LOOPKIT_MODEL", "").strip() or "",
        "base_url": base_url,
    }
    return status


def config(loop: Loop, session: SessionStore | None = None) -> dict[str, object]:
    """The body of ``GET /config``."""

    return {
        "version": __version__,
        "session": {
            "enabled": session is not None,
            "path": None if session is None else str(session.path),
        },
        "loop": {
            **settings(loop),
            "knowledge": knowledge_summary(loop.knowledge),
            "debug_path": None if loop.debug_path is None else str(loop.debug_path),
            "trace": loop.trace,
        },
        "providers": provider_status(),
        "default_instructions": [
            {
                "mode": mode,
                "response": response,
                "search": search,
                "instructions": list(default_instructions(mode, response, search)),
            }
            for mode in ("knowledge", "open")
            for response in RESPONSES
            for search in (False, True)
        ],
        "option_keys": sorted(OPTION_KEYS),
        "knowledge_sources": list(KNOWLEDGE_SOURCES),
    }


def loop_for(base: Loop, options: Mapping[str, object] | None) -> Loop:
    """The loop one request should run on.

    With no options this is ``base``. Otherwise a new :class:`Loop` is built
    from ``base``'s knowledge (already read; no disk access) and the given
    settings. A setting that is missing keeps the server's value, which is
    what the page shows in its form; ``base_url`` without a ``provider``
    implies ``compatible``, as it does for :class:`Loop`. Bad values raise
    :class:`LoopError` so the handler answers 400 with the reason.
    """

    if not options:
        return base
    unknown = sorted(set(options) - OPTION_KEYS)
    if unknown:
        raise LoopError(
            f"Unknown option: {', '.join(unknown)}. Allowed options: {', '.join(sorted(OPTION_KEYS))}."
        )

    source = _choice(
        options.get("knowledge_source"), "project", "knowledge_source", KNOWLEDGE_SOURCES
    )
    if source == "project":
        knowledge: Knowledge = base.knowledge
    elif source == "none":
        knowledge = Knowledge.empty()
    else:
        text = _optional_str(options.get("knowledge_text"), "knowledge_text")
        if text is None or not text.strip():
            raise LoopError(
                "knowledge_text is empty. Paste some text, or choose another knowledge source."
            )
        knowledge = Knowledge.load(text)

    instructions = _instructions(options.get("instructions"))
    kwargs: dict[str, object] = {
        "max_turns": _positive_int(options.get("max_turns"), base.max_turns, "max_turns"),
        "top_k": _positive_int(options.get("top_k"), base.top_k, "top_k"),
        "context_budget": _positive_int(
            options.get("context_budget"), base.context_budget, "context_budget"
        ),
        "check_mode": _choice(
            options.get("check_mode"), base.check_mode, "check_mode", CHECK_MODES
        ),
        "response": _choice(options.get("response"), base.response, "response", RESPONSES),
        "search": _bool(options.get("search"), base.search, "search"),
        "auto_optimize": _bool(options.get("auto_optimize"), base.auto_optimize, "auto_optimize"),
        "trace": base.trace,
    }
    if base.debug_path is not None:
        kwargs["debug"] = base.debug_path

    provider = _optional_str(options.get("provider"), "provider")
    model = _optional_str(options.get("model"), "model")
    base_url = _optional_str(options.get("base_url"), "base_url")
    if base.provider == "custom":
        # The server was given a client object (tests do this). Keep calling it;
        # a different provider cannot be built from here.
        if (provider and provider != "custom") or base_url:
            raise LoopError(
                "This server runs on a custom client. provider and base_url cannot be changed."
            )
        # Same package: the loop has no public accessor for its client.
        kwargs["client"] = base._client
        if model:
            kwargs["model"] = model
    else:
        kwargs["provider"] = provider or ("compatible" if base_url else base.provider)
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
    with warnings.catch_warnings():
        # Open mode was chosen on the page, so the "no knowledge base" warning
        # meant for a programmer who forgot one is noise here.
        warnings.simplefilter("ignore", LoopWarning)
        return Loop(knowledge, instructions, **kwargs)  # type: ignore[arg-type]


def _instructions(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        return lines or None
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise LoopError("instructions must be a list of strings, one instruction per item.")
        items = [item.strip() for item in value if item.strip()]
        return items or None
    raise LoopError("instructions must be a list of strings, one instruction per item.")


def _choice(value: object, default: str, name: str, allowed: tuple[str, ...]) -> str:
    if value is None or value == "":
        return default
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise LoopError(f"{name} must be one of: {', '.join(allowed)}.")
    return value.strip().lower()


def _positive_int(value: object, default: int, name: str) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LoopError(f"{name} must be an integer of at least 1.")
    return value


def _bool(value: object, default: bool, name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise LoopError(f"{name} must be true or false.")
    return value


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LoopError(f"{name} must be a string.")
    return value.strip() or None


class _Handler(BaseHTTPRequestHandler):
    loop: Loop
    session: SessionStore | None = None
    server_version = "loopkit"

    def log_message(self, format: str, *args: object) -> None:
        if self.loop.trace:
            super().log_message(format, *args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"ok": True})
            return
        if path == "/config":
            self._send(200, config(self.loop, self.session))
            return
        if path == "/session":
            if self.session is None:
                self._send(404, {"error": _SESSIONS_OFF})
                return
            try:
                state = self.session.load()
            except LoopError as exc:
                self._send(500, {"error": str(exc)})
                return
            self._send(200, {"session": state, "path": str(self.session.path)})
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

    def do_PUT(self) -> None:
        path = urlparse(self.path).path
        if path != "/session":
            self._send(404, {"error": "Not found."})
            return
        if self.session is None:
            self._send(404, {"error": _SESSIONS_OFF})
            return
        payload = self._read_json(MAX_SESSION_BYTES, "Send the session as a JSON object.")
        if payload is None:
            return
        try:
            saved = self.session.save(payload)
        except LoopError as exc:
            self._send(400 if "session file" not in str(exc) else 500, {"error": str(exc)})
            return
        self._send(200, {"ok": True, "saved_at": saved["saved_at"], "path": str(self.session.path)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/ask":
            self._send(404, {"error": "Not found."})
            return
        payload = self._read_json(MAX_BODY_BYTES, "Send a JSON object with a task string.")
        if payload is None:
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("task"), str):
            self._send(400, {"error": "Send a JSON object with a task string."})
            return
        budget = payload.get("token_budget")
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
        ):
            self._send(400, {"error": "token_budget must be an integer of at least 1."})
            return
        try:
            task = compose_task(payload["task"], parse_history(payload.get("history")))
            loop = loop_for(self.loop, options)
            if budget is None:
                answer = loop.ask(payload["task"])
            else:
                answer = loop.ask(payload["task"], token_budget=budget)
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
        self._send(200, {**answer.to_dict(), "task": task, "loop": settings(loop)})

    def _read_json(self, limit: int, expected: str) -> object | None:
        """The request body as JSON, or ``None`` after an error response was sent."""

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "Content-Length must be an integer."})
            return None
        if length < 0 or length > limit:
            self._send(413, {"error": "Request body is too large."})
            return None
        raw = self.rfile.read(length) if length else b""
        try:
            payload: object = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": expected})
            return None
        if payload is None:
            self._send(400, {"error": expected})
            return None
        return payload

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


_SESSIONS_OFF = "This server keeps no session. Start it with: loopkit serve --session [FILE]"


def make_handler(loop: Loop, session: SessionStore | None = None) -> type[_Handler]:
    class Handler(_Handler):
        pass

    Handler.loop = loop
    Handler.session = session
    return Handler


def make_server(
    loop: Loop, host: str, port: int, session: SessionStore | None = None
) -> HTTPServer:
    try:
        return HTTPServer((host, port), make_handler(loop, session))
    except OSError as exc:
        raise LoopError(f"Could not listen on {host}:{port}. {exc}") from exc


def server_url(server: HTTPServer) -> str:
    """The ``http://host:port`` the server is actually bound to (port 0 picks a free one)."""

    bound_host, bound_port = server.server_address[:2]
    host_text = bound_host.decode() if isinstance(bound_host, bytes) else str(bound_host)
    return f"http://{host_text}:{bound_port}"


def serve(
    loop: Loop,
    host: str = "127.0.0.1",
    port: int = 8080,
    session: str | Path | SessionStore | None = None,
) -> None:
    """Serve the page until interrupted. ``session`` is a file that keeps the chats across refreshes."""

    store = (
        session if session is None or isinstance(session, SessionStore) else SessionStore(session)
    )
    if store is not None:
        store.load()  # a corrupt file is reported now, not on the first refresh
    server = make_server(loop, host, port, store)
    print(f"loopkit listening on {server_url(server)}", file=sys.stderr)
    if store is not None:
        print(f"loopkit: session file: {store.path}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
