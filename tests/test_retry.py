"""Retry after rate limits and server errors. No network: urlopen is replaced."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from email.message import Message

import pytest

from loopkit.errors import LoopError
from loopkit.httpapi import post_json
from loopkit.retry import Transient, parse_retry_after, retry_call


class _Body(io.BytesIO):
    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _http_error(url: str, code: int, body: bytes, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "error", hdrs=headers, fp=io.BytesIO(body))


def test_retries_429_and_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = iter([429, 503, 500])
    attempts: list[int] = []

    def fake(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        code = next(codes, None)
        attempts.append(code or 200)
        if code is not None:
            raise _http_error(request.full_url, code, b'{"error":"busy"}', retry_after="0.01" if code == 429 else None)
        return _Body(json.dumps({"ok": True}).encode())

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", fake)
    waits: list[float] = []
    body = post_json("https://api.example.test/v1/x", {}, headers={}, timeout=1, retries=4, sleep=waits.append)
    assert body == {"ok": True}
    assert attempts == [429, 503, 500, 200]
    assert len(waits) == 3
    assert waits[0] == pytest.approx(0.01)  # Retry-After wins over the backoff
    assert 0.5 <= waits[1] <= 6.0  # doubled backoff with jitter


def test_gives_up_after_the_budget_with_the_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        nonlocal calls
        calls += 1
        raise _http_error(request.full_url, 529, b"overloaded")

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", fake)
    with pytest.raises(LoopError, match="HTTP 529: overloaded"):
        post_json("https://api.example.test/v1/x", {}, headers={}, timeout=1, retries=2, sleep=lambda _: None)
    assert calls == 3


def test_client_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        nonlocal calls
        calls += 1
        raise _http_error(request.full_url, 404, b'{"error":{"code":"model_not_found"}}')

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", fake)
    with pytest.raises(LoopError, match="HTTP 404"):
        post_json("https://api.example.test/v1/x", {}, headers={}, timeout=1, retries=4, sleep=lambda _: None)
    assert calls == 1


def test_dropped_connection_is_retried_but_refused_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def reset(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError(ConnectionResetError("reset by peer"))
        return _Body(b"{}")

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", reset)
    assert post_json("https://x.test/v1", {}, headers={}, timeout=1, retries=2, sleep=lambda _: None) == {}
    assert calls == 2

    calls = 0

    def refused(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        nonlocal calls
        calls += 1
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", refused)
    with pytest.raises(LoopError, match="Could not reach"):
        post_json("http://localhost:11434/v1", {}, headers={}, timeout=1, retries=2, sleep=lambda _: None)
    assert calls == 1


def test_env_sets_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPKIT_MAX_RETRIES", "1")
    tries = 0

    def attempt() -> None:
        nonlocal tries
        tries += 1
        raise Transient("busy", cause=LoopError("still busy"))

    with pytest.raises(LoopError, match="still busy"):
        retry_call(attempt, what="test", sleep=lambda _: None)
    assert tries == 2

    monkeypatch.setenv("LOOPKIT_MAX_RETRIES", "0")
    tries = 0
    with pytest.raises(LoopError):
        retry_call(attempt, what="test", sleep=lambda _: None)
    assert tries == 1


def test_retry_after_header_forms() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(" 1.5 ") == 1.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past: no wait


def test_retry_is_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    codes = iter([429])

    def fake(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        code = next(codes, None)
        if code is not None:
            raise _http_error(request.full_url, code, b"slow down")
        return _Body(b"{}")

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", fake)
    with caplog.at_level("WARNING", logger="loopkit"):
        post_json("https://x.test/v1", {}, headers={}, timeout=1, retries=1, sleep=lambda _: None)
    assert any("HTTP 429" in record.message and "Retrying" in record.message for record in caplog.records)


def test_cli_shows_the_retry_on_stderr(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from loopkit.cli import main

    for name in ("XAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOOPKIT_PROVIDER", "LOOPKIT_DEBUG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LOOPKIT_MAX_RETRIES", "1")
    monkeypatch.setattr("loopkit.retry.time.sleep", lambda _: None)
    state = {"n": 0}

    def fake(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        state["n"] += 1
        if state["n"] == 1:
            raise _http_error(request.full_url, 503, b"upstream busy")
        sent = json.loads(request.data)
        name = sent["text"]["format"]["name"]
        if name == "Draft":
            text = json.dumps({"answer": "Thirty days. [p1]", "cited_ids": ["p1"]})
        elif name == "OneCheck":
            text = json.dumps({"followed": True, "note": ""})
        else:
            text = json.dumps({"grounded": True, "revise": False, "notes": "", "gaps": []})
        return _Body(json.dumps({"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}).encode())

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", fake)
    code = main(["ask", "Refund window?", "--knowledge", "Refunds last 30 days.", "--instructions", "Be brief.", "--quiet"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Thirty days. [p1]" in captured.out
    assert "loopkit: POST https://api.openai.com/v1/responses: HTTP 503: upstream busy. Retrying in" in captured.err
