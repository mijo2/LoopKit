"""The debug log records prompts, raw replies, and failures, in sections."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pytest

from loopkit.debuglog import disable_debug_log, resolve_debug
from loopkit.errors import LoopError
from loopkit.loop import Loop
from loopkit.schemas import Draft
from tests.fakes import ScriptedModel, accepted

POLICY = "Refunds are available for 30 days. After that the sale is final."


class _Body(io.BytesIO):
    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("XAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOOPKIT_PROVIDER", "LOOPKIT_MODEL", "LOOPKIT_DEBUG"):
        monkeypatch.delenv(name, raising=False)


def test_resolve_debug_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOOPKIT_DEBUG", raising=False)
    assert resolve_debug(None) is None
    assert resolve_debug(False) is None
    assert resolve_debug(True) == Path("loopkit-debug.log")
    assert resolve_debug("runs/a.log") == Path("runs/a.log")
    monkeypatch.setenv("LOOPKIT_DEBUG", "1")
    assert resolve_debug(None) == Path("loopkit-debug.log")
    monkeypatch.setenv("LOOPKIT_DEBUG", "off")
    assert resolve_debug(None) is None
    monkeypatch.setenv("LOOPKIT_DEBUG", "x/y.log")
    assert resolve_debug(None) == Path("x/y.log")


def test_http_client_raw_reply_is_in_the_log_on_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    replies = iter(
        [
            json.dumps({"answer": "Thirty days. [p1]", "cited_ids": ["p1"]}),
            "Sure! Here is my verdict: followed.",  # not JSON: validation fails
        ]
    )

    def fake(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        text = next(replies)
        body = {"id": "msg_1", "stop_reason": "end_turn", "content": [{"type": "text", "text": text}], "usage": {"input_tokens": 10, "output_tokens": 5}}
        return _Body(json.dumps(body).encode())

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", fake)
    log = tmp_path / "dbg" / "run.log"
    loop = Loop(knowledge=POLICY, instructions=["Be brief."], debug=log, trace=False)
    try:
        with pytest.raises(LoopError, match="does not match OneCheck") as info:
            loop.ask("Refund window?")
        assert "Sure! Here is my verdict" in str(info.value)
        text = log.read_text(encoding="utf-8")
        assert "CALL 1: draft (turn 1)" in text
        assert "--- RAW MODEL TEXT ---" in text
        assert '"answer": "Thirty days. [p1]"' in text
        assert "--- RAW RESPONSE BODY ---" in text
        assert '"stop_reason": "end_turn"' in text
        assert "--- REQUEST PARAMETERS ---" in text
        assert '"max_tokens": 8192' in text
        assert "RESULT 2: check instruction 1 (turn 1) -> error" in text
        assert "Sure! Here is my verdict: followed." in text
        assert "--- ERROR ---" in text
        # Prompts are logged once, in the CALL block, not again in the request dump.
        assert text.count("You draft an answer for one step") == 1
    finally:
        disable_debug_log(log)


def test_scripted_client_without_raw_text_still_logs_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    log = tmp_path / "scripted.log"
    model = ScriptedModel(accepted("Thirty days. [p1]", "p1", instructions=1))
    loop = Loop(knowledge=POLICY, instructions=["Be brief."], client=model, debug=str(log), trace=False)
    try:
        loop.ask("Refund window?")
        text = log.read_text(encoding="utf-8")
        assert "--- PARSED ---" in text
        assert '"cited_ids"' in text
        assert "RAW MODEL TEXT ---\n(none)" in text
        assert "ANSWER" in text and "stopped_reason=passed" in text
    finally:
        disable_debug_log(log)


def test_env_enables_the_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    log = tmp_path / "env.log"
    monkeypatch.setenv("LOOPKIT_DEBUG", str(log))
    model = ScriptedModel(accepted("Thirty days. [p1]", "p1", instructions=1))
    loop = Loop(knowledge=POLICY, instructions=["Be brief."], client=model, trace=False)
    try:
        assert loop.debug_path == log.resolve()
        loop.ask("Refund window?")
        assert "TURN 1 DECISION: passed" in log.read_text(encoding="utf-8")
    finally:
        disable_debug_log(log)


def test_no_debug_means_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.chdir(tmp_path)
    model = ScriptedModel(accepted("Thirty days. [p1]", "p1", instructions=1))
    loop = Loop(knowledge=POLICY, instructions=["Be brief."], client=model, trace=False)
    loop.ask("Refund window?")
    assert loop.debug_path is None
    assert not (tmp_path / "loopkit-debug.log").exists()
    assert Draft  # keep the import used for readers of this file
