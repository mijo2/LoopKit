from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path
from typing import Any, Self

import pytest

from loopkit.errors import LoopError
from loopkit.httpapi import strict_json_schema
from loopkit.loop import BOOK_MAX_OUTPUT_TOKENS, Loop
from loopkit.schemas import Draft, DraftCheck, GroundingCheck, InstructionVerdict, OneCheck
from tests.fakes import ScriptedModel, accepted

POLICY = "Refunds last 14 days after the charge."
_KEYS = (
    "XAI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LOOPKIT_API_KEY",
    "LOOPKIT_PROVIDER",
    "LOOPKIT_MODEL",
    "LOOPKIT_BASE_URL",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _KEYS:
        monkeypatch.delenv(name, raising=False)


class _Body:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _install(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> list[urllib.request.Request]:
    seen: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        seen.append(request)
        return _Body(payload)

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", fake_urlopen)
    return seen


def _request_json(request: urllib.request.Request) -> dict[str, Any]:
    assert isinstance(request.data, bytes)
    body = json.loads(request.data.decode())
    assert isinstance(body, dict)
    return body


def test_schema_is_closed() -> None:
    for schema in (Draft, DraftCheck, OneCheck, GroundingCheck, InstructionVerdict):
        _assert_closed(strict_json_schema(schema))


def _assert_closed(node: object) -> None:
    if isinstance(node, list):
        for item in node:
            _assert_closed(item)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        assert node["additionalProperties"] is False
        properties = node["properties"]
        assert isinstance(properties, dict)
        assert set(node["required"]) == set(properties)
    for value in node.values():
        _assert_closed(value)


def test_one_openai_key_selects_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    loop = Loop(knowledge=POLICY)
    assert loop.provider == "openai"
    assert loop.model == "gpt-6.1-sol"


def test_one_anthropic_key_selects_sonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    loop = Loop(knowledge=POLICY)
    assert loop.provider == "anthropic"
    assert loop.model == "claude-sonnet-5-5"


def test_only_xai_key_uses_the_xai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    created: dict[str, object] = {}

    class Fake:
        provider = "xai"

        def __init__(
            self,
            *,
            model: str,
            api_key: str,
            timeout: float,
            max_output_tokens: int | None = None,
            search: bool = False,
        ) -> None:
            created.update(
                model=model, api_key=api_key, timeout=timeout, max_output_tokens=max_output_tokens, search=search
            )
            self.model = model

    monkeypatch.setattr("loopkit.client.XAIClient", Fake)
    loop = Loop(knowledge=POLICY)
    assert loop.provider == "xai"
    assert loop.model == "grok-4.7"
    assert created["api_key"] == "xai-test"
    assert created["max_output_tokens"] is None


def test_several_keys_require_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    with pytest.raises(LoopError, match="LOOPKIT_PROVIDER"):
        Loop(knowledge=POLICY)


def test_explicit_provider_uses_that_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    loop = Loop(knowledge=POLICY, provider="anthropic", model="claude-opus-5-5")
    assert loop.provider == "anthropic"
    assert loop.model == "claude-opus-5-5"


def test_model_env_overrides_the_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LOOPKIT_MODEL", "gpt-6-astra")
    loop = Loop(knowledge=POLICY)
    assert loop.model == "gpt-6-astra"


def test_no_key_names_the_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(LoopError, match="OPENAI_API_KEY"):
        Loop(knowledge=POLICY)


def test_local_server_needs_no_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    seen = _install(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": Draft(answer="Local answer", cited_ids=[]).model_dump_json()
                    }
                }
            ]
        },
    )
    loop = Loop(knowledge=POLICY, base_url="http://127.0.0.1:11434/v1", model="llama3.1")
    parsed = loop._client.parse(system_text="rules", user_text="task", schema=Draft)
    assert loop.provider == "compatible"
    assert parsed.answer == "Local answer"
    request = seen[0]
    assert request.full_url == "http://127.0.0.1:11434/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer local"
    body = _request_json(request)
    assert body["model"] == "llama3.1"
    response_format = body["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["json_schema"]["strict"] is True


def test_openai_responses_call_does_not_store_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    draft = Draft(answer="Fourteen days.", cited_ids=["p1"]).model_dump_json()
    seen = _install(
        monkeypatch,
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": f"```json\n{draft}\n```"}],
                }
            ]
        },
    )
    loop = Loop(knowledge=POLICY)
    parsed = loop._client.parse(system_text="rules", user_text="task", schema=Draft)
    assert parsed.cited_ids == ["p1"]
    request = seen[0]
    assert request.full_url == "https://api.openai.com/v1/responses"
    body = _request_json(request)
    assert body["store"] is False
    assert body["model"] == "gpt-6.1-sol"
    text_format = body["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["schema"]["additionalProperties"] is False


def test_anthropic_reads_the_text_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    draft = DraftCheck(
        instructions=[InstructionVerdict(number=1, followed=True, note="")],
        grounded=True,
        revise=False,
        notes="",
        gaps=[],
    ).model_dump_json()
    seen = _install(
        monkeypatch,
        {
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": draft},
            ]
        },
    )
    loop = Loop(knowledge=POLICY, api_key="sk-ant", provider="anthropic")
    parsed = loop._client.parse(system_text="rules", user_text="task", schema=DraftCheck)
    assert parsed.revise is False
    request = seen[0]
    assert request.full_url == "https://api.anthropic.com/v1/messages"
    assert request.get_header("X-api-key") == "sk-ant"
    body = _request_json(request)
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 8192
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_book_response_raises_the_anthropic_output_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    draft = Draft(answer="A long chapter.", cited_ids=[]).model_dump_json()
    seen = _install(monkeypatch, {"content": [{"type": "text", "text": draft}]})
    loop = Loop(knowledge=POLICY, api_key="sk-ant", provider="anthropic", response="book")
    parsed = loop._client.parse(system_text="rules", user_text="task", schema=Draft)
    assert parsed.answer == "A long chapter."
    body = _request_json(seen[0])
    assert body["max_tokens"] == BOOK_MAX_OUTPUT_TOKENS


def test_http_error_is_a_loop_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fail(request: urllib.request.Request, timeout: float | None = None) -> _Body:
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b"missing key"),
        )

    monkeypatch.setattr("loopkit.httpapi.urllib.request.urlopen", fail)
    loop = Loop(knowledge=POLICY)
    with pytest.raises(LoopError, match="HTTP 401"):
        loop._client.parse(system_text="rules", user_text="task", schema=Draft)


def test_toml_provider_is_passed_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    (tmp_path / "notes.md").write_text("Desks ship in two weeks.\n", encoding="utf-8")
    (tmp_path / "loopkit.toml").write_text(
        'knowledge = "notes.md"\nprovider = "openai"\nmodel = "gpt-6-astra"\n',
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_make_client(**kwargs: object) -> ScriptedModel:
        seen.update(kwargs)
        model = ScriptedModel(accepted("Two weeks. [p1]", "p1"))
        model.provider = "openai"  # type: ignore[attr-defined]
        model.model = "gpt-6-astra"  # type: ignore[attr-defined]
        return model

    monkeypatch.setattr("loopkit.loop.make_client", fake_make_client)
    loop = Loop.load(tmp_path / "loopkit.toml")
    assert seen["provider"] == "openai"
    assert seen["model"] == "gpt-6-astra"
    assert loop.ask("When do desks ship?").text == "Two weeks. [p1]"


def test_cli_provider_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from loopkit.cli import main

    _clear(monkeypatch)
    seen: dict[str, object] = {}

    def fake_make_client(**kwargs: object) -> ScriptedModel:
        seen.update(kwargs)
        model = ScriptedModel(accepted("Fourteen days.", "p1"))
        model.provider = "openai"  # type: ignore[attr-defined]
        model.model = "gpt-6.1-sol"  # type: ignore[attr-defined]
        return model

    monkeypatch.setattr("loopkit.loop.make_client", fake_make_client)
    code = main(
        [
            "ask",
            "What is the refund window?",
            "--knowledge",
            POLICY,
            "--provider",
            "openai",
            "--base-url",
            "https://example.test/v1",
            "--quiet",
        ]
    )
    assert code == 0
    assert seen["provider"] == "openai"
    assert seen["base_url"] == "https://example.test/v1"
    assert capsys.readouterr().out.strip() == "Fourteen days."


def test_xai_client_retries_transient_grpc_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real SDK is installed; only the network call is replaced."""

    import grpc

    _clear(monkeypatch)
    monkeypatch.setenv("LOOPKIT_MAX_RETRIES", "2")
    monkeypatch.setattr("loopkit.retry.time.sleep", lambda _: None)
    from loopkit.client import XAIClient

    client = XAIClient(model="grok-test", api_key="xai-test")
    calls = {"n": 0}

    class Busy(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.RESOURCE_EXHAUSTED

        def details(self) -> str:
            return "rate limited"

    class Response:
        content = '{"followed": true, "note": ""}'
        id = "resp_1"
        finish_reason = "stop"
        usage = None

    class FakeChat:
        def append(self, message: object) -> None:
            pass

        def parse(self, shape: type[object]) -> tuple[object, object]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise Busy()
            return Response(), shape(followed=True, note="")

    monkeypatch.setattr(client._sdk.chat, "create", lambda **params: FakeChat())
    parsed = client.parse(system_text="rules", user_text="task", schema=OneCheck)
    assert isinstance(parsed, OneCheck) and parsed.followed
    assert calls["n"] == 2
    assert client.last_call is not None
    assert client.last_call["raw_text"] == '{"followed": true, "note": ""}'
    assert client.last_call["response"] == {"id": "resp_1", "finish_reason": "stop"}

    class Bad(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.INVALID_ARGUMENT

    class FailingChat(FakeChat):
        def parse(self, shape: type[object]) -> tuple[object, object]:
            calls["n"] += 1
            raise Bad()

    calls["n"] = 0
    monkeypatch.setattr(client._sdk.chat, "create", lambda **params: FailingChat())
    with pytest.raises(grpc.RpcError):
        client.parse(system_text="rules", user_text="task", schema=OneCheck)
    assert calls["n"] == 1  # a bad request is not retried


def test_search_adds_the_provider_tool_and_reads_the_final_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    draft = {"answer": "The 2024 rule is 30 days (per example.com).", "cited_ids": []}
    seen = _install(
        monkeypatch,
        {
            "output": [
                {"type": "web_search_call", "status": "completed"},
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(draft)}]},
            ]
        },
    )
    loop = Loop(knowledge=POLICY, search=True, trace=False)
    parsed = loop._client.parse(system_text="rules", user_text="task", schema=Draft, search=True)
    assert isinstance(parsed, Draft)
    body = _request_json(seen[0])
    assert body["tools"] == [{"type": "web_search"}]
    # Without search the tool is not sent.
    loop._client.parse(system_text="rules", user_text="task", schema=Draft)
    assert "tools" not in _request_json(seen[1])

    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    seen = _install(
        monkeypatch,
        {
            "content": [
                {"type": "text", "text": "I'll look that up."},
                {"type": "server_tool_use", "name": "web_search", "input": {"query": "refund rule"}},
                {"type": "web_search_tool_result", "content": []},
                {"type": "text", "text": json.dumps(draft)},
            ],
            "stop_reason": "end_turn",
        },
    )
    loop = Loop(knowledge=POLICY, search=True, trace=False)
    parsed = loop._client.parse(system_text="rules", user_text="task", schema=Draft, search=True)
    assert isinstance(parsed, Draft) and parsed.answer.startswith("The 2024 rule")
    body = _request_json(seen[0])
    assert body["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]


def test_search_is_refused_for_the_compatible_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(LoopError, match="search needs a provider"):
        Loop(knowledge=POLICY, base_url="http://localhost:11434/v1", model="llama3.1", search=True)


def test_no_text_reason_names_the_stop_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _install(monkeypatch, {"content": [], "stop_reason": "max_tokens"})
    loop = Loop(knowledge=POLICY, trace=False)
    with pytest.raises(LoopError, match="stop_reason: max_tokens"):
        loop._client.parse(system_text="rules", user_text="task", schema=Draft)

    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _install(monkeypatch, {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "I cannot help with that."}]}], "status": "completed"})
    loop = Loop(knowledge=POLICY, trace=False)
    with pytest.raises(LoopError, match="The model refused: I cannot help"):
        loop._client.parse(system_text="rules", user_text="task", schema=Draft)
