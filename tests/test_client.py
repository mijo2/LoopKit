import sys
import types

import pytest

from loopkit.client import XAIClient
from loopkit.errors import LoopError
from loopkit.loop import BOOK_MAX_OUTPUT_TOKENS
from loopkit.schemas import Draft


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(LoopError, match="XAI_API_KEY"):
        XAIClient(model="grok-4.7")


def test_parse_uses_grok_and_does_not_store_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeChat:
        def append(self, message: object) -> None:
            calls.setdefault("messages", []).append(message)  # type: ignore[union-attr]

        def parse(self, schema: type[object]) -> tuple[object, Draft]:
            calls["schema"] = schema
            return ("response", schema(answer="hello", cited_ids=["p1"]))  # type: ignore[call-arg]

    class FakeChatAPI:
        def create(self, **kwargs: object) -> FakeChat:
            calls["create"] = kwargs
            return FakeChat()

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            calls["client"] = kwargs
            self.chat = FakeChatAPI()

    chat_module = types.ModuleType("xai_sdk.chat")
    chat_module.system = lambda text: ("system", text)  # type: ignore[attr-defined]
    chat_module.user = lambda text: ("user", text)  # type: ignore[attr-defined]
    sdk_module = types.ModuleType("xai_sdk")
    sdk_module.Client = FakeSDK  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xai_sdk", sdk_module)
    monkeypatch.setitem(sys.modules, "xai_sdk.chat", chat_module)

    client = XAIClient(model="grok-4.7", api_key="test-key", timeout=12)
    parsed = client.parse(system_text="rules", user_text="task", schema=Draft)

    assert parsed.answer == "hello"
    assert calls["client"] == {"api_key": "test-key", "timeout": 12}
    assert calls["create"] == {"model": "grok-4.7", "store_messages": False}

    calls.clear()
    client = XAIClient(model="grok-4.7", api_key="test-key", max_output_tokens=BOOK_MAX_OUTPUT_TOKENS)
    client.parse(system_text="rules", user_text="task", schema=Draft)
    assert calls["create"] == {
        "model": "grok-4.7",
        "store_messages": False,
        "max_tokens": BOOK_MAX_OUTPUT_TOKENS,
    }
    assert calls["schema"] is Draft
    assert calls["messages"] == [("system", "rules"), ("user", "task")]
