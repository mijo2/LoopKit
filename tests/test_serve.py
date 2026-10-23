import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from loopkit.errors import LoopError
from loopkit.loop import OPEN_INSTRUCTIONS, Loop
from loopkit.models import Answer
from loopkit.serve import (
    MAX_BODY_BYTES,
    OPTION_KEYS,
    PAGE,
    SessionStore,
    compose_task,
    config,
    loop_for,
    make_server,
    parse_history,
    server_url,
    validate_session,
)
from tests.fakes import ScriptedModel, T, accepted, checks


@contextmanager
def running(loop: Loop, session: SessionStore | None = None) -> Iterator[str]:
    server = make_server(loop, "127.0.0.1", 0, session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server_url(server)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def page() -> Iterator[tuple[str, ScriptedModel]]:
    model = ScriptedModel(accepted("Refunds last 14 days. [p1]", "p1"))
    loop = Loop(knowledge="Refunds last 14 days after the charge.", client=model, trace=False)
    with running(loop) as base:
        yield base, model


def test_page_and_ask(page: tuple[str, ScriptedModel]) -> None:
    base, model = page
    with urllib.request.urlopen(base + "/", timeout=5) as response:
        html = response.read().decode("utf-8")
    assert response.status == 200
    assert "<form" in html
    assert "Loop.ask" in html
    assert 'fetch("/config")' in html
    assert 'fetch("/ask"' in html
    assert "history" in html
    # The follow-up task is composed by the server, not the page.
    assert "Previous conversation" not in html
    assert "innerHTML" not in PAGE

    with urllib.request.urlopen(base + "/health", timeout=5) as response:
        assert json.loads(response.read()) == {"ok": True}

    request = urllib.request.Request(
        base + "/ask",
        data=json.dumps({"task": "What is the refund window?"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read())
    assert payload["text"] == "Refunds last 14 days. [p1]"
    assert payload["citations"][0]["source"] == "text"
    assert payload["loop"]["provider"] == "custom"
    assert payload["loop"]["mode"] == "knowledge"
    assert "14 days" in str(model.calls[0]["user"])


def test_bad_requests(page: tuple[str, ScriptedModel]) -> None:
    base, _model = page
    request = urllib.request.Request(
        base + "/ask",
        data=b"not-json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=5)
    assert raised.value.code == 400

    empty = urllib.request.Request(
        base + "/ask",
        data=json.dumps({"task": "  "}).encode(),
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(empty, timeout=5)
    assert raised.value.code == 400
    assert "empty" in json.loads(raised.value.read())["error"]


def _sub(body: dict[str, object], key: str) -> dict[str, object]:
    """A nested object in a response body, typed for mypy."""

    value = body[key]
    assert isinstance(value, dict)
    return value


def _post(base: str, payload: bytes) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(base + "/ask", data=payload, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_model_failure_reports_the_stage(capsys: pytest.CaptureFixture[str]) -> None:
    class Boom:
        def parse(self, *, system_text: str, user_text: str, schema: type[T]) -> T:
            raise RuntimeError("network down")

    loop = Loop(knowledge="Refunds last 14 days after the charge.", client=Boom(), trace=False)
    with running(loop) as base:
        status, body = _post(base, json.dumps({"task": "What is the refund window?"}).encode())
    assert status == 400
    assert "network down" in str(body["error"])
    capsys.readouterr()


def test_unexpected_error_is_not_leaked_in_the_body(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = Loop(
        knowledge="Refunds last 14 days after the charge.", client=ScriptedModel([]), trace=False
    )

    def explode(_task: str) -> Answer:
        raise RuntimeError("boom secret")

    monkeypatch.setattr(loop, "ask", explode)
    with running(loop) as base:
        status, body = _post(base, json.dumps({"task": "What is the refund window?"}).encode())
    assert status == 500
    assert body["error"] == "The model call failed."
    assert "boom secret" not in json.dumps(body)
    assert "boom secret" in capsys.readouterr().err


def test_body_limit(page: tuple[str, ScriptedModel], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loopkit.serve.MAX_BODY_BYTES", 16)
    assert MAX_BODY_BYTES > 16
    base, _model = page
    request = urllib.request.Request(
        base + "/ask",
        data=b'{"task": "this body is definitely too long"}',
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=5)
    assert raised.value.code == 413


def test_config_reports_settings_and_key_presence_only(
    page: tuple[str, ScriptedModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    base, _model = page
    with urllib.request.urlopen(base + "/config", timeout=5) as response:
        raw = response.read().decode("utf-8")
    body = json.loads(raw)
    assert "sk-secret-value" not in raw
    assert body["providers"]["openai"]["key_set"] is True
    assert body["providers"]["xai"]["key_set"] is False
    assert body["providers"]["openai"]["env"] == "OPENAI_API_KEY"
    loop = body["loop"]
    assert loop["provider"] == "custom"
    assert loop["mode"] == "knowledge"
    assert loop["check_mode"] == "each"
    assert loop["response"] == "normal"
    assert loop["max_turns"] == 3
    assert loop["auto_optimize"] is False
    assert loop["knowledge"]["passages"] == 1
    assert loop["knowledge"]["sources"] == ["text"]
    assert len(loop["instructions"]) == 4
    assert sorted(body["option_keys"]) == sorted(OPTION_KEYS)
    combos = {(d["mode"], d["response"], d["search"]) for d in body["default_instructions"]}
    assert ("open", "book", False) in combos
    assert ("knowledge", "normal", True) in combos


def test_ask_with_options_builds_a_loop_for_that_request() -> None:
    from loopkit.schemas import Draft

    # One draft + one instruction check + grounding: the override asks for a
    # single custom instruction, so the scripted replies are shorter than the
    # default four-instruction list.
    model = ScriptedModel(
        [
            Draft(answer="Fourteen days. [p1]", cited_ids=["p1"]),
            *checks(instruction_count=1),
        ]
    )
    loop = Loop(knowledge="Refunds last 14 days after the charge.", client=model, trace=False)
    options = {
        "instructions": ["Answer in one sentence."],
        "check_mode": "each",
        "response": "book",
        "max_turns": 1,
        "top_k": 2,
        "context_budget": 500,
        "search": False,
        "auto_optimize": False,
    }
    with running(loop) as base:
        status, body = _post(
            base, json.dumps({"task": "Refund window?", "options": options}).encode()
        )
    assert status == 200, body
    assert body["text"] == "Fourteen days. [p1]"
    used = _sub(body, "loop")
    assert used["instructions"] == ["Answer in one sentence."]
    assert used["response"] == "book"
    assert used["max_turns"] == 1
    assert used["top_k"] == 2
    assert used["provider"] == "custom"
    # The same scripted client answered; the base loop's settings are untouched.
    assert len(model.calls) == 3
    assert "Answer in one sentence." in str(model.calls[0]["user"])
    assert loop.response == "normal"
    assert loop.max_turns == 3


def test_options_switch_to_open_mode_and_pasted_text() -> None:
    from loopkit.schemas import Draft

    model = ScriptedModel(
        [
            Draft(answer="Thirty days.", cited_ids=[]),
            *checks(instruction_count=3),
            Draft(answer="Thirty days. [p1]", cited_ids=["p1"]),
            *checks(),
        ]
    )
    loop = Loop(knowledge="Refunds last 14 days after the charge.", client=model, trace=False)
    with running(loop) as base:
        status, body = _post(
            base,
            json.dumps(
                {"task": "Refund window?", "options": {"knowledge_source": "none"}}
            ).encode(),
        )
        assert status == 200, body
        assert body["mode"] == "open"
        used = _sub(body, "loop")
        assert used["mode"] == "open"
        assert used["instructions"] == list(OPEN_INSTRUCTIONS)
        assert "PASSAGES" not in str(model.calls[0]["user"])

        status, body = _post(
            base,
            json.dumps(
                {
                    "task": "Refund window?",
                    "options": {
                        "knowledge_source": "text",
                        "knowledge_text": "Acme refunds any order within 30 days.",
                    },
                }
            ).encode(),
        )
        assert status == 200, body
        assert body["mode"] == "knowledge"
        citations = body["citations"]
        assert isinstance(citations, list)
        assert citations[0]["text"] == "Acme refunds any order within 30 days."
        # Open mode used 5 calls (draft, 3 checks, accuracy); this draft is the 6th.
        assert "30 days" in str(model.calls[5]["user"])
        assert "14 days" not in str(model.calls[5]["user"])
    assert loop.mode == "knowledge"


@pytest.mark.parametrize(
    ("options", "fragment"),
    [
        ({"bogus": 1}, "Unknown option"),
        ({"max_turns": "3"}, "max_turns must be an integer"),
        ({"max_turns": 0}, "max_turns must be an integer"),
        ({"top_k": True}, "top_k must be an integer"),
        ({"check_mode": "some"}, "check_mode must be one of"),
        ({"response": "poem"}, "response must be one of"),
        ({"search": "yes"}, "search must be true or false"),
        ({"knowledge_source": "disk"}, "knowledge_source must be one of"),
        ({"knowledge_source": "text"}, "knowledge_text is empty"),
        ({"knowledge_source": "text", "knowledge_text": "   "}, "knowledge_text is empty"),
        ({"instructions": [1, 2]}, "instructions must be a list of strings"),
        ({"instructions": "   "}, None),  # blank means the built-in list, not an error
        ({"provider": "openai"}, "custom client"),
        ({"base_url": "http://localhost:11434/v1"}, "custom client"),
        ({"model": 3}, "model must be a string"),
    ],
)
def test_bad_options_are_400s(options: dict[str, object], fragment: str | None) -> None:
    model = ScriptedModel(accepted("Refunds last 14 days. [p1]", "p1"))
    loop = Loop(knowledge="Refunds last 14 days after the charge.", client=model, trace=False)
    with running(loop) as base:
        status, body = _post(
            base, json.dumps({"task": "Refund window?", "options": options}).encode()
        )
    if fragment is None:
        assert status == 200, body
        return
    assert status == 400, body
    assert fragment in str(body["error"])


def test_options_must_be_an_object(page: tuple[str, ScriptedModel]) -> None:
    base, _model = page
    status, body = _post(base, json.dumps({"task": "Refund window?", "options": [1]}).encode())
    assert status == 400
    assert "options must be a JSON object" in str(body["error"])


def test_loop_for_without_options_is_the_same_loop() -> None:
    loop = Loop(
        knowledge="Refunds last 14 days after the charge.", client=ScriptedModel([]), trace=False
    )
    assert loop_for(loop, None) is loop
    assert loop_for(loop, {}) is loop


def test_loop_for_reuses_the_knowledge_already_in_memory() -> None:
    loop = Loop(
        knowledge="Refunds last 14 days after the charge.", client=ScriptedModel([]), trace=False
    )
    other = loop_for(loop, {"max_turns": 5, "model": "scripted-2"})
    assert other is not loop
    assert other.knowledge is loop.knowledge
    assert other.max_turns == 5
    assert other.model == "scripted-2"
    assert other.provider == "custom"
    summary = config(loop)["loop"]
    assert isinstance(summary, dict)
    assert summary["model"] == loop.model


def test_base_url_without_provider_implies_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    # Building a Loop only reads the environment; no request is made until ask().
    monkeypatch.delenv("LOOPKIT_PROVIDER", raising=False)
    monkeypatch.delenv("LOOPKIT_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    base = Loop(knowledge="Refunds last 14 days after the charge.", provider="openai", trace=False)
    other = loop_for(base, {"base_url": "http://localhost:11434/v1", "model": "llama3.1"})
    assert other.provider == "compatible"
    assert other.model == "llama3.1"


def test_compose_task_first_ask_is_verbatim() -> None:
    assert compose_task("What is the refund window?", None) == "What is the refund window?"
    assert compose_task("What is the refund window?", []) == "What is the refund window?"


def test_compose_task_quotes_earlier_turns_most_recent_first() -> None:
    history = [
        ("What is the refund window?", "14 days. [p1]"),
        ("And after that?", 'The sale is "final", see\n[p2].'),
    ]
    composed = compose_task("How do I ask?", history)
    assert composed == (
        "User asked: How do I ask?; Previous conversation: ["
        '("And after that?", "The sale is \\"final\\", see\\n[p2]."), '
        '("What is the refund window?", "14 days. [p1]")]'
    )
    # Every quoted piece is valid JSON, so the model sees unambiguous pairs.
    assert json.loads('"The sale is \\"final\\", see\\n[p2]."') == history[1][1]


def test_parse_history() -> None:
    assert parse_history(None) == []
    assert parse_history([["a", "b"], ("c", "d")]) == [("a", "b"), ("c", "d")]
    for bad in ({"a": "b"}, [["a"]], [["a", "b", "c"]], [["a", 1]], ["ab"], "ab"):
        with pytest.raises(LoopError, match="history must be"):
            parse_history(bad)


def test_ask_with_history_sends_one_composed_task(page: tuple[str, ScriptedModel]) -> None:
    base, model = page
    payload = {
        "task": "And after that?",
        "history": [["What is the refund window?", "Refunds last 14 days. [p1]"]],
    }
    status, body = _post(base, json.dumps(payload).encode())
    assert status == 200, body
    expected = (
        "User asked: And after that?; Previous conversation: "
        '[("What is the refund window?", "Refunds last 14 days. [p1]")]'
    )
    assert body["task"] == expected
    assert expected in str(model.calls[0]["user"])


def test_ask_without_history_reports_the_task_as_typed(page: tuple[str, ScriptedModel]) -> None:
    base, _model = page
    status, body = _post(base, json.dumps({"task": "What is the refund window?"}).encode())
    assert status == 200
    assert body["task"] == "What is the refund window?"


def test_bad_history_is_a_400(page: tuple[str, ScriptedModel]) -> None:
    base, _model = page
    status, body = _post(base, json.dumps({"task": "x", "history": [["only one"]]}).encode())
    assert status == 400
    assert "history must be" in str(body["error"])


def _session_state(**overrides: object) -> dict[str, object]:
    tab: dict[str, object] = {
        "id": 1,
        "kind": "chat",
        "label": "Chat 1",
        "title": "Refund window?",
        "options": {"max_turns": 2},
        "messages": [
            {"role": "user", "text": "Refund window?"},
            {"role": "assistant", "typed": "Refund window?", "answer": {"text": "14 days. [p1]"}},
            {"role": "error", "text": "boom", "typed": "x"},
        ],
        "history": [["Refund window?", "14 days. [p1]"]],
    }
    state: dict[str, object] = {"version": 1, "active_id": 1, "tabs": [tab]}
    state.update(overrides)
    return state


def _request(
    base: str, path: str, method: str, payload: bytes | None = None
) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(base + path, data=payload, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_session_store_roundtrip_is_atomic(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "runs" / "chats.json")
    assert store.load() is None
    saved = store.save(_session_state())
    assert isinstance(saved["saved_at"], str)
    assert store.path.is_file()
    assert [p.name for p in store.path.parent.iterdir()] == [
        "chats.json"
    ]  # no temp file left behind
    loaded = store.load()
    assert loaded is not None
    assert loaded["active_id"] == 1
    tabs = loaded["tabs"]
    assert isinstance(tabs, list)
    assert tabs[0]["history"] == [["Refund window?", "14 days. [p1]"]]
    assert tabs[0]["messages"][1]["answer"] == {"text": "14 days. [p1]"}


def test_session_store_reports_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "chats.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LoopError, match="not valid JSON"):
        SessionStore(path).load()
    path.write_text("   ", encoding="utf-8")
    assert SessionStore(path).load() is None
    path.write_text(json.dumps({"tabs": "nope"}), encoding="utf-8")
    with pytest.raises(LoopError, match="tabs list"):
        SessionStore(path).load()


@pytest.mark.parametrize(
    ("state", "fragment"),
    [
        ([], "must be a JSON object"),
        ({"version": 2, "tabs": []}, "Unsupported session version"),
        ({"tabs": {}}, "tabs list"),
        ({"tabs": [1]}, "Tab 1 must be an object"),
        ({"tabs": [{"id": "1"}]}, "integer id"),
        ({"tabs": [{"id": True}]}, "integer id"),
        (
            {
                "tabs": [
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": "a",
                        "title": "b",
                        "options": {},
                        "messages": [],
                    },
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": "a",
                        "title": "b",
                        "options": {},
                        "messages": [],
                    },
                ]
            },
            "used twice",
        ),
        (
            {
                "tabs": [
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": 3,
                        "title": "b",
                        "options": {},
                        "messages": [],
                    }
                ]
            },
            "string label",
        ),
        (
            {
                "tabs": [
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": "a",
                        "title": "b",
                        "options": [],
                        "messages": [],
                    }
                ]
            },
            "options object",
        ),
        (
            {
                "tabs": [
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": "a",
                        "title": "b",
                        "options": {},
                        "messages": {},
                    }
                ]
            },
            "messages list",
        ),
        (
            {
                "tabs": [
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": "a",
                        "title": "b",
                        "options": {},
                        "messages": [{"role": "pending"}],
                    }
                ]
            },
            "needs a role",
        ),
        (
            {
                "tabs": [
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": "a",
                        "title": "b",
                        "options": {},
                        "messages": [{"role": "assistant"}],
                    }
                ]
            },
            "without an answer object",
        ),
        (
            {
                "tabs": [
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": "a",
                        "title": "b",
                        "options": {},
                        "messages": [{"role": "user"}],
                    }
                ]
            },
            "text string",
        ),
        (
            {
                "tabs": [
                    {
                        "id": 1,
                        "kind": "chat",
                        "label": "a",
                        "title": "b",
                        "options": {},
                        "messages": [],
                        "history": [["x"]],
                    }
                ]
            },
            "Tab 1: history must be",
        ),
        ({"tabs": [], "active_id": "1"}, "active_id must be"),
    ],
)
def test_validate_session_names_the_problem(state: object, fragment: str) -> None:
    with pytest.raises(LoopError, match=fragment):
        validate_session(state)


def test_validate_session_repairs_a_stale_active_id() -> None:
    checked = validate_session(_session_state(active_id=99))
    assert checked["active_id"] == 1
    assert validate_session({"tabs": []})["active_id"] is None


def test_session_routes_are_off_without_a_file(page: tuple[str, ScriptedModel]) -> None:
    base, _model = page
    status, body = _request(base, "/session", "GET")
    assert status == 404
    assert "--session" in str(body["error"])
    status, body = _request(base, "/session", "PUT", json.dumps(_session_state()).encode())
    assert status == 404
    with urllib.request.urlopen(base + "/config", timeout=5) as response:
        assert json.loads(response.read())["session"] == {"enabled": False, "path": None}


def test_session_routes_save_and_load(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "chats.json")
    loop = Loop(
        knowledge="Refunds last 14 days after the charge.", client=ScriptedModel([]), trace=False
    )
    with running(loop, store) as base:
        with urllib.request.urlopen(base + "/config", timeout=5) as response:
            assert json.loads(response.read())["session"] == {
                "enabled": True,
                "path": str(store.path),
            }

        status, body = _request(base, "/session", "GET")
        assert status == 200
        assert body == {"session": None, "path": str(store.path)}

        status, body = _request(base, "/session", "PUT", json.dumps(_session_state()).encode())
        assert status == 200, body
        assert body["ok"] is True and body["path"] == str(store.path)

        status, body = _request(base, "/session", "GET")
        assert status == 200
        saved = body["session"]
        assert isinstance(saved, dict)
        assert saved["active_id"] == 1
        assert saved["tabs"][0]["title"] == "Refund window?"

        status, body = _request(base, "/session", "PUT", json.dumps({"tabs": "x"}).encode())
        assert status == 400
        assert "tabs list" in str(body["error"])
        status, body = _request(base, "/session", "PUT", b"not json")
        assert status == 400
        status, body = _request(base, "/session", "PUT", b"null")
        assert status == 400
        status, body = _request(base, "/other", "PUT", b"{}")
        assert status == 404

    # The file is what a later server start reads.
    again = SessionStore(store.path).load()
    assert again is not None and again["active_id"] == 1


def test_session_body_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loopkit.serve.MAX_SESSION_BYTES", 16)
    store = SessionStore(tmp_path / "chats.json")
    loop = Loop(
        knowledge="Refunds last 14 days after the charge.", client=ScriptedModel([]), trace=False
    )
    with running(loop, store) as base:
        status, _body = _request(base, "/session", "PUT", json.dumps(_session_state()).encode())
    assert status == 413
    assert not store.path.exists()


def test_page_loads_the_session_first() -> None:
    assert 'fetch("/session")' in PAGE
    assert 'method: "PUT"' in PAGE
    assert PAGE.index('fetch("/session")') < PAGE.index("activateTab(state.activeId")
