import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from loopkit.loop import Loop
from loopkit.models import Answer
from loopkit.serve import MAX_BODY_BYTES, PAGE, make_server, server_url
from tests.fakes import ScriptedModel, T, accepted


@contextmanager
def running(loop: Loop) -> Iterator[str]:
    server = make_server(loop, "127.0.0.1", 0)
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
    assert "answer.textContent" in html
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
