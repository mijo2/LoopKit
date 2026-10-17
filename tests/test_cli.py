import json
from pathlib import Path

import pytest

from loopkit.cli import main
from loopkit.schemas import Draft
from tests.fakes import ScriptedModel, accepted, audit


def test_init_then_ask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    model = ScriptedModel(accepted("Refunds within 30 days. [p1]", "p1"))
    assert main(["ask", "What is the refund window?"], client=model) == 0
    captured = capsys.readouterr()
    assert "Refunds within 30 days." in captured.out
    assert "within 30 days" in str(model.calls[0]["user"])
    assert "turn 1: draft" in captured.err

    assert main(["init"]) == 0
    again = capsys.readouterr()
    assert "already exists" in again.err
    assert (tmp_path / "loopkit.toml").read_text(encoding="utf-8").count('knowledge = "knowledge"') == 1


def test_quiet_json_and_inline_knowledge(capsys: pytest.CaptureFixture[str]) -> None:
    model = ScriptedModel(accepted("Fourteen days. [p1]", "p1"))
    code = main(
        [
            "ask",
            "What is the refund window?",
            "--knowledge",
            "Refunds last 14 days after the charge.",
            "--instructions",
            "Be brief.",
            "--quiet",
            "--json",
        ],
        client=model,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "Fourteen days. [p1]"
    assert payload["stopped_reason"] == "passed"
    assert payload["citations"][0]["id"] == "p1"
    assert "Be brief." in str(model.calls[0]["user"])


def test_repeated_instructions_stay_separate(capsys: pytest.CaptureFixture[str]) -> None:
    model = ScriptedModel(accepted("Fourteen days. [p1]", "p1"))
    code = main(
        [
            "ask",
            "Refunds?",
            "--knowledge",
            "Refunds last 14 days after purchase.",
            "--instructions",
            "Be brief.",
            "--instructions",
            "Cite a passage.",
            "--quiet",
        ],
        client=model,
    )
    assert code == 0
    prompt = str(model.calls[0]["user"])
    assert "1. Be brief." in prompt
    assert "2. Cite a passage." in prompt
    capsys.readouterr()


def test_instructions_file_and_conflicts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "instructions.txt"
    path.write_text("Answer in Spanish.\n", encoding="utf-8")
    model = ScriptedModel(accepted("Catorce días.", "p1"))
    code = main(
        [
            "ask",
            "Refunds?",
            "--knowledge",
            "Refunds last 14 days after purchase.",
            "--instructions-file",
            str(path),
            "--quiet",
        ],
        client=model,
    )
    assert code == 0
    assert "Spanish" in str(model.calls[0]["user"])

    code = main(
        ["ask", "Refunds?", "--knowledge", "Refunds last 14 days.", "--instructions", "A", "--instructions-file", str(path)],
        client=ScriptedModel([]),
    )
    assert code == 2
    assert "either" in capsys.readouterr().err


def test_missing_project_is_a_user_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["ask", "Hello there"]) == 2
    assert "loopkit init" in capsys.readouterr().err


def test_serve_command_uses_the_loop(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, object] = {}

    def fake_serve(loop: object, host: str, port: int) -> None:
        seen["host"] = host
        seen["port"] = port
        seen["passages"] = len(getattr(loop, "knowledge").passages)

    monkeypatch.setattr("loopkit.cli.serve", fake_serve)
    code = main(
        ["serve", "--knowledge", "Refunds last 14 days after the charge.", "--port", "9"],
        client=ScriptedModel([]),
    )
    assert code == 0
    assert seen == {"host": "127.0.0.1", "port": 9, "passages": 1}
    capsys.readouterr()


def test_revision_trace_from_the_cli(capsys: pytest.CaptureFixture[str]) -> None:
    model = ScriptedModel(
        [
            Draft(answer="Unsure.", cited_ids=[]),
            audit(revise=True, notes="Name the window."),
            *accepted("14 days. [p1]", "p1"),
        ]
    )
    code = main(
        ["ask", "Refund window?", "--knowledge", "Refunds last 14 days after purchase.", "--quiet"],
        client=model,
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "14 days. [p1]"
