import io
import json
from pathlib import Path

import pytest

from loopkit.cli import main
from loopkit.loop import Loop
from loopkit.schemas import Draft
from tests.fakes import ScriptedModel, accepted, checks


def test_init_then_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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
    assert (tmp_path / "loopkit.toml").read_text(encoding="utf-8").count(
        'knowledge = "knowledge"'
    ) == 1


def test_quiet_json_and_inline_knowledge(capsys: pytest.CaptureFixture[str]) -> None:
    model = ScriptedModel(accepted("Fourteen days. [p1]", "p1", instructions=1))
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
    model = ScriptedModel(accepted("Fourteen days. [p1]", "p1", instructions=2))
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


def test_instructions_file_and_conflicts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "instructions.txt"
    path.write_text("Answer in Spanish.\n", encoding="utf-8")
    model = ScriptedModel(accepted("Catorce días.", "p1", instructions=1))
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
        [
            "ask",
            "Refunds?",
            "--knowledge",
            "Refunds last 14 days.",
            "--instructions",
            "A",
            "--instructions-file",
            str(path),
        ],
        client=ScriptedModel([]),
    )
    assert code == 2
    assert "either" in capsys.readouterr().err


def test_missing_project_is_a_user_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    model = ScriptedModel(accepted("Hello. How can I help?", instructions=3))
    assert main(["ask", "Hello there", "--quiet"], client=model) == 0
    captured = capsys.readouterr()
    assert "no knowledge was given" in captured.err
    assert "loopkit init" in captured.err
    assert "Hello. How can I help?" in captured.out
    prompt = str(model.calls[0]["user"])
    assert "passage" not in prompt.lower()
    assert "knowledge" not in prompt.lower()


def test_task_from_stdin_and_file_keeps_backticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task = "What does `ls -la` do?\n```sh\nls -la\n```"
    knowledge = "ls lists files. The -la flags show all files in long form."

    monkeypatch.setattr("sys.stdin", io.StringIO(task))
    model = ScriptedModel(accepted("It lists files. [p1]", "p1"))
    assert main(["ask", "--knowledge", knowledge, "--quiet"], client=model) == 0
    assert task in str(model.calls[0]["user"])

    monkeypatch.setattr("sys.stdin", io.StringIO(task))
    model = ScriptedModel(accepted("It lists files. [p1]", "p1"))
    assert main(["ask", "-", "--knowledge", knowledge, "--quiet"], client=model) == 0
    assert task in str(model.calls[0]["user"])

    task_file = tmp_path / "task.txt"
    task_file.write_text(task, encoding="utf-8")
    model = ScriptedModel(accepted("It lists files. [p1]", "p1"))
    assert (
        main(
            ["ask", "--task-file", str(task_file), "--knowledge", knowledge, "--quiet"],
            client=model,
        )
        == 0
    )
    assert task in str(model.calls[0]["user"])

    assert (
        main(
            ["ask", "also here", "--task-file", str(task_file), "--knowledge", knowledge],
            client=model,
        )
        == 2
    )
    assert "not both" in capsys.readouterr().err

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty())
    assert main(["ask", "--knowledge", knowledge], client=model) == 2
    assert "Give a task" in capsys.readouterr().err


def test_debug_flag_writes_every_call_to_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    model = ScriptedModel(
        [
            Draft(answer="Unsure.", cited_ids=[]),
            *checks(revise=True, notes="Name the window."),
            *accepted("14 days. [p1]", "p1"),
        ]
    )
    code = main(
        [
            "ask",
            "Refund window?",
            "--knowledge",
            "Refunds last 14 days after purchase.",
            "--quiet",
            "--debug",
        ],
        client=model,
    )
    assert code == 0
    assert "debug log:" in capsys.readouterr().err
    log = (tmp_path / "loopkit-debug.log").read_text(encoding="utf-8")
    assert "] ASK" in log
    assert "CALL 1: draft (turn 1)" in log
    assert "--- SYSTEM PROMPT ---" in log
    assert "--- USER PROMPT ---" in log
    assert "RESULT 1: draft (turn 1) -> ok" in log
    assert "CALL 2: check instruction 1 (turn 1)" in log
    assert "check grounding (turn 1)" in log
    assert "TURN 1 DECISION: revise" in log
    assert "Name the window." in log
    assert "TURN 2 DECISION: passed" in log
    assert "] ANSWER" in log
    assert "14 days. [p1]" in log
    from loopkit.debuglog import disable_debug_log

    disable_debug_log(tmp_path / "loopkit-debug.log")


def test_serve_command_uses_the_loop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, object] = {}

    def fake_serve(loop: Loop, host: str, port: int, session: str | None = None) -> None:
        seen["host"] = host
        seen["port"] = port
        seen["session"] = session
        seen["passages"] = len(loop.knowledge.passages)

    monkeypatch.setattr("loopkit.cli.serve", fake_serve)
    code = main(
        ["serve", "--knowledge", "Refunds last 14 days after the charge.", "--port", "9"],
        client=ScriptedModel([]),
    )
    assert code == 0
    assert seen == {"host": "127.0.0.1", "port": 9, "session": None, "passages": 1}

    code = main(
        ["serve", "--knowledge", "Refunds last 14 days.", "--session"],
        client=ScriptedModel([]),
    )
    assert code == 0
    assert seen.get("session") == "loopkit-session.json"

    code = main(
        ["serve", "--knowledge", "Refunds last 14 days.", "--session", "runs/chats.json"],
        client=ScriptedModel([]),
    )
    assert code == 0
    assert seen.get("session") == "runs/chats.json"
    capsys.readouterr()


def test_revision_trace_from_the_cli(capsys: pytest.CaptureFixture[str]) -> None:
    model = ScriptedModel(
        [
            Draft(answer="Unsure.", cited_ids=[]),
            *checks(revise=True, notes="Name the window."),
            *accepted("14 days. [p1]", "p1"),
        ]
    )
    code = main(
        ["ask", "Refund window?", "--knowledge", "Refunds last 14 days after purchase.", "--quiet"],
        client=model,
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "14 days. [p1]"
