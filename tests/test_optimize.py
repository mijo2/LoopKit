"""The optimize step: one call before the loop that writes the role."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopkit.cli import main
from loopkit.errors import LoopError, LoopWarning
from loopkit.loop import Loop
from loopkit.prompts import OPTIMIZE_SYSTEM
from loopkit.schemas import Draft, RolePrefix
from tests.fakes import ScriptedModel, accepted, checks

POLICY = "Refunds are available for 30 days. After that the sale is final."
ROLE = "You are a consumer-billing support specialist who knows this company's refund policy. You quote the exact window and conditions."


@pytest.fixture
def optimize_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOOPKIT_AUTO_OPTIMIZE", raising=False)


def test_on_by_default_and_prefixes_every_later_call(optimize_on: None) -> None:
    model = ScriptedModel([RolePrefix(role=ROLE), *accepted("Thirty days. [p1]", "p1", instructions=2)])
    loop = Loop(knowledge=POLICY, instructions=["Be brief.", "Cite [p1]."], client=model, trace=False)
    assert loop.auto_optimize is True
    answer = loop.ask("What is the refund window?")

    assert answer.role == ROLE
    assert answer.to_dict()["role"] == ROLE
    # 1 optimize + 1 draft + 2 instruction checks + 1 grounding
    assert len(model.calls) == 5
    first = model.calls[0]
    assert first["schema"] is RolePrefix
    assert first["system"] == OPTIMIZE_SYSTEM
    assert "TASK:\nWhat is the refund window?" in str(first["user"])
    assert "1. Be brief." in str(first["user"])
    assert "DOCUMENTS:\ntext" in str(first["user"])
    assert "Refunds are available" not in str(first["user"])  # the document list, not the documents
    for call in model.calls[1:]:
        assert str(call["system"]).startswith(ROLE + "\n\n"), call["system"]
        assert ROLE not in str(call["user"])


def test_open_mode_optimize_prompt_has_no_document_list(optimize_on: None) -> None:
    model = ScriptedModel([RolePrefix(role="You are a scholar of early Buddhist texts."), *accepted("...", instructions=3)])
    with pytest.warns(LoopWarning):
        loop = Loop(client=model, trace=False)
    loop.ask("Explain Dhammapada verse 1.")
    prompt = str(model.calls[0]["user"])
    assert "DOCUMENTS" not in prompt
    assert "passage" not in prompt.lower()
    assert str(model.calls[1]["system"]).startswith("You are a scholar of early Buddhist texts.\n\n")


def test_role_whitespace_is_collapsed_and_empty_role_means_no_prefix(optimize_on: None) -> None:
    model = ScriptedModel([RolePrefix(role="  You are\n a   tester.  "), *accepted("Ok. [p1]", "p1", instructions=1)])
    loop = Loop(knowledge=POLICY, instructions=["Be brief."], client=model, trace=False)
    assert loop.ask("Refunds?").role == "You are a tester."

    model = ScriptedModel([RolePrefix(role="   "), *accepted("Ok. [p1]", "p1", instructions=1)])
    loop = Loop(knowledge=POLICY, instructions=["Be brief."], client=model, trace=False)
    answer = loop.ask("Refunds?")
    assert answer.role == ""
    assert str(model.calls[1]["system"]).startswith("You draft an answer")


def test_a_failed_optimize_step_does_not_stop_the_answer(optimize_on: None, caplog: pytest.LogCaptureFixture) -> None:
    class Flaky(ScriptedModel):
        def parse(self, *, system_text: str, user_text: str, schema: type[object]) -> object:
            if schema is RolePrefix:
                self.calls.append({"system": system_text, "user": user_text, "schema": schema})
                raise LoopError("The model API returned HTTP 500: boom")
            return super().parse(system_text=system_text, user_text=user_text, schema=schema)

    model = Flaky(accepted("Thirty days. [p1]", "p1", instructions=1))
    loop = Loop(knowledge=POLICY, instructions=["Be brief."], client=model, trace=False)
    with caplog.at_level("WARNING", logger="loopkit"):
        answer = loop.ask("Refund window?")
    assert answer.text == "Thirty days. [p1]"
    assert answer.role == ""
    assert any("optimize step failed" in record.message for record in caplog.records)
    assert str(model.calls[1]["system"]).startswith("You draft an answer")


def test_off_switches(optimize_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = ScriptedModel(accepted("Ok. [p1]", "p1", instructions=1))
    loop = Loop(knowledge=POLICY, instructions=["Be brief."], client=model, auto_optimize=False, trace=False)
    assert loop.ask("Refunds?").role == ""
    assert model.calls[0]["schema"] is Draft

    monkeypatch.setenv("LOOPKIT_AUTO_OPTIMIZE", "false")
    assert Loop(knowledge=POLICY, client=ScriptedModel([])).auto_optimize is False
    monkeypatch.setenv("LOOPKIT_AUTO_OPTIMIZE", "garbage")
    assert Loop(knowledge=POLICY, client=ScriptedModel([])).auto_optimize is True
    monkeypatch.delenv("LOOPKIT_AUTO_OPTIMIZE")

    (tmp_path / "notes.md").write_text("Desks ship in two weeks.\n", encoding="utf-8")
    (tmp_path / "loopkit.toml").write_text('knowledge = "notes.md"\nauto_optimize = false\n', encoding="utf-8")
    assert Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([])).auto_optimize is False
    (tmp_path / "loopkit.toml").write_text('knowledge = "notes.md"\nauto_optimize = "no"\n', encoding="utf-8")
    with pytest.raises(LoopError, match="true or false"):
        Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))


def test_cli_trace_json_and_flag(optimize_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    model = ScriptedModel([RolePrefix(role=ROLE), *accepted("Thirty days. [p1]", "p1", instructions=1)])
    code = main(["ask", "Refund window?", "--knowledge", POLICY, "--instructions", "Be brief.", "--json", "--debug"], client=model)
    assert code == 0
    captured = capsys.readouterr()
    assert "loopkit: optimize: writing the role for this task" in captured.err
    assert "loopkit: role: You are a consumer-billing support specialist" in captured.err
    assert json.loads(captured.out)["role"] == ROLE
    log = (tmp_path / "loopkit-debug.log").read_text(encoding="utf-8")
    assert "CALL 1: optimize" in log
    assert "RESULT 1: optimize -> ok" in log
    assert "auto_optimize=True" in log
    assert "--- ROLE ---\n" + ROLE in log
    from loopkit.debuglog import disable_debug_log

    disable_debug_log(tmp_path / "loopkit-debug.log")

    model = ScriptedModel(accepted("Thirty days. [p1]", "p1", instructions=1))
    code = main(["ask", "Refund window?", "--knowledge", POLICY, "--instructions", "Be brief.", "--quiet", "--no-auto-optimize"], client=model)
    assert code == 0
    assert model.calls[0]["schema"] is Draft
    assert checks  # imported for readers who extend these scripts
