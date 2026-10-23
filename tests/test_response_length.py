"""Response length control: a target from the estimate or the caller, measured in code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopkit.cli import main
from loopkit.config import load_project
from loopkit.errors import LoopError
from loopkit.estimate import (
    CHARS_PER_TOKEN,
    MIN_RESPONSE_CHARS,
    AskShape,
    first_draft,
    first_output,
)
from loopkit.loop import Loop
from loopkit.prompts import length_rule
from loopkit.schemas import Draft
from tests.fakes import ScriptedModel, accepted, checks

POLICY = "Refunds are available for 14 days after the charge. Keys are shown once."
LONG = "Refunds last 14 days. " * 40 + "[p1]"  # 884 characters
SHORT = "Refunds last 14 days. [p1]"


def test_off_by_default() -> None:
    model = ScriptedModel(accepted(LONG, "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False)
    answer = loop.ask("What is the refund window?")
    assert loop.response_length is None
    assert answer.response_length is None
    assert answer.checks[0].length is None
    assert "RESPONSE LIMIT" not in str(model.calls[0]["system"])
    assert "RESPONSE LIMIT" not in str(model.calls[0]["user"])
    recorded = answer.to_dict()
    assert recorded["response_length"] is None
    assert isinstance(recorded["checks"], list)
    assert recorded["checks"][0]["length"] is None


def test_an_explicit_length_is_asked_for_measured_and_enforced(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ScriptedModel(
        [
            Draft(answer=LONG, cited_ids=["p1"]),
            *checks(),
            Draft(answer=SHORT, cited_ids=["p1"]),
            *checks(),
        ]
    )
    loop = Loop(knowledge=POLICY, client=model, trace=True, response_length=300)
    answer = loop.ask("What is the refund window?")

    assert loop.response_length == 300
    assert answer.response_length == 300
    assert answer.stopped_reason == "passed"
    assert answer.turns == 2
    assert answer.text == SHORT
    # The limit is a rule in the system prompt, not part of the task.
    first = str(model.calls[0]["system"])
    assert length_rule(300, 50) in first
    assert "under 300 characters, about 50 words" in first
    assert "RESPONSE LIMIT" not in str(model.calls[0]["user"])
    # The checkers do not see it; length is judged in code.
    assert "RESPONSE LIMIT" not in str(model.calls[1]["system"])
    # Turn 1: every instruction held and grounding passed, but the length did not.
    over = answer.checks[0]
    assert over.follows_instructions is True
    assert over.grounded is True
    assert over.revise is True
    assert over.length is not None
    assert over.length.limit == 300
    assert over.length.actual == len(LONG)
    assert over.length.followed is False
    assert f"The answer is {len(LONG):,} characters; keep it under 300 characters" in over.notes
    # The revision prompt carried the note, and its system prompt the limit.
    revision = model.calls[len(checks()) + 1]
    assert "keep it under 300 characters" in str(revision["user"])
    assert length_rule(300, 50) in str(revision["system"])
    # Turn 2 is within the limit.
    within = answer.checks[1]
    assert within.length is not None
    assert within.length.actual == len(SHORT)
    assert within.length.followed is True
    assert within.revise is False
    recorded = answer.to_dict()["checks"]
    assert isinstance(recorded, list)
    assert recorded[0]["length"] == {"limit": 300, "actual": len(LONG), "followed": False}
    err = capsys.readouterr().err
    assert "response length: under 300 characters, about 50 words" in err
    assert f"turn 1: response length {len(LONG)} of 300 characters, over the limit" in err
    assert f"turn 2: response length {len(SHORT)} of 300 characters, within the limit" in err


def test_the_tolerance_lets_a_small_overrun_pass() -> None:
    text = "x" * 325 + " [p1]"  # 330 characters: exactly 300 * 1.1
    model = ScriptedModel(accepted(text, "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False, response_length=300)
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"
    assert answer.checks[0].length is not None
    assert answer.checks[0].length.followed is True


def test_still_over_at_the_last_turn_stops_on_max_turns() -> None:
    model = ScriptedModel([Draft(answer=LONG, cited_ids=["p1"]), *checks()] * 2)
    loop = Loop(knowledge=POLICY, client=model, trace=False, response_length=300, max_turns=2)
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "max_turns"
    assert answer.text == LONG
    assert all(check.length is not None and not check.length.followed for check in answer.checks)


def test_auto_takes_the_length_from_the_estimate_step() -> None:
    model = ScriptedModel(accepted(SHORT, "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False, response_length="auto")
    answer = loop.ask("What is the refund window?")
    assert loop.response_length == "auto"
    shape = AskShape(
        task_tokens=loop.token_estimator("What is the refund window?"),
        instruction_count=len(loop.instructions),
        passage_tokens=sum(loop.token_estimator(p.text) for p in loop.knowledge.passages),
    )
    expected = (first_draft(shape) - 30) * CHARS_PER_TOKEN
    assert answer.response_length == max(MIN_RESPONSE_CHARS, expected)
    assert answer.response_length is not None
    assert length_rule(answer.response_length, round(answer.response_length / 6)) in str(
        model.calls[0]["system"]
    )
    assert answer.checks[0].length is not None
    assert answer.checks[0].length.followed is True


def test_the_target_is_held_under_what_the_model_returns_in_one_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ScriptedModel(accepted(SHORT, "p1"))
    loop = Loop(
        knowledge=POLICY,
        client=model,
        trace=True,
        model="claude-sonnet-5-5",
        response_length=10_000_000,
    )
    answer = loop.ask("What is the refund window?")
    assert answer.response_length == 64_000 * CHARS_PER_TOKEN
    err = capsys.readouterr().err
    assert (
        "response length 10,000,000 is over what claude-sonnet-5-5 returns in one call; "
        "held to 256,000 characters"
    ) in err

    # An unknown model gets the default, and a client with an output cap lowers it.
    class Capped(ScriptedModel):
        max_output_tokens = 1_030

    model = Capped(accepted(SHORT, "p1"))
    loop = Loop(
        knowledge=POLICY, client=model, trace=False, model="llama3.1", response_length="auto"
    )
    loop.response_length = 10_000_000
    answer = loop.ask("What is the refund window?")
    assert answer.response_length == 1_000 * CHARS_PER_TOKEN


def test_the_budget_expects_the_draft_to_be_the_length_asked_for() -> None:
    model = ScriptedModel(accepted(SHORT, "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False, response_length=4_000)
    loop.ask("What is the refund window?")
    assert loop._meter.shape.response_chars == 4_000
    # 4000 characters is 1000 tokens of answer plus the JSON around it.
    assert first_output("Draft", loop._meter.shape) == 1_030


def test_validation_env_and_project_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for bad in (0, -1, "none", "1.5", True, 2.5):
        with pytest.raises(LoopError, match='response_length must be "auto" or an integer'):
            Loop(knowledge=POLICY, client=ScriptedModel([]), response_length=bad)  # type: ignore[arg-type]
    assert (
        Loop(knowledge=POLICY, client=ScriptedModel([]), response_length=" Auto ").response_length
        == "auto"
    )
    assert (
        Loop(knowledge=POLICY, client=ScriptedModel([]), response_length="1,200").response_length
        == 1200
    )

    monkeypatch.setenv("LOOPKIT_RESPONSE_LENGTH", "auto")
    assert Loop(knowledge=POLICY, client=ScriptedModel([])).response_length == "auto"
    monkeypatch.setenv("LOOPKIT_RESPONSE_LENGTH", "800")
    assert Loop(knowledge=POLICY, client=ScriptedModel([])).response_length == 800
    assert Loop(knowledge=POLICY, client=ScriptedModel([]), response_length=5).response_length == 5
    monkeypatch.setenv("LOOPKIT_RESPONSE_LENGTH", "long")
    with pytest.raises(LoopError, match='response_length must be "auto" or an integer'):
        Loop(knowledge=POLICY, client=ScriptedModel([]))
    monkeypatch.delenv("LOOPKIT_RESPONSE_LENGTH")

    (tmp_path / "knowledge.md").write_text(POLICY, encoding="utf-8")
    toml = tmp_path / "loopkit.toml"
    toml.write_text('knowledge = "knowledge.md"\nresponse_length = "auto"\n', encoding="utf-8")
    assert load_project(toml).response_length == "auto"
    toml.write_text('knowledge = "knowledge.md"\nresponse_length = 600\n', encoding="utf-8")
    assert load_project(toml).response_length == 600
    monkeypatch.chdir(tmp_path)
    assert Loop.load(client=ScriptedModel([])).response_length == 600
    assert Loop.load(client=ScriptedModel([]), response_length="auto").response_length == "auto"
    toml.write_text('knowledge = "knowledge.md"\nresponse_length = "short"\n', encoding="utf-8")
    with pytest.raises(
        LoopError, match=r'response_length in loopkit\.toml must be "auto" or an integer'
    ):
        load_project(toml)
    toml.write_text('knowledge = "knowledge.md"\nresponse_length = 0\n', encoding="utf-8")
    with pytest.raises(LoopError, match=r"response_length in loopkit\.toml"):
        load_project(toml)


def test_cli_response_length(capsys: pytest.CaptureFixture[str]) -> None:
    model = ScriptedModel(
        [
            Draft(answer=LONG, cited_ids=["p1"]),
            *checks(),
            Draft(answer=SHORT, cited_ids=["p1"]),
            *checks(),
        ]
    )
    code = main(
        ["ask", "What is the refund window?", "-k", POLICY, "--response-length", "300", "--json"],
        client=model,
    )
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["response_length"] == 300
    assert payload["turns"] == 2
    assert payload["text"] == SHORT
    assert "response length: under 300 characters" in captured.err

    model = ScriptedModel(accepted(SHORT, "p1"))
    code = main(
        ["ask", "Refunds?", "-k", POLICY, "--response-length", "auto", "--quiet"], client=model
    )
    assert code == 0
    assert "RESPONSE LIMIT" in str(model.calls[0]["system"])
    capsys.readouterr()

    code = main(
        ["ask", "Refunds?", "-k", POLICY, "--response-length", "brief"], client=ScriptedModel([])
    )
    assert code == 2
    assert 'response_length must be "auto" or an integer' in capsys.readouterr().err
