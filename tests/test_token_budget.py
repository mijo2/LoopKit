"""The token budget: estimate before a call, count after it, stop when it is spent.

``MeteredModel`` reports every call as 100 tokens in and 500 out, and the
loop is given an estimator that counts prompts as zero, so each call costs
600 on the meter and the only estimate before a call is the expected reply:
from the (empty) shape before a kind of step has been seen, which is 120 for
a draft, 100 for an instruction check, 200 for grounding, and 320 for the
combined check of four instructions; then the latest reply of that kind,
500. That keeps the arithmetic in these tests in round numbers.

Soft mode estimates a whole turn and lets a started turn finish. Hard mode
estimates one call at a time and stops the moment the budget is over.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from loopkit.cli import main
from loopkit.config import load_project
from loopkit.errors import LoopError
from loopkit.estimate import EstimationStrategy
from loopkit.loop import DEFAULT_INSTRUCTIONS, NOT_CHECKED_NOTE, Loop
from loopkit.schemas import Draft, RolePrefix
from tests.fakes import MeteredModel, ScriptedModel, accepted, audit, checks
from tests.test_serve import running

POLICY = "Refunds are available for 14 days after the charge. Keys are shown once."
CALL = 600  # what MeteredModel reports for one call: 100 in, 500 out
OUT = 500
# One whole turn of the default four instructions, estimated before any call:
# draft 120, four checks at 100 + the 120-token draft they read, grounding
# 200 + 120.
FIRST_TURN_ESTIMATE = 120 + 4 * (100 + 120) + (200 + 120)
# After a turn every kind has been seen at 500 tokens: draft 500, four
# checks at 500 + 500, grounding 500 + 500.
NEXT_TURN_ESTIMATE = 500 + 4 * (500 + 500) + (500 + 500)


def _zero(text: str) -> int:
    return 0


def _metered(replies: Sequence[BaseModel], **kwargs: Any) -> tuple[MeteredModel, Loop]:
    model = MeteredModel(list(replies), input_tokens=100, output_tokens=OUT)
    loop = Loop(knowledge=POLICY, client=model, trace=False, token_estimator=_zero, **kwargs)
    return model, loop


def _hard(replies: Sequence[BaseModel], **kwargs: Any) -> tuple[MeteredModel, Loop]:
    return _metered(replies, token_budget_mode="hard", **kwargs)


def test_no_budget_counts_tokens_and_never_stops_early() -> None:
    model = ScriptedModel(accepted("Refunds last 14 days. [p1]", "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False)
    answer = loop.ask("What is the refund window?")

    assert loop.token_budget is None
    assert loop.token_budget_mode == "soft"
    assert loop.estimation_strategy is EstimationStrategy.LAST_TURN
    assert answer.stopped_reason == "passed"
    assert answer.tokens.budget is None
    assert answer.tokens.calls == len(model.calls) == 1 + len(DEFAULT_INSTRUCTIONS) + 1
    # A client that reports no usage is counted from the prompt and reply text.
    assert answer.tokens.estimated is True
    assert answer.tokens.input_tokens > 0
    assert answer.tokens.output_tokens > 0
    assert answer.tokens.total == answer.tokens.input_tokens + answer.tokens.output_tokens
    assert answer.tokens.exceeded is False
    recorded = answer.to_dict()["tokens"]
    assert recorded == {
        "input": answer.tokens.input_tokens,
        "output": answer.tokens.output_tokens,
        "total": answer.tokens.total,
        "calls": answer.tokens.calls,
        "estimated": True,
        "budget": None,
    }


def test_reported_usage_is_what_the_answer_carries() -> None:
    _, loop = _metered(accepted("Refunds last 14 days. [p1]", "p1"), token_budget=100_000)
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "passed"
    assert answer.tokens.estimated is False
    assert answer.tokens.calls == 6
    assert answer.tokens.input_tokens == 600
    assert answer.tokens.output_tokens == 3000
    assert answer.tokens.budget == 100_000
    assert answer.tokens.exceeded is False


def test_soft_mode_skips_a_revision_turn_that_would_not_fit() -> None:
    # Turn 1 costs 6 calls = 3600 and always runs. The revision turn is then
    # estimated at 5500, which would reach 9100.
    model, loop = _metered(
        [
            Draft(answer="Refunds last 14 days.", cited_ids=[]),
            *checks(failed=(2,), notes="Add a citation."),
            Draft(answer="never reached", cited_ids=[]),
        ],
        token_budget=9000,
    )
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "token_budget"
    assert answer.turns == 1
    assert answer.text == "Refunds last 14 days."
    assert len(model.calls) == 6
    assert answer.tokens.total == 3600
    assert not answer.tokens.exceeded
    check = answer.checks[0]
    assert check.revise is True
    assert check.follows_instructions is False
    assert [item.followed for item in check.instructions] == [True, False, True, True]
    assert "Add a citation." in check.notes
    assert NOT_CHECKED_NOTE not in check.notes


def test_soft_mode_runs_a_revision_turn_that_fits() -> None:
    model, loop = _metered(
        [
            Draft(answer="Refunds last 14 days.", cited_ids=[]),
            *checks(failed=(2,), notes="Add a citation."),
            Draft(answer="Refunds last 14 days. [p1]", cited_ids=["p1"]),
            *checks(),
        ],
        token_budget=3600 + NEXT_TURN_ESTIMATE,
    )
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"
    assert answer.turns == 2
    assert len(model.calls) == 12
    assert answer.tokens.total == 7200


def test_soft_mode_always_finishes_the_turn_it_started(capsys: pytest.CaptureFixture[str]) -> None:
    # The first turn is estimated at 1320 and fits in 2500, but really costs
    # 3600. It still runs to the end, and the answer is fully checked.
    model, loop = _metered(accepted("Refunds last 14 days. [p1]", "p1"), token_budget=2500)
    loop.trace = True
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "passed"
    assert len(model.calls) == 6
    assert answer.tokens.total == 3600
    assert answer.tokens.exceeded
    assert answer.checks[0].follows_instructions is True
    assert answer.checks[0].grounded is True
    err = capsys.readouterr().err
    assert "loopkit: token budget: 2,500 (soft); first draft reply estimated at 120 tokens" in err
    assert "skipped" not in err
    assert "stop: passed" in err
    assert "tokens: 3,600 of 2,500 tokens used" in err


def test_soft_mode_runs_the_first_turn_even_when_it_is_too_big(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, loop = _metered(accepted("Refunds last 14 days. [p1]", "p1"), token_budget=500)
    loop.trace = True
    answer = loop.ask("What is the refund window?")

    assert answer.text == "Refunds last 14 days. [p1]"
    assert answer.stopped_reason == "passed"
    assert len(model.calls) == 6
    assert answer.tokens.exceeded
    err = capsys.readouterr().err
    assert (
        f"the first turn is estimated at {FIRST_TURN_ESTIMATE:,} tokens, over the budget of 500; "
        "it runs anyway so there is a checked answer to return"
    ) in err


def test_soft_mode_estimates_the_combined_check_as_one_call() -> None:
    replies = [
        Draft(answer="Refunds last 14 days.", cited_ids=[]),
        audit(revise=True, notes="Cite it.", follows=False),
        Draft(answer="Refunds last 14 days. [p1]", cited_ids=["p1"]),
        audit(revise=False),
    ]
    # Turn 1 costs 1200; the revision turn is estimated at 500 + (500 + 500).
    model, loop = _metered(replies, token_budget=1200 + 1500 - 1, check_mode="all")
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "token_budget"
    assert len(model.calls) == 2
    model, loop = _metered(replies, token_budget=1200 + 1500, check_mode="all")
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"
    assert len(model.calls) == 4


def test_hard_mode_skips_a_revision_that_would_not_fit() -> None:
    # Turn 1 costs 6 calls = 3600. Each check is allowed one at a time; the
    # revision draft, now expected at the 500 tokens the first one produced,
    # would reach 4100.
    model, loop = _hard(
        [
            Draft(answer="Refunds last 14 days.", cited_ids=[]),
            *checks(failed=(2,), notes="Add a citation."),
            Draft(answer="never reached", cited_ids=[]),
        ],
        token_budget=4000,
    )
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "token_budget"
    assert answer.turns == 1
    assert answer.text == "Refunds last 14 days."
    assert len(model.calls) == 6
    assert answer.tokens.total == 3600
    assert not answer.tokens.exceeded
    check = answer.checks[0]
    assert check.revise is True
    assert [item.followed for item in check.instructions] == [True, False, True, True]
    assert "Add a citation." in check.notes
    assert NOT_CHECKED_NOTE not in check.notes


def test_hard_mode_stops_the_moment_the_budget_is_over() -> None:
    model, loop = _hard(
        [
            Draft(answer="Refunds last 14 days.", cited_ids=[]),
            *checks(failed=(2,), notes="Add a citation."),
            Draft(answer="Refunds last 14 days. [p1]", cited_ids=["p1"]),
            *checks(),
        ],
        token_budget=4100,
    )
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "token_budget"
    assert answer.turns == 2
    assert answer.text == "Refunds last 14 days. [p1]"
    assert answer.citations[0].id == "p1"
    assert len(model.calls) == 7
    assert answer.tokens.total == 4200
    assert answer.tokens.exceeded
    assert len(answer.checks) == 2
    unchecked = answer.checks[1]
    assert unchecked.follows_instructions is False
    assert unchecked.grounded is False
    assert unchecked.revise is True
    assert [item.note for item in unchecked.instructions] == [NOT_CHECKED_NOTE] * 4
    assert "token budget ran out before the check of instruction 1" in unchecked.notes
    assert "grounding was not checked either" in unchecked.notes


def test_hard_mode_keeps_the_verdicts_a_cut_check_reached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Draft 600, check 1 at +100 expected, checks 2 and 3 at +500 each: 2400
    # used. Check 4 would reach 2900, over 2500.
    model, loop = _hard(accepted("Refunds last 14 days. [p1]", "p1"), token_budget=2500)
    loop.trace = True
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "token_budget"
    assert answer.turns == 1
    assert len(model.calls) == 4
    assert [call["schema"].__name__ for call in model.calls] == ["Draft"] + ["OneCheck"] * 3
    check = answer.checks[0]
    assert [item.followed for item in check.instructions] == [True, True, True, False]
    assert [item.note for item in check.instructions] == ["", "", "", NOT_CHECKED_NOTE]
    assert check.grounded is False
    assert "Instruction 4 was not judged" in check.notes
    assert "token budget ran out before the check of instruction 4" in check.notes
    err = capsys.readouterr().err
    assert "loopkit: token budget: 2,500 (hard); first draft reply estimated at 120 tokens" in err
    assert "turn 1: check instruction 3\n" in err
    assert "turn 1: check instruction 4 skipped: token budget (2,400 of 2,500 tokens used)" in err
    assert "instructions not checked: 4" in err
    assert "instructions not followed" not in err
    assert "stop: token budget" in err
    assert "tokens: 2,400 of 2,500 tokens used" in err


def test_hard_mode_makes_no_draft_when_the_first_would_not_fit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, loop = _hard(accepted("Refunds last 14 days. [p1]", "p1"), token_budget=100)
    loop.trace = True
    answer = loop.ask("What is the refund window?")

    assert answer.text == ""
    assert str(answer) == ""
    assert answer.turns == 0
    assert answer.checks == ()
    assert answer.citations == ()
    assert answer.stopped_reason == "token_budget"
    assert model.calls == []
    assert answer.tokens.total == 0
    assert answer.tokens.calls == 0
    assert answer.tokens.budget == 100
    err = capsys.readouterr().err
    assert "the first draft would not fit the token budget; nothing was drafted" in err
    assert "stop: token budget" in err
    assert "tokens: 0 of 100 tokens used" in err


def test_hard_mode_drafts_then_stops_when_the_draft_alone_was_too_much(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The draft is expected at 120 and fits in 500; it really costs 600.
    model, loop = _hard(accepted("Refunds last 14 days. [p1]", "p1"), token_budget=500)
    loop.trace = True
    answer = loop.ask("What is the refund window?")

    assert answer.text == "Refunds last 14 days. [p1]"
    assert answer.stopped_reason == "token_budget"
    assert len(model.calls) == 1
    assert answer.tokens.total == 600
    assert answer.tokens.exceeded
    assert [item.note for item in answer.checks[0].instructions] == [NOT_CHECKED_NOTE] * 4
    err = capsys.readouterr().err
    assert "check instruction 1 skipped: token budget (600 of 500 tokens used)" in err


def test_hard_mode_check_mode_all_is_one_guarded_call() -> None:
    replies = [Draft(answer="Refunds last 14 days. [p1]", cited_ids=["p1"]), audit(revise=False)]
    # Draft 600 used; the combined check of four instructions is expected at 320.
    model, loop = _hard(replies, token_budget=600 + 320 - 1, check_mode="all")
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "token_budget"
    assert len(model.calls) == 1
    assert "token budget ran out before the check" in answer.checks[0].notes
    assert [item.note for item in answer.checks[0].instructions] == [NOT_CHECKED_NOTE] * 4

    model, loop = _hard(replies, token_budget=600 + 320, check_mode="all")
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"
    assert len(model.calls) == 2


def test_optimize_is_skipped_when_it_would_not_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPKIT_AUTO_OPTIMIZE", "1")
    # The role reply is expected at 120, over 100: skipped in either mode.
    # Soft mode then runs the first turn anyway; hard mode drafts nothing.
    model, loop = _metered(accepted("Fourteen days. [p1]", "p1"), token_budget=100)
    answer = loop.ask("What is the refund window?")
    assert loop.auto_optimize is True
    assert model.calls[0]["schema"] is Draft
    assert len(model.calls) == 6
    assert answer.role == ""
    assert answer.stopped_reason == "passed"

    model, loop = _hard(accepted("Fourteen days. [p1]", "p1"), token_budget=100)
    answer = loop.ask("What is the refund window?")
    assert model.calls == []
    assert answer.role == ""
    assert answer.stopped_reason == "token_budget"

    model, loop = _hard(
        [RolePrefix(role="You are a billing specialist."), *accepted("Fourteen days. [p1]", "p1")],
        token_budget=500,
    )
    answer = loop.ask("What is the refund window?")
    assert model.calls[0]["schema"] is RolePrefix
    assert answer.role == "You are a billing specialist."
    # The role call used 600 of 500: nothing else is made.
    assert len(model.calls) == 1
    assert answer.turns == 0
    assert answer.stopped_reason == "token_budget"


def test_ask_can_override_the_budget_for_one_task() -> None:
    model, loop = _hard(accepted("Refunds last 14 days. [p1]", "p1"), token_budget=100_000)
    answer = loop.ask("What is the refund window?", token_budget=2500)
    assert answer.stopped_reason == "token_budget"
    assert answer.tokens.budget == 2500
    assert len(model.calls) == 4
    assert loop.token_budget == 100_000

    model, loop = _hard(accepted("Refunds last 14 days. [p1]", "p1"), token_budget=2500)
    answer = loop.ask("What is the refund window?", token_budget=100_000)
    assert answer.stopped_reason == "passed"
    assert answer.tokens.budget == 100_000


def test_a_custom_estimator_drives_the_estimates() -> None:
    seen: list[str] = []

    def counting(text: str) -> int:
        seen.append(text)
        return len(text)

    model = ScriptedModel(accepted("Refunds last 14 days. [p1]", "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False, token_estimator=counting)
    answer = loop.ask("What is the refund window?")
    assert seen
    assert answer.tokens.estimated is True
    prompts = sum(len(call["system"]) + len(call["user"]) for call in model.calls)
    assert answer.tokens.input_tokens == prompts


def test_budget_validation() -> None:
    for bad in (0, -5, True, "1000", 2.5):
        with pytest.raises(LoopError, match="token_budget must be an integer of at least 1"):
            Loop(knowledge=POLICY, client=ScriptedModel([]), token_budget=bad)  # type: ignore[arg-type]
    loop = Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False)
    with pytest.raises(LoopError, match="token_budget must be an integer of at least 1"):
        loop.ask("Refunds?", token_budget=0)
    with pytest.raises(LoopError, match="token_estimator must be a function"):
        Loop(knowledge=POLICY, client=ScriptedModel([]), token_estimator="len")  # type: ignore[arg-type]
    with pytest.raises(LoopError, match="token_budget_mode must be one of: soft, hard"):
        Loop(knowledge=POLICY, client=ScriptedModel([]), token_budget_mode="strict")
    with pytest.raises(LoopError, match="estimation_strategy must be one of: last_turn"):
        Loop(knowledge=POLICY, client=ScriptedModel([]), estimation_strategy="guess")
    loop = Loop(
        knowledge=POLICY,
        client=ScriptedModel([]),
        token_budget_mode=" Hard ",
        estimation_strategy="last_turn",
    )
    assert loop.token_budget_mode == "hard"
    assert loop.estimation_strategy is EstimationStrategy.LAST_TURN


def test_budget_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPKIT_TOKEN_BUDGET", "5,000")
    loop = Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False)
    assert loop.token_budget == 5000
    # An explicit argument wins over the variable.
    loop = Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False, token_budget=7)
    assert loop.token_budget == 7
    monkeypatch.setenv("LOOPKIT_TOKEN_BUDGET", "lots")
    with pytest.raises(LoopError, match="LOOPKIT_TOKEN_BUDGET must be an integer of at least 1"):
        Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False)
    monkeypatch.setenv("LOOPKIT_TOKEN_BUDGET", "0")
    with pytest.raises(LoopError, match="LOOPKIT_TOKEN_BUDGET must be an integer of at least 1"):
        Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False)
    monkeypatch.setenv("LOOPKIT_TOKEN_BUDGET", "")
    assert Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False).token_budget is None
    monkeypatch.setenv("LOOPKIT_TOKEN_BUDGET_MODE", "hard")
    assert Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False).token_budget_mode == "hard"
    loop = Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False, token_budget_mode="soft")
    assert loop.token_budget_mode == "soft"
    monkeypatch.setenv("LOOPKIT_TOKEN_BUDGET_MODE", "medium")
    with pytest.raises(LoopError, match="token_budget_mode must be one of: soft, hard"):
        Loop(knowledge=POLICY, client=ScriptedModel([]), trace=False)


def test_budget_from_the_project_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "knowledge.md").write_text(POLICY, encoding="utf-8")
    (tmp_path / "loopkit.toml").write_text(
        'knowledge = "knowledge.md"\ntoken_budget = 2500\ntoken_budget_mode = "hard"\n',
        encoding="utf-8",
    )
    project = load_project(tmp_path / "loopkit.toml")
    assert project.token_budget == 2500
    assert project.token_budget_mode == "hard"
    monkeypatch.chdir(tmp_path)
    loop = Loop.load(client=ScriptedModel([]), trace=False)
    assert loop.token_budget == 2500
    assert loop.token_budget_mode == "hard"
    loop = Loop.load(
        client=ScriptedModel([]), trace=False, token_budget=9, token_budget_mode="soft"
    )
    assert loop.token_budget == 9
    assert loop.token_budget_mode == "soft"
    with pytest.raises(LoopError, match="token_budget must be an integer of at least 1"):
        Loop.load(client=ScriptedModel([]), trace=False, token_budget=0)

    (tmp_path / "loopkit.toml").write_text(
        'knowledge = "knowledge.md"\ntoken_budget = "2500"\n', encoding="utf-8"
    )
    with pytest.raises(LoopError, match=r"token_budget in loopkit\.toml must be an integer"):
        load_project(tmp_path / "loopkit.toml")
    (tmp_path / "loopkit.toml").write_text(
        'knowledge = "knowledge.md"\ntoken_budget_mode = "firm"\n', encoding="utf-8"
    )
    with pytest.raises(LoopError, match=r"token_budget_mode in loopkit\.toml must be one of"):
        load_project(tmp_path / "loopkit.toml")


def test_cli_token_budget(capsys: pytest.CaptureFixture[str]) -> None:
    model = MeteredModel(accepted("Fourteen days. [p1]", "p1"), input_tokens=100, output_tokens=OUT)
    code = main(
        [
            "ask",
            "What is the refund window?",
            "--knowledge",
            POLICY,
            "--token-budget",
            "100000",
            "--json",
        ],
        client=model,
    )
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["stopped_reason"] == "passed"
    assert payload["tokens"] == {
        "input": 600,
        "output": 3000,
        "total": 3600,
        "calls": 6,
        "estimated": False,
        "budget": 100000,
    }
    assert "loopkit: token budget: 100,000 (soft)" in captured.err
    assert "loopkit: tokens: 3,600 of 100,000 tokens used" in captured.err

    code = main(
        ["ask", "Refunds?", "--knowledge", POLICY, "--token-budget", "0"],
        client=ScriptedModel([]),
    )
    assert code == 2
    assert "token_budget must be an integer of at least 1" in capsys.readouterr().err


def test_cli_hard_mode(capsys: pytest.CaptureFixture[str]) -> None:
    model = MeteredModel(accepted("Fourteen days. [p1]", "p1"), input_tokens=100, output_tokens=OUT)
    code = main(
        [
            "ask",
            "What is the refund window?",
            "--knowledge",
            POLICY,
            "--token-budget",
            "2500",
            "--token-budget-mode",
            "hard",
        ],
        client=model,
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Fourteen days. [p1]"
    assert "(hard)" in captured.err
    assert "stop: token budget" in captured.err
    assert len(model.calls) < 6

    # Nothing drafted: nothing on stdout, a line on stderr, still exit 0.
    model = MeteredModel(accepted("Fourteen days. [p1]", "p1"), input_tokens=100, output_tokens=OUT)
    code = main(
        [
            "ask",
            "Refunds?",
            "--knowledge",
            POLICY,
            "--token-budget",
            "5",
            "--token-budget-mode",
            "hard",
        ],
        client=model,
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "loopkit: no answer: the token budget was too small for the first draft." in captured.err
    assert model.calls == []

    # With --json the (empty) answer object is printed as usual.
    model = MeteredModel(accepted("Fourteen days. [p1]", "p1"), input_tokens=100, output_tokens=OUT)
    code = main(
        [
            "ask",
            "Refunds?",
            "--knowledge",
            POLICY,
            "--token-budget",
            "5",
            "--token-budget-mode",
            "hard",
            "--json",
        ],
        client=model,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == ""
    assert payload["turns"] == 0
    assert payload["stopped_reason"] == "token_budget"
    assert payload["tokens"]["calls"] == 0


def test_serve_accepts_a_budget_per_request() -> None:
    model = MeteredModel(
        accepted("Refunds last 14 days. [p1]", "p1"), input_tokens=100, output_tokens=OUT
    )
    loop = Loop(
        knowledge=POLICY, client=model, trace=False, token_estimator=_zero, token_budget_mode="hard"
    )
    with running(loop) as base:
        status, body = _post(base, {"task": "What is the refund window?", "token_budget": 2500})
        assert status == 200
        assert body["stopped_reason"] == "token_budget"
        assert body["tokens"]["budget"] == 2500
        assert body["tokens"]["total"] == 2400
        status, body = _post(base, {"task": "What is the refund window?", "token_budget": "2500"})
        assert status == 400
        assert "token_budget must be an integer" in body["error"]
        status, body = _post(base, {"task": "What is the refund window?", "token_budget": 0})
        assert status == 400


def _post(base: str, payload: dict[str, object]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        base + "/ask",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read())
            assert isinstance(body, dict)
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read())
        assert isinstance(body, dict)
        return exc.code, body
