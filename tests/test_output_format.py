"""Structured output: the schema in the system prompt, the draft validated in code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopkit.cli import main
from loopkit.config import load_project
from loopkit.errors import LoopError
from loopkit.loop import DEFAULT_INSTRUCTIONS, JSON_CITE_INSTRUCTION, Loop, default_instructions
from loopkit.prompts import JSON_CHECK_RULE, JSON_DRAFT_RULE
from loopkit.schemas import Draft
from loopkit.structure import infer_schema, render_schema
from tests.fakes import ScriptedModel, accepted, checks

POLICY = "Refunds are available for 14 days after the charge. Keys are shown once."
EXAMPLE = {"refund_days": 14, "eligible": True}
SCHEMA = infer_schema(EXAMPLE)
GOOD = '{"refund_days": 14, "eligible": true}'
BAD = '{"refund_days": "fourteen", "eligible": true, "note": "x"}'


def test_text_is_the_default_and_a_schema_implies_json() -> None:
    loop = Loop(knowledge=POLICY, client=ScriptedModel([]))
    assert loop.output_format == "text"
    assert loop.output_schema is None
    assert loop.instructions == DEFAULT_INSTRUCTIONS

    loop = Loop(knowledge=POLICY, client=ScriptedModel([]), output_schema=EXAMPLE)
    assert loop.output_format == "json"
    assert loop.output_schema == SCHEMA
    assert loop.instructions[1] == JSON_CITE_INSTRUCTION
    assert loop.instructions[0] == DEFAULT_INSTRUCTIONS[0]
    assert default_instructions("open", "normal", output="json") == default_instructions(
        "open", "normal"
    )

    with pytest.raises(LoopError, match='output_format="json" needs output_schema'):
        Loop(knowledge=POLICY, client=ScriptedModel([]), output_format="json")
    with pytest.raises(LoopError, match='output_schema needs output_format="json"'):
        Loop(
            knowledge=POLICY, client=ScriptedModel([]), output_format="text", output_schema=EXAMPLE
        )
    with pytest.raises(LoopError, match="output_format must be one of: text, json"):
        Loop(knowledge=POLICY, client=ScriptedModel([]), output_format="yaml")
    with pytest.raises(LoopError, match="not valid JSON"):
        Loop(knowledge=POLICY, client=ScriptedModel([]), output_schema="{nope")


def test_a_compliant_document_passes_and_is_parsed() -> None:
    model = ScriptedModel(accepted(GOOD, "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False, output_schema=EXAMPLE)
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "passed"
    assert answer.output_format == "json"
    assert answer.text == GOOD
    assert answer.data == {"refund_days": 14, "eligible": True}
    assert answer.citations[0].id == "p1"
    check = answer.checks[0]
    assert check.structure is not None
    assert check.structure.followed is True
    assert check.structure.problems == ()
    assert check.length is None
    # The schema and the rule are in the drafter's system prompt, not the task.
    system = str(model.calls[0]["system"])
    assert JSON_DRAFT_RULE in system
    assert "OUTPUT SCHEMA:\n" + render_schema(SCHEMA) in system
    assert "OUTPUT SCHEMA" not in str(model.calls[0]["user"])
    # The checkers are told the draft is JSON by design.
    assert JSON_CHECK_RULE in str(model.calls[1]["system"])
    assert JSON_CHECK_RULE in str(model.calls[-1]["system"])
    recorded = answer.to_dict()
    assert recorded["output_format"] == "json"
    assert recorded["data"] == {"refund_days": 14, "eligible": True}
    assert isinstance(recorded["checks"], list)
    assert recorded["checks"][0]["structure"] == {"followed": True, "problems": []}


def test_a_draft_off_the_schema_is_revised_with_the_exact_problems(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ScriptedModel(
        [
            Draft(answer=BAD, cited_ids=["p1"]),
            *checks(),
            Draft(answer=GOOD, cited_ids=["p1"]),
            *checks(),
        ]
    )
    loop = Loop(knowledge=POLICY, client=model, trace=True, output_schema=EXAMPLE)
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "passed"
    assert answer.turns == 2
    assert answer.data == {"refund_days": 14, "eligible": True}
    first = answer.checks[0]
    assert first.follows_instructions is True
    assert first.grounded is True
    assert first.revise is True
    assert first.structure is not None
    assert first.structure.followed is False
    assert first.structure.problems == (
        "$.refund_days: expected integer, got string",
        "$.note: this key is not in the schema",
    )
    assert "must be one JSON document that follows the OUTPUT SCHEMA" in first.notes
    assert "- $.refund_days: expected integer, got string" in first.notes
    revision = str(model.calls[len(checks()) + 1]["user"])
    assert "$.note: this key is not in the schema" in revision
    assert answer.checks[1].structure is not None
    assert answer.checks[1].structure.followed is True
    err = capsys.readouterr().err
    assert "turn 1: output does not follow the schema: 2 problems" in err
    assert "1 problems" not in err
    assert "turn 2: output follows the schema" in err


def test_prose_instead_of_json_is_a_structure_miss_with_no_data() -> None:
    model = ScriptedModel([Draft(answer="Fourteen days. [p1]", cited_ids=["p1"]), *checks()] * 2)
    loop = Loop(knowledge=POLICY, client=model, trace=False, output_schema=EXAMPLE, max_turns=2)
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "max_turns"
    assert answer.data is None
    assert answer.text == "Fourteen days. [p1]"
    assert answer.checks[-1].structure is not None
    assert answer.checks[-1].structure.problems[0].startswith("The answer is not valid JSON")


def test_a_fenced_document_is_accepted() -> None:
    model = ScriptedModel(accepted("```json\n" + GOOD + "\n```", "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False, output_schema=EXAMPLE)
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"
    assert answer.data == {"refund_days": 14, "eligible": True}


def test_a_json_schema_and_a_list_example_are_accepted_as_given() -> None:
    schema = {
        "type": "object",
        "properties": {"days": {"type": "integer", "minimum": 1}},
        "required": ["days"],
    }
    model = ScriptedModel(accepted('{"days": 14, "other": "fine"}', "p1"))
    loop = Loop(
        knowledge=POLICY, client=model, trace=False, output_format="json", output_schema=schema
    )
    assert loop.output_schema == schema
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"

    model = ScriptedModel(accepted('[{"day": 1}, {"day": 2}]', "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False, output_schema=[{"day": 1}])
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"
    assert answer.data == [{"day": 1}, {"day": 2}]


def test_length_and_structure_are_checked_together() -> None:
    long_bad = json.dumps({"refund_days": "x" * 400, "eligible": True})
    model = ScriptedModel(
        [
            Draft(answer=long_bad, cited_ids=["p1"]),
            *checks(),
            Draft(answer=GOOD, cited_ids=["p1"]),
            *checks(),
        ]
    )
    loop = Loop(
        knowledge=POLICY, client=model, trace=False, output_schema=EXAMPLE, response_length=300
    )
    answer = loop.ask("What is the refund window?")
    first = answer.checks[0]
    assert first.length is not None and not first.length.followed
    assert first.structure is not None and not first.structure.followed
    assert "keep it under 300 characters" in first.notes
    assert "$.refund_days: expected integer, got string" in first.notes
    assert answer.stopped_reason == "passed"
    assert answer.response_length == 300


def test_project_file_table_or_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "knowledge.md").write_text(POLICY, encoding="utf-8")
    toml = tmp_path / "loopkit.toml"
    toml.write_text(
        'knowledge = "knowledge.md"\noutput_format = "json"\n\n[output_schema]\nrefund_days = 14\neligible = true\n',
        encoding="utf-8",
    )
    project = load_project(toml)
    assert project.output_format == "json"
    assert project.output_schema == {"refund_days": 14, "eligible": True}
    monkeypatch.chdir(tmp_path)
    loop = Loop.load(client=ScriptedModel([]))
    assert loop.output_format == "json"
    assert loop.output_schema == SCHEMA

    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    toml.write_text('knowledge = "knowledge.md"\noutput_schema = "schema.json"\n', encoding="utf-8")
    assert load_project(toml).output_schema == SCHEMA
    loop = Loop.load(client=ScriptedModel([]))
    assert loop.output_format == "json"
    assert loop.output_schema == SCHEMA
    loop = Loop.load(client=ScriptedModel([]), output_format="text", output_schema=None)
    assert loop.output_format == "text"

    (tmp_path / "list.json").write_text('[{"day": 1}]', encoding="utf-8")
    toml.write_text('knowledge = "knowledge.md"\noutput_schema = "list.json"\n', encoding="utf-8")
    assert Loop.load(client=ScriptedModel([])).output_schema == infer_schema([{"day": 1}])

    toml.write_text(
        'knowledge = "knowledge.md"\noutput_schema = "missing.json"\n', encoding="utf-8"
    )
    with pytest.raises(LoopError, match="is not a file"):
        load_project(toml)
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    toml.write_text('knowledge = "knowledge.md"\noutput_schema = "bad.json"\n', encoding="utf-8")
    with pytest.raises(LoopError, match="not valid JSON"):
        load_project(toml)
    toml.write_text('knowledge = "knowledge.md"\noutput_schema = 3\n', encoding="utf-8")
    with pytest.raises(LoopError, match="must be a table"):
        load_project(toml)
    toml.write_text('knowledge = "knowledge.md"\noutput_format = "xml"\n', encoding="utf-8")
    with pytest.raises(
        LoopError, match=r"output_format in loopkit\.toml must be one of: text, json"
    ):
        load_project(toml)


def test_cli_output_schema_inline_or_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = ScriptedModel(accepted(GOOD, "p1"))
    code = main(
        ["ask", "Refunds?", "-k", POLICY, "--output-schema", json.dumps(EXAMPLE), "--json"],
        client=model,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_format"] == "json"
    assert payload["data"] == {"refund_days": 14, "eligible": True}
    assert payload["text"] == GOOD
    assert "OUTPUT SCHEMA" in str(model.calls[0]["system"])

    path = tmp_path / "schema.json"
    path.write_text(json.dumps(SCHEMA), encoding="utf-8")
    model = ScriptedModel(accepted(GOOD, "p1"))
    code = main(
        [
            "ask",
            "Refunds?",
            "-k",
            POLICY,
            "--output-format",
            "json",
            "--output-schema",
            str(path),
            "--quiet",
        ],
        client=model,
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == GOOD

    code = main(
        ["ask", "Refunds?", "-k", POLICY, "--output-format", "json"], client=ScriptedModel([])
    )
    assert code == 2
    assert "needs output_schema" in capsys.readouterr().err
    code = main(
        ["ask", "Refunds?", "-k", POLICY, "--output-schema", str(tmp_path / "none.json")],
        client=ScriptedModel([]),
    )
    assert code == 2
    assert "not valid JSON" in capsys.readouterr().err
