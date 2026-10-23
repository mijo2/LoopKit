"""The JSON-Schema-subset validator, schema inference, and answer parsing."""

from __future__ import annotations

import pytest

from loopkit.errors import LoopError
from loopkit.structure import (
    check_structure,
    coerce_schema,
    infer_schema,
    parse_answer,
    render_schema,
    validate,
)

EXAMPLE = {"refund_days": 14, "eligible": True, "steps": ["cancel"], "rate": 0.5, "note": None}


def test_infer_schema_from_an_example_is_closed() -> None:
    schema = infer_schema(EXAMPLE)
    assert schema == {
        "type": "object",
        "properties": {
            "refund_days": {"type": "integer"},
            "eligible": {"type": "boolean"},
            "steps": {"type": "array", "items": {"type": "string"}},
            "rate": {"type": "number"},
            "note": {},
        },
        "required": ["refund_days", "eligible", "steps", "rate", "note"],
        "additionalProperties": False,
    }
    assert infer_schema([]) == {"type": "array", "items": {}}
    assert infer_schema([{"a": "b"}])["items"] == infer_schema({"a": "b"})
    with pytest.raises(LoopError, match="not JSON"):
        infer_schema({"when": object()})


def test_coerce_schema_tells_a_schema_from_an_example_and_reads_json_text() -> None:
    schema = {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]}
    assert coerce_schema(schema) == schema
    assert coerce_schema(schema) is not schema
    assert coerce_schema(EXAMPLE) == infer_schema(EXAMPLE)
    assert coerce_schema('{"type": "array"}') == {"type": "array"}
    assert coerce_schema('{"k": "v"}') == infer_schema({"k": "v"})
    assert coerce_schema('["a"]') == infer_schema(["a"])
    with pytest.raises(LoopError, match="not valid JSON"):
        coerce_schema("{not json")
    with pytest.raises(LoopError, match="empty"):
        coerce_schema("   ")
    with pytest.raises(LoopError, match="must be a JSON Schema object"):
        coerce_schema(42)
    assert render_schema({"type": "object"}) == '{\n  "type": "object"\n}'


def test_validate_reports_every_problem_with_its_path() -> None:
    schema = infer_schema(EXAMPLE)
    problems = validate(
        {"refund_days": "14", "eligible": 1, "steps": [1, "ok"], "rate": 1, "extra": True}, schema
    )
    assert problems == [
        '$: missing required key "note"',
        "$.refund_days: expected integer, got string",
        "$.eligible: expected boolean, got integer",
        "$.steps[0]: expected string, got integer",
        "$.extra: this key is not in the schema",
    ]
    assert validate(EXAMPLE, schema) == []
    assert validate(dict(EXAMPLE, note="anything"), schema) == []
    assert validate([], schema) == ["$: expected object, got array"]
    # A boolean is not an integer, and an integer is a number.
    assert validate(True, {"type": "integer"}) == ["$: expected integer, got boolean"]
    assert validate(True, {"type": "number"}) == ["$: expected number, got boolean"]
    assert validate(3, {"type": "number"}) == []
    assert validate(None, {"type": ["string", "null"]}) == []
    assert validate(3, {"type": ["string", "null"]}) == ["$: expected string, null, got integer"]


def test_validate_enum_const_bounds_anyof_and_refs() -> None:
    schema = {
        "type": "object",
        "properties": {
            "kind": {"enum": ["refund", "credit"]},
            "version": {"const": 2},
            "amount": {"type": "number", "minimum": 0, "maximum": 100},
            "name": {"type": "string", "minLength": 1, "maxLength": 3},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "inner": {"$ref": "#/$defs/inner"},
            "lost": {"$ref": "#/$defs/missing"},
        },
        "additionalProperties": {"type": "boolean"},
        "$defs": {"inner": {"type": "object", "properties": {"b": {"type": "string"}}}},
    }
    good = {
        "kind": "refund",
        "version": 2,
        "amount": 50,
        "name": "ab",
        "tags": ["x"],
        "either": 1,
        "inner": {"b": "c"},
        "flag": True,
    }
    assert validate(good, schema) == []
    bad = {
        "kind": "void",
        "version": 3,
        "amount": 101,
        "name": "",
        "tags": [],
        "either": 1.5,
        "inner": {"b": 1},
        "lost": 1,
        "flag": "yes",
    }
    assert validate(bad, schema) == [
        '$.kind: must be one of ["refund", "credit"]',
        "$.version: must be 2",
        "$.amount: must be at most 100",
        "$.name: must be at least 1 characters",
        "$.tags: must have at least 1 items",
        "$.either: matches none of the 2 allowed shapes",
        "$.inner.b: expected string, got integer",
        "$.lost: the schema refers to #/$defs/missing, which is not in the schema",
        "$.flag: expected boolean, got string",
    ]
    assert validate({"amount": -1, "name": "abcd", "tags": ["a", "b", "c"]}, schema) == [
        "$.amount: must be at least 0",
        "$.name: must be at most 3 characters",
        "$.tags: must have at most 2 items",
    ]
    # Keywords the validator does not know are ignored, not errors.
    assert validate("x", {"type": "string", "format": "email", "pattern": "^a"}) == []


def test_parse_answer_forgives_a_fence_and_nothing_else() -> None:
    assert parse_answer('{"a": 1}') == ({"a": 1}, None)
    assert parse_answer('```json\n{"a": 1}\n```') == ({"a": 1}, None)
    assert parse_answer("```\n[1, 2]\n```") == ([1, 2], None)
    value, problem = parse_answer('Here you go: {"a": 1}')
    assert value is None
    assert problem is not None and problem.startswith("The answer is not valid JSON")
    assert parse_answer("  ") == (None, "The answer is empty; it must be one JSON document.")


def test_check_structure_returns_the_value_and_the_problems() -> None:
    schema = infer_schema({"days": 1})
    assert check_structure('{"days": 14}', schema) == ({"days": 14}, [])
    value, problems = check_structure('{"days": "14"}', schema)
    assert value == {"days": "14"}
    assert problems == ["$.days: expected integer, got string"]
    value, problems = check_structure("fourteen", schema)
    assert value is None
    assert len(problems) == 1
