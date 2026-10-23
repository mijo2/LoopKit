"""Structured output: the shape the caller wants the answer in, and whether a draft has it.

``output_format="json"`` asks the drafter for one JSON document in the
answer and holds it to a schema. The caller gives the schema as a JSON
Schema object, or as an example of the output, from which
:func:`infer_schema` writes one. After every draft :func:`check_structure`
parses the answer and validates it in code, so a miss is exact and costs
no tokens; the loop turns the problems into a revision note.

:func:`validate` covers the part of JSON Schema that describes a shape:
``type``, ``enum``, ``const``, ``properties``, ``required``,
``additionalProperties``, ``items``, ``minItems``/``maxItems``,
``minLength``/``maxLength``, ``minimum``/``maximum``, ``anyOf``, and
``$ref`` to ``$defs``/``definitions`` in the same document. Keywords it
does not know are ignored, so a richer schema still validates its shape.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from loopkit.errors import LoopError

OutputFormat = Literal["text", "json"]
OUTPUT_FORMATS = ("text", "json")

_TYPES: Mapping[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}

_SCHEMA_MARKERS = {
    "type",
    "properties",
    "items",
    "$schema",
    "anyOf",
    "oneOf",
    "allOf",
    "enum",
    "$ref",
}


def coerce_schema(value: object) -> dict[str, object]:
    """A JSON Schema from what the caller passed.

    A mapping that uses schema keywords (``type``, ``properties``, ...) is
    taken as a schema. Any other mapping, or a list, is an example of the
    output and a schema is inferred from it. A string is parsed as JSON
    first and then treated the same way.
    """

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise LoopError("output_schema is empty.")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LoopError(
                f"output_schema is not valid JSON: {exc.msg} at position {exc.pos}."
            ) from None
    if isinstance(value, Mapping):
        if set(value) & _SCHEMA_MARKERS:
            return _to_plain(value)
        return infer_schema(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return infer_schema(value)
    raise LoopError("output_schema must be a JSON Schema object, an example object, or JSON text.")


def infer_schema(example: object) -> dict[str, object]:
    """A closed JSON Schema with the shape of ``example``.

    Objects require every key they show and allow no others; arrays take
    the shape of their first item; ``None`` places no constraint on a value.
    """

    if isinstance(example, bool):
        return {"type": "boolean"}
    if isinstance(example, int):
        return {"type": "integer"}
    if isinstance(example, float):
        return {"type": "number"}
    if isinstance(example, str):
        return {"type": "string"}
    if example is None:
        return {}
    if isinstance(example, Mapping):
        properties = {str(key): infer_schema(value) for key, value in example.items()}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if isinstance(example, Sequence) and not isinstance(example, (bytes, bytearray)):
        items = infer_schema(example[0]) if len(example) else {}
        return {"type": "array", "items": items}
    raise LoopError(
        f"output_schema example holds a value that is not JSON: {type(example).__name__}."
    )


def render_schema(schema: Mapping[str, object]) -> str:
    """The schema as the drafter is shown it."""

    return json.dumps(schema, indent=2, ensure_ascii=False)


def parse_answer(text: str) -> tuple[object, str | None]:
    """The JSON document in a draft's answer, or why there is none.

    A code fence around the document is forgiven; anything else outside it
    is not, because the answer is the document.
    """

    stripped = _strip_fence(text.strip())
    if not stripped:
        return None, "The answer is empty; it must be one JSON document."
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError as exc:
        return None, f"The answer is not valid JSON: {exc.msg} at position {exc.pos}."


def check_structure(text: str, schema: Mapping[str, object]) -> tuple[object, list[str]]:
    """Parse the answer and validate it. Returns the parsed value and the problems."""

    value, problem = parse_answer(text)
    if problem is not None:
        return None, [problem]
    return value, validate(value, schema)


def validate(instance: object, schema: Mapping[str, object], path: str = "$") -> list[str]:
    """Problems with ``instance`` under ``schema``, as ``$.path: problem`` lines; none when it fits."""

    return _validate(instance, schema, path, schema)


def _validate(
    instance: object, schema: Mapping[str, object], path: str, root: Mapping[str, object]
) -> list[str]:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _resolve(ref, root)
        if target is None:
            return [f"{path}: the schema refers to {ref}, which is not in the schema"]
        return _validate(instance, target, path, root)
    problems: list[str] = []
    expected = schema.get("type")
    if expected is not None and not _has_type(instance, expected):
        names = ", ".join(expected) if isinstance(expected, list) else str(expected)
        return [f"{path}: expected {names}, got {_describe(instance)}"]
    if "enum" in schema:
        allowed = schema["enum"]
        if isinstance(allowed, list) and not any(_same(instance, item) for item in allowed):
            problems.append(f"{path}: must be one of {json.dumps(allowed, ensure_ascii=False)}")
    if "const" in schema and not _same(instance, schema["const"]):
        problems.append(f"{path}: must be {json.dumps(schema['const'], ensure_ascii=False)}")
    options = schema.get("anyOf")
    if isinstance(options, list) and options:
        fits = any(
            isinstance(option, Mapping) and not _validate(instance, option, path, root)
            for option in options
        )
        if not fits:
            problems.append(f"{path}: matches none of the {len(options)} allowed shapes")
    if isinstance(instance, Mapping):
        problems.extend(_validate_object(instance, schema, path, root))
    elif isinstance(instance, list):
        problems.extend(_validate_array(instance, schema, path, root))
    elif isinstance(instance, str):
        low, high = schema.get("minLength"), schema.get("maxLength")
        if isinstance(low, int) and len(instance) < low:
            problems.append(f"{path}: must be at least {low} characters")
        if isinstance(high, int) and len(instance) > high:
            problems.append(f"{path}: must be at most {high} characters")
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, (int, float)) and instance < low:
            problems.append(f"{path}: must be at least {low}")
        if isinstance(high, (int, float)) and instance > high:
            problems.append(f"{path}: must be at most {high}")
    return problems


def _validate_object(
    instance: Mapping[object, object],
    schema: Mapping[str, object],
    path: str,
    root: Mapping[str, object],
) -> list[str]:
    problems: list[str] = []
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    required = schema.get("required")
    for name in required if isinstance(required, list) else []:
        if name not in instance:
            problems.append(f"{path}: missing required key {json.dumps(name)}")
    for name, value in instance.items():
        child = f"{path}.{name}"
        sub = properties.get(name)
        if isinstance(sub, Mapping):
            problems.extend(_validate(value, sub, child, root))
        elif schema.get("additionalProperties") is False:
            problems.append(f"{child}: this key is not in the schema")
        elif isinstance(schema.get("additionalProperties"), Mapping):
            extra = schema["additionalProperties"]
            if isinstance(extra, Mapping):
                problems.extend(_validate(value, extra, child, root))
    return problems


def _validate_array(
    instance: list[object], schema: Mapping[str, object], path: str, root: Mapping[str, object]
) -> list[str]:
    problems: list[str] = []
    low, high = schema.get("minItems"), schema.get("maxItems")
    if isinstance(low, int) and len(instance) < low:
        problems.append(f"{path}: must have at least {low} items")
    if isinstance(high, int) and len(instance) > high:
        problems.append(f"{path}: must have at most {high} items")
    items = schema.get("items")
    if isinstance(items, Mapping):
        for index, value in enumerate(instance):
            problems.extend(_validate(value, items, f"{path}[{index}]", root))
    return problems


def _resolve(ref: str, root: Mapping[str, object]) -> Mapping[str, object] | None:
    if not ref.startswith("#/"):
        return None
    node: object = root
    for part in ref[2:].split("/"):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, Mapping) else None


def _has_type(instance: object, expected: object) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        kinds = _TYPES.get(str(name))
        if kinds is None:
            continue
        if name in {"integer", "number"} and isinstance(instance, bool):
            continue
        if isinstance(instance, kinds):
            return True
    return False


def _same(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return left == right


def _describe(instance: object) -> str:
    if instance is None:
        return "null"
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, Mapping):
        return "object"
    return type(instance).__name__


def _to_plain(value: Mapping[str, object]) -> dict[str, object]:
    """A plain dict copy, so a TOML table or a user's mapping is not aliased."""

    plain = json.loads(json.dumps(value))
    if not isinstance(plain, dict):
        raise LoopError("output_schema must be a JSON object.")
    return plain


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: body.rstrip().rfind("```")]
    return body.strip()
