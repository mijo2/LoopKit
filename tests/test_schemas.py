import json

from loopkit.loop import DEFAULT_MODEL
from loopkit.schemas import Draft, DraftCheck, GroundingCheck, InstructionVerdict, OneCheck


def test_structured_output_schemas_stay_simple() -> None:
    assert DEFAULT_MODEL == "grok-4.7"
    for schema in (Draft, DraftCheck, InstructionVerdict, OneCheck, GroundingCheck):
        body = schema.model_json_schema()
        encoded = json.dumps(body)
        assert body["type"] == "object"
        assert "minContains" not in encoded
        assert "maxContains" not in encoded
        for prop in body["properties"].values():
            assert not isinstance(prop.get("items"), list)
