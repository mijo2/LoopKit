import json
from pathlib import Path

from loopkit.knowledge import Knowledge
from loopkit.schemas import Draft, DraftCheck

ROOT = Path(__file__).parents[1] / "examples" / "northwind"


def test_northwind_knowledge_and_schemas() -> None:
    knowledge = Knowledge.load(ROOT / "knowledge")
    refunds = [passage for passage in knowledge.passages if "14 days" in passage.text]
    resets = [passage for passage in knowledge.passages if "30 minutes" in passage.text]
    assert refunds[0].heading == "Refunds"
    assert refunds[0].source.endswith("billing.md")
    assert resets[0].heading == "Password reset"
    instructions = (ROOT / "instructions.txt").read_text(encoding="utf-8")
    assert "help@northwind.example" in instructions
    assert "Loop(" in (ROOT / "ask.py").read_text(encoding="utf-8")

    for schema in (Draft, DraftCheck):
        encoded = json.dumps(schema.model_json_schema())
        assert "minContains" not in encoded
        assert "maxContains" not in encoded
        assert '"items": [' not in encoded
