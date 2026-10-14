from pathlib import Path

import pytest

from loopkit.errors import LoopError
from loopkit.knowledge import CHUNK_CHARS, Knowledge, Passage


def test_directory_reads_text_skips_other_types_and_keeps_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# Later\n\nSecond file.\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# Earlier\n\nFirst file.\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    (tmp_path / ".secret.md").write_text("hidden", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "note.txt").write_text("Nested note.", encoding="utf-8")

    knowledge = Knowledge.load(tmp_path)

    sources = [passage.source for passage in knowledge.passages]
    assert sources == [
        f"{tmp_path.name}/a.md",
        f"{tmp_path.name}/b.md",
        f"{tmp_path.name}/nested/note.txt",
    ]
    assert knowledge.passages[0].heading == "Earlier"
    assert any("unsupported file type" in warning for warning in knowledge.warnings)
    assert all("hidden" not in passage.text for passage in knowledge.passages)


def test_missing_single_token_is_an_error() -> None:
    with pytest.raises(LoopError, match="No knowledge at docs"):
        Knowledge.load("docs")


def test_sentence_is_inline_even_when_it_mentions_a_path() -> None:
    knowledge = Knowledge.load("See docs/auth.md before you rotate a key.")
    assert knowledge.passages[0].source == "text"
    assert "rotate a key" in knowledge.passages[0].text


def test_existing_file_wins_over_treating_the_string_as_text(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("Refunds last 14 days.\n", encoding="utf-8")
    knowledge = Knowledge.load(str(path))
    assert "14 days" in knowledge.passages[0].text
    assert knowledge.passages[0].source == "policy.md"


def test_dict_and_list_of_sentences() -> None:
    knowledge = Knowledge.load(
        [
            {"policy": "Keys are shown once."},
            "Refunds last 14 days after purchase.",
            "Reset links expire after 30 minutes flat.",
        ]
    )
    blob = "\n".join(passage.text for passage in knowledge.passages)
    assert "shown once" in blob
    assert "14 days" in blob
    assert "30 minutes" in blob
    assert knowledge.passages[0].source == "policy"


def test_code_fence_is_not_split_into_a_heading() -> None:
    text = "# Setup\n\n```python\n# not a heading\nprint('hi')\n```\n\nInstall the package.\n"
    knowledge = Knowledge.load({"guide": text})
    assert len(knowledge.passages) == 1
    assert knowledge.passages[0].heading == "Setup"
    assert "# not a heading" in knowledge.passages[0].text
    assert "Install the package." in knowledge.passages[0].text


def test_long_document_keeps_the_start_and_the_end() -> None:
    text = "STARTWORD\n\n" + ("middle word " * 400) + "\n\nENDWORD"
    assert len(text) > CHUNK_CHARS
    knowledge = Knowledge.load({"guide": text})
    blob = "\n".join(passage.text for passage in knowledge.passages)
    assert "STARTWORD" in blob
    assert "ENDWORD" in blob
    assert len(knowledge.passages) > 1


def test_empty_directory_and_unreadable_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(LoopError, match="No files"):
        Knowledge.load(empty)

    images = tmp_path / "images"
    images.mkdir()
    (images / "logo.png").write_bytes(b"\x89PNG")
    with pytest.raises(LoopError, match="No readable text"):
        Knowledge.load(images)


def test_oversized_file_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loopkit.knowledge.MAX_FILE_BYTES", 8)
    (tmp_path / "big.md").write_text("this is too long", encoding="utf-8")
    (tmp_path / "ok.md").write_text("short", encoding="utf-8")
    knowledge = Knowledge.load(tmp_path)
    assert "short" in knowledge.passages[0].text
    assert any("larger than" in warning for warning in knowledge.warnings)


def test_duplicate_passage_ids_raise() -> None:
    passage = Passage(id="p1", source="a", heading="", text="hello there")
    with pytest.raises(LoopError, match="unique"):
        Knowledge(passages=(passage, passage))


def test_empty_text_raises() -> None:
    with pytest.raises(LoopError, match="empty"):
        Knowledge.load("   ")
