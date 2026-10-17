from pathlib import Path

import pytest

from loopkit.errors import LoopError
from loopkit.loop import Loop
from tests.fakes import ScriptedModel, accepted


def _project(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "loopkit.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_resolves_knowledge_next_to_the_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "a.md").write_text("The launch code is bluebird.\n", encoding="utf-8")
    _project(
        tmp_path,
        'knowledge = "knowledge"\n'
        'instructions = ["Be brief.", "Use the passages."]\n'
        'model = "grok-4.7"\n'
        "max_turns = 2\n",
    )
    monkeypatch.chdir(tmp_path / "..")
    model = ScriptedModel(accepted("Bluebird. [p1]", "p1", instructions=2))
    loop = Loop.load(tmp_path / "loopkit.toml", client=model)
    answer = loop.ask("What is the launch code?")
    assert answer.text == "Bluebird. [p1]"
    assert loop.max_turns == 2
    assert loop.instructions == ("Be brief.", "Use the passages.")
    prompt = str(model.calls[0]["user"])
    assert "1. Be brief." in prompt
    assert "2. Use the passages." in prompt
    assert "bluebird" in prompt


def test_unknown_setting_and_bad_numbers(tmp_path: Path) -> None:
    _project(tmp_path, 'knowledge = "knowledge"\napi_key = "secret"\n')
    with pytest.raises(LoopError, match="api_key"):
        Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))

    _project(tmp_path, 'knowledge = "knowledge"\nmax_turns = true\n')
    with pytest.raises(LoopError, match="max_turns"):
        Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))


def test_a_multiline_list_item_stays_one_instruction(tmp_path: Path) -> None:
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "a.md").write_text("The launch code is bluebird.\n", encoding="utf-8")
    _project(
        tmp_path,
        'knowledge = "knowledge"\ninstructions = ["""\nBe brief.\nCite a passage.\n"""]\n',
    )
    loop = Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))
    assert loop.instructions == ("Be brief.\nCite a passage.",)


def test_instructions_must_be_a_list(tmp_path: Path) -> None:
    _project(tmp_path, 'knowledge = "knowledge"\ninstructions = "Be brief."\n')
    with pytest.raises(LoopError, match="list of strings"):
        Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))

    _project(tmp_path, 'knowledge = "knowledge"\ninstructions = []\n')
    with pytest.raises(LoopError, match="at least one"):
        Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))


def test_check_mode_and_response_come_from_the_file(tmp_path: Path) -> None:
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "a.md").write_text("The launch code is bluebird.\n", encoding="utf-8")
    _project(
        tmp_path,
        'knowledge = "knowledge"\ncheck_mode = "all"\nresponse = "book"\n',
    )
    loop = Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))
    assert loop.check_mode == "all"
    assert loop.response == "book"

    _project(tmp_path, 'knowledge = "knowledge"\ncheck_mode = "batch"\n')
    with pytest.raises(LoopError, match="check_mode"):
        Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))


def test_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LoopError, match="loopkit init"):
        Loop.load()


def test_ask_reads_the_project_when_knowledge_is_omitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "notes.md").write_text("Desks ship in two weeks.\n", encoding="utf-8")
    _project(tmp_path, 'knowledge = "notes.md"\n')
    monkeypatch.chdir(tmp_path)
    from loopkit import ask

    model = ScriptedModel(accepted("Two weeks. [p1]", "p1"))
    answer = ask("When do desks ship?", client=model)
    assert answer.text == "Two weeks. [p1]"
    assert "two weeks" in str(model.calls[0]["user"])
