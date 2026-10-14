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
        'knowledge = "knowledge"\ninstructions = "Be brief."\nmodel = "grok-4.7"\nmax_turns = 2\n',
    )
    monkeypatch.chdir(tmp_path / "..")
    model = ScriptedModel(accepted("Bluebird. [p1]", "p1"))
    loop = Loop.load(tmp_path / "loopkit.toml", client=model)
    answer = loop.ask("What is the launch code?")
    assert answer.text == "Bluebird. [p1]"
    assert loop.max_turns == 2
    assert "Be brief." in str(model.calls[0]["user"])
    assert "bluebird" in str(model.calls[0]["user"])


def test_unknown_setting_and_bad_numbers(tmp_path: Path) -> None:
    _project(tmp_path, 'knowledge = "knowledge"\napi_key = "secret"\n')
    with pytest.raises(LoopError, match="api_key"):
        Loop.load(tmp_path / "loopkit.toml", client=ScriptedModel([]))

    _project(tmp_path, 'knowledge = "knowledge"\nmax_turns = true\n')
    with pytest.raises(LoopError, match="max_turns"):
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
