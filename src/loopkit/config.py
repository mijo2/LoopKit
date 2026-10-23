"""Optional ``loopkit.toml``: the settings for one knowledge base, kept out of code.

The file lives next to the knowledge folder and is read when ``Loop`` is
built without ``knowledge=`` (``Loop.load()``, the top-level ``ask()``, and
``loopkit ask`` without ``--knowledge``). Paths in it are relative to the
file. It never holds an API key. ``loopkit init`` writes a starter copy.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from loopkit.errors import LoopError

ALLOWED_KEYS = {
    "knowledge",
    "instructions",
    "model",
    "provider",
    "base_url",
    "max_turns",
    "top_k",
    "check_mode",
    "response",
    "search",
    "auto_optimize",
    "token_budget",
    "token_budget_mode",
    "response_length",
    "output_format",
    "output_schema",
    "check_model",
    "check_provider",
    "check_base_url",
}
CHECK_MODES = ("each", "all")
RESPONSES = ("normal", "book")
TOKEN_BUDGET_MODES = ("soft", "hard")
OUTPUT_FORMATS = ("text", "json")
FILENAME = "loopkit.toml"


@dataclass(frozen=True)
class Project:
    """Settings read from one ``loopkit.toml``.

    ``knowledge`` is None when the file did not name one, or when there was
    no file and ``missing_ok`` was set. ``note`` then says why, so the loop
    can warn the user in those words.
    """

    knowledge: Path | tuple[Path, ...] | None
    instructions: tuple[str, ...] | None
    model: str | None
    provider: str | None
    base_url: str | None
    max_turns: int | None
    top_k: int | None
    check_mode: str | None
    response: str | None
    directory: Path
    note: str | None = None
    search: bool | None = None
    auto_optimize: bool | None = None
    token_budget: int | None = None
    token_budget_mode: str | None = None
    response_length: int | str | None = None
    output_format: str | None = None
    output_schema: dict[str, object] | list[object] | None = None
    check_model: str | None = None
    check_provider: str | None = None
    check_base_url: str | None = None


def load_project(path: str | Path | None = None, *, missing_ok: bool = False) -> Project:
    """Read ``loopkit.toml``.

    With no ``path`` the file is looked for in the working directory. When it
    is not there and ``missing_ok`` is set, an empty project is returned with
    a ``note`` instead of an error, so a loop can still run with no passages.
    An explicit ``path`` that does not exist is always an error.
    """

    file = Path(path) if path is not None else Path.cwd() / FILENAME
    if not file.is_file():
        if path is None and missing_ok:
            return _empty_project(
                file.parent,
                f"No {FILENAME} in {file.parent} and no knowledge was given. "
                "The loop runs in open mode: the model answers from its own knowledge and the "
                "audit checks accuracy instead of grounding. To answer from documents, "
                "pass --knowledge, or run `loopkit init`.",
            )
        raise LoopError(
            f"No {file.name} at {file}. Pass --knowledge, or run `loopkit init`."
            if path is None
            else f"No project file at {file}."
        )
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise LoopError(f"Could not read {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise LoopError(f"{file} must be a TOML table.")

    unknown = sorted(set(data) - ALLOWED_KEYS)
    if unknown:
        names = ", ".join(unknown)
        allowed = ", ".join(sorted(ALLOWED_KEYS))
        raise LoopError(
            f"Unknown {file.name} setting: {names}. Allowed settings: {allowed}. "
            "Set an API key in the environment, not in this file."
        )
    base = file.parent
    note = None
    if "knowledge" not in data:
        note = (
            f"{file} does not set knowledge. The loop runs in open mode: the model answers "
            "from its own knowledge and the audit checks accuracy instead of grounding. "
            'To answer from documents, add knowledge = "knowledge" to the file, or pass --knowledge.'
        )
    return Project(
        knowledge=None if note else _knowledge(data["knowledge"], base, file.name),
        instructions=_instructions(data.get("instructions"), file.name),
        model=_optional_str(data.get("model"), "model", file.name),
        provider=_optional_str(data.get("provider"), "provider", file.name),
        base_url=_optional_str(data.get("base_url"), "base_url", file.name),
        max_turns=_optional_int(data.get("max_turns"), "max_turns", file.name),
        top_k=_optional_int(data.get("top_k"), "top_k", file.name),
        check_mode=_optional_choice(data.get("check_mode"), "check_mode", file.name, CHECK_MODES),
        response=_optional_choice(data.get("response"), "response", file.name, RESPONSES),
        directory=base,
        note=note,
        search=_optional_bool(data.get("search"), "search", file.name),
        auto_optimize=_optional_bool(data.get("auto_optimize"), "auto_optimize", file.name),
        token_budget=_optional_int(data.get("token_budget"), "token_budget", file.name),
        token_budget_mode=_optional_choice(
            data.get("token_budget_mode"), "token_budget_mode", file.name, TOKEN_BUDGET_MODES
        ),
        response_length=_response_length(data.get("response_length"), file.name),
        output_format=_optional_choice(
            data.get("output_format"), "output_format", file.name, OUTPUT_FORMATS
        ),
        output_schema=_output_schema(data.get("output_schema"), base, file.name),
        check_model=_optional_str(data.get("check_model"), "check_model", file.name),
        check_provider=_optional_str(data.get("check_provider"), "check_provider", file.name),
        check_base_url=_optional_str(data.get("check_base_url"), "check_base_url", file.name),
    )


def _empty_project(directory: Path, note: str) -> Project:
    return Project(
        knowledge=None,
        instructions=None,
        model=None,
        provider=None,
        base_url=None,
        max_turns=None,
        top_k=None,
        check_mode=None,
        response=None,
        directory=directory,
        note=note,
    )


def _knowledge(value: object, base: Path, filename: str) -> Path | tuple[Path, ...]:
    if isinstance(value, str):
        return _resolve(value, base)
    if isinstance(value, list):
        if not value or not all(isinstance(item, str) for item in value):
            raise LoopError(f"knowledge in {filename} must be a path or a list of paths.")
        return tuple(_resolve(item, base) for item in value)
    raise LoopError(f"knowledge in {filename} must be a path or a list of paths.")


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def _instructions(value: object, filename: str) -> tuple[str, ...] | None:
    """A project lists instructions so each one can be checked on its own."""

    if value is None:
        return None
    if not isinstance(value, list):
        raise LoopError(
            f"instructions in {filename} must be a list of strings, one instruction per item."
        )
    if not value:
        raise LoopError(f"instructions in {filename} must list at least one instruction.")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LoopError(f"instructions in {filename} must be a list of non-empty strings.")
        # A list item is already one instruction. Line breaks inside it stay put.
        items.append(item.strip())
    return tuple(items)


def _optional_str(value: object, name: str, filename: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise LoopError(f"{name} in {filename} must be a non-empty string.")
    return value


def _optional_choice(
    value: object,
    name: str,
    filename: str,
    allowed: tuple[str, ...],
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        names = ", ".join(allowed)
        raise LoopError(f"{name} in {filename} must be one of: {names}.")
    return value.strip().lower()


def _optional_bool(value: object, name: str, filename: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise LoopError(f"{name} in {filename} must be true or false.")
    return value


def _response_length(value: object, filename: str) -> int | str | None:
    """``"auto"`` or a character count of at least 1."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LoopError(
            f'response_length in {filename} must be "auto" or an integer of at least 1 (characters).'
        )
    return value


def _output_schema(
    value: object, base: Path, filename: str
) -> dict[str, object] | list[object] | None:
    """A table in the file, used as is, or the path of a JSON file to read.

    Either may be a JSON Schema or an example of the output; the loop tells
    them apart.
    """

    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        path = _resolve(value.strip(), base)
        if not path.is_file():
            raise LoopError(f"output_schema in {filename} names {path}, which is not a file.")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LoopError(f"output_schema file {path} is not valid JSON: {exc.msg}.") from None
        except OSError as exc:
            raise LoopError(f"Could not read output_schema file {path}: {exc}") from None
        if not isinstance(loaded, (dict, list)):
            raise LoopError(f"output_schema file {path} must hold a JSON object or array.")
        return loaded
    raise LoopError(
        f"output_schema in {filename} must be a table (the schema or an example) or the path of a JSON file."
    )


def _optional_int(value: object, name: str, filename: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LoopError(f"{name} in {filename} must be an integer of at least 1.")
    return value
