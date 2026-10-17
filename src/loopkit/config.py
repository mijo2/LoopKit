"""Optional ``loopkit.toml``. A project needs a knowledge path and nothing else."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from loopkit.errors import LoopError

ALLOWED_KEYS = {"knowledge", "instructions", "model", "provider", "base_url", "max_turns", "top_k"}
FILENAME = "loopkit.toml"


@dataclass(frozen=True)
class Project:
    knowledge: Path | tuple[Path, ...]
    instructions: tuple[str, ...] | None
    model: str | None
    provider: str | None
    base_url: str | None
    max_turns: int | None
    top_k: int | None
    directory: Path


def load_project(path: str | Path | None = None) -> Project:
    file = Path(path) if path is not None else Path.cwd() / FILENAME
    if not file.is_file():
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
    if "knowledge" not in data:
        raise LoopError(f"{file.name} needs a knowledge path.")

    base = file.parent
    return Project(
        knowledge=_knowledge(data["knowledge"], base, file.name),
        instructions=_instructions(data.get("instructions"), file.name),
        model=_optional_str(data.get("model"), "model", file.name),
        provider=_optional_str(data.get("provider"), "provider", file.name),
        base_url=_optional_str(data.get("base_url"), "base_url", file.name),
        max_turns=_optional_int(data.get("max_turns"), "max_turns", file.name),
        top_k=_optional_int(data.get("top_k"), "top_k", file.name),
        directory=base,
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
        if not isinstance(item, str):
            raise LoopError(f"instructions in {filename} must be a list of non-empty strings.")
        lines = [line.strip() for line in item.splitlines() if line.strip()]
        if not lines:
            raise LoopError(f"instructions in {filename} must be a list of non-empty strings.")
        items.extend(lines)
    return tuple(items)


def _optional_str(value: object, name: str, filename: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise LoopError(f"{name} in {filename} must be a non-empty string.")
    return value


def _optional_int(value: object, name: str, filename: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LoopError(f"{name} in {filename} must be an integer of at least 1.")
    return value
