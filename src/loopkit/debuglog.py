"""Debug log: every prompt and every raw model reply, in sections.

Everything goes through the standard ``logging`` logger named ``loopkit``,
so a program can attach its own handler. :func:`enable_debug_log` is the
short way: it appends to a file, one block per event, with a ruled header
and a named section for each part of the event.

Events logged at DEBUG level:

* ``ASK``       the task, the settings, and the passages that were retrieved
* ``CALL``      one model call: the stage, the system prompt, the user prompt
* ``RESULT``    the raw text the model returned, the parsed JSON, timing
* ``ERROR``     a call that failed, with the raw reply when there was one
* ``TURN``      what the loop decided after a check: verdicts and the next step
* ``ANSWER``    what :meth:`Loop.ask` returned

Retries are logged at WARNING level from :mod:`loopkit.retry`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("loopkit")

DEFAULT_DEBUG_FILE = "loopkit-debug.log"
RULE = "=" * 88
SUBRULE = "-" * 88

_handlers: dict[Path, logging.Handler] = {}


class _SectionFormatter(logging.Formatter):
    """Timestamp plus level on the first line; the message is already laid out."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        message = record.getMessage()
        if getattr(record, "loopkit_block", False):
            return message
        return f"[{stamp}] {record.levelname} {message}"


def debug_path_from_env() -> Path | None:
    """``LOOPKIT_DEBUG`` is a file path, or a truthy flag for the default file."""

    raw = os.environ.get("LOOPKIT_DEBUG", "").strip()
    if not raw or raw.lower() in {"0", "false", "no", "off"}:
        return None
    if raw.lower() in {"1", "true", "yes", "on"}:
        return Path(DEFAULT_DEBUG_FILE)
    return Path(raw).expanduser()


def resolve_debug(value: object) -> Path | None:
    """Turn a ``debug=`` argument into a path, or None for off.

    ``None`` reads ``LOOPKIT_DEBUG``. ``True`` uses the default file name.
    A string or path is the file to write.
    """

    if value is None:
        return debug_path_from_env()
    if value is False:
        return None
    if value is True:
        return Path(DEFAULT_DEBUG_FILE)
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        return Path(text).expanduser() if text else None
    raise TypeError("debug must be a path, True, False, or None.")


def enable_debug_log(path: str | Path) -> Path:
    """Append debug blocks to ``path``. Returns the resolved path.

    The same file is opened once per process, so two loops writing to one
    file share a handler and do not duplicate lines.
    """

    target = Path(path).expanduser().resolve()
    if target not in _handlers:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(_SectionFormatter())
        logger.addHandler(handler)
        _handlers[target] = handler
    if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)
    return target


def disable_debug_log(path: str | Path) -> None:
    """Close the handler for ``path`` when one was opened."""

    target = Path(path).expanduser().resolve()
    handler = _handlers.pop(target, None)
    if handler is not None:
        logger.removeHandler(handler)
        handler.close()
    if not _handlers and logger.level == logging.DEBUG:
        logger.setLevel(logging.NOTSET)


def block(title: str, *sections: tuple[str, object], meta: dict[str, object] | None = None) -> None:
    """Log one ruled block: a title line, optional key=value line, then sections.

    A section value that is a string is written as is. Anything else is
    written as JSON so lists and parsed results stay readable.
    """

    if not logger.isEnabledFor(logging.DEBUG):
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [RULE, f"[{stamp}] {title}"]
    if meta:
        lines.append("  ".join(f"{key}={_meta_value(value)}" for key, value in meta.items()))
    for name, value in sections:
        lines.append(SUBRULE)
        lines.append(f"--- {name} ---")
        lines.append(render(value))
    lines.append("")
    logger.debug("\n".join(lines), extra={"loopkit_block": True})


def render(value: object) -> str:
    if isinstance(value, str):
        return value if value.strip() else "(empty)"
    if value is None:
        return "(none)"
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=_fallback)
    except (TypeError, ValueError):
        return repr(value)


def _fallback(value: Any) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    return repr(value)


def _meta_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
