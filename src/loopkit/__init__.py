"""Answer from a knowledge base. The loop stays in this package."""

from importlib.metadata import PackageNotFoundError, version

from loopkit.client import DEFAULT_MODELS
from loopkit.errors import LoopError, LoopWarning
from loopkit.knowledge import Knowledge, Passage
from loopkit.loop import (
    BOOK_INSTRUCTIONS,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MODEL,
    OPEN_BOOK_INSTRUCTIONS,
    OPEN_INSTRUCTIONS,
    SEARCH_INSTRUCTIONS,
    Loop,
    ask,
    default_instructions,
)
from loopkit.models import Answer, Check, Citation, InstructionCheck, ResponseKind

try:
    __version__ = version("loop-kit")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "BOOK_INSTRUCTIONS",
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_MODEL",
    "DEFAULT_MODELS",
    "OPEN_BOOK_INSTRUCTIONS",
    "OPEN_INSTRUCTIONS",
    "SEARCH_INSTRUCTIONS",
    "Answer",
    "Check",
    "Citation",
    "InstructionCheck",
    "Knowledge",
    "Loop",
    "LoopError",
    "LoopWarning",
    "Passage",
    "ResponseKind",
    "__version__",
    "ask",
    "default_instructions",
]
