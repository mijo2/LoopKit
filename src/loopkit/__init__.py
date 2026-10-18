"""Answer from a knowledge base. The loop stays in this package."""

from importlib.metadata import PackageNotFoundError, version

from loopkit.errors import LoopError
from loopkit.knowledge import Knowledge, Passage
from loopkit.client import DEFAULT_MODELS
from loopkit.loop import BOOK_INSTRUCTIONS, DEFAULT_INSTRUCTIONS, DEFAULT_MODEL, Loop, ask
from loopkit.models import Answer, Check, Citation, InstructionCheck, ResponseKind

try:
    __version__ = version("loop-kit")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "Answer",
    "BOOK_INSTRUCTIONS",
    "Check",
    "Citation",
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_MODEL",
    "DEFAULT_MODELS",
    "InstructionCheck",
    "Knowledge",
    "Loop",
    "LoopError",
    "Passage",
    "ResponseKind",
    "ask",
    "__version__",
]
