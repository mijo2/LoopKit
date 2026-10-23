"""Answer from a knowledge base. The loop stays in this package."""

from importlib.metadata import PackageNotFoundError, version

from loopkit.client import DEFAULT_MODELS
from loopkit.errors import LoopError, LoopWarning
from loopkit.estimate import EstimationStrategy
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
from loopkit.models import (
    Answer,
    Check,
    Citation,
    InstructionCheck,
    LengthCheck,
    OutputFormat,
    ResponseKind,
    StructureCheck,
    TokenUsage,
)
from loopkit.tokens import estimate_tokens

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
    "EstimationStrategy",
    "InstructionCheck",
    "Knowledge",
    "LengthCheck",
    "Loop",
    "LoopError",
    "LoopWarning",
    "OPEN_BOOK_INSTRUCTIONS",
    "OPEN_INSTRUCTIONS",
    "OutputFormat",
    "SEARCH_INSTRUCTIONS",
    "default_instructions",
    "Passage",
    "ResponseKind",
    "StructureCheck",
    "TokenUsage",
    "__version__",
    "ask",
    "estimate_tokens",
    "default_instructions",
]
