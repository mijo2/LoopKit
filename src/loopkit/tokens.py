"""Token accounting for one ``ask``: what was spent, and whether the next call fits.

The loop is code, so the budget is enforced in code. :class:`TokenMeter`
keeps a running total across every model call of one ``ask``. Providers
report what a call really cost (``usage`` in the reply); when that is
missing, the cost is estimated from the prompt and the parsed reply with
:func:`estimate_tokens`.

Before a call the loop asks :meth:`TokenMeter.fits` with
:meth:`TokenMeter.estimate_call`. The estimate is the prompt length, scaled
by what the provider has reported so far for this task (the same passages
appear in every prompt, so the ratio settles quickly), plus the reply size
that :mod:`loopkit.estimate` expects for that kind of call.

The loop never cuts a call short, so a total can end over the limit by at
most the call that crossed it. What happens then, and how far ahead the
loop looks, is the loop's ``token_budget_mode``: see :class:`loopkit.Loop`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from loopkit.estimate import EMPTY_SHAPE, AskShape, EstimationStrategy, expected_output

TokenEstimator = Callable[[str], int]
"""A function from text to a token count. The default is :func:`estimate_tokens`."""

# Characters per token for English prose under the common BPE tokenizers.
_CHARS_PER_TOKEN = 4.0
# Tokens per whitespace-separated word for the same text; a cross-check so
# that text with long words, ids, or markup is not under-counted.
_TOKENS_PER_WORD = 1.3

# The correction applied to prompt estimates is kept inside this range, so a
# single odd report cannot make the meter blind or paranoid.
_MIN_SCALE = 0.5
_MAX_SCALE = 4.0


def estimate_tokens(text: str) -> int:
    """A rough token count for ``text`` with no tokenizer dependency.

    English prose comes out within about a quarter of what the provider
    tokenizers report; code, JSON, and ids run denser and are counted by
    words as well as characters, whichever is larger. Characters outside
    ASCII count one token each, which is close for most scripts. Pass a real
    tokenizer as ``token_estimator`` to :class:`~loopkit.Loop` when the
    estimate matters more than the dependency.
    """

    if not text:
        return 0
    ascii_count = sum(1 for character in text if ord(character) < 128)
    other = len(text) - ascii_count
    by_chars = ascii_count / _CHARS_PER_TOKEN
    by_words = len(text.split()) * _TOKENS_PER_WORD
    return math.ceil(max(by_chars, by_words) + other)


@dataclass(frozen=True)
class Usage:
    """Tokens one model call consumed, as reported by the provider or estimated."""

    input_tokens: int
    output_tokens: int
    estimated: bool = False

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


def usage_from_response(payload: object) -> Usage | None:
    """Read reported usage out of a provider reply, or ``None`` when it has none.

    Accepts the OpenAI Responses shape (``input_tokens``/``output_tokens``),
    the chat completions shape (``prompt_tokens``/``completion_tokens``),
    Anthropic's (``input_tokens``/``output_tokens`` plus cache counters),
    and the xAI SDK's usage object, which exposes the chat completions names
    as attributes. ``payload`` may be the whole reply or just its ``usage``.
    """

    usage: object = payload
    if isinstance(payload, Mapping):
        if "usage" in payload:
            usage = payload["usage"]
    elif hasattr(payload, "usage"):
        usage = payload.usage
    if usage is None:
        return None

    def read(*names: str) -> int | None:
        for name in names:
            value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value >= 0:
                return int(value)
        return None

    input_tokens = read("input_tokens", "prompt_tokens")
    output_tokens = read("output_tokens", "completion_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    # Anthropic bills prompt cache writes and reads separately from input_tokens.
    for name in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        extra = read(name)
        if extra:
            input_tokens += extra
    # Reasoning tokens are billed as output. Some providers fold them into
    # completion_tokens, others report them only in total_tokens. The total,
    # when given, is the number to trust.
    total = read("total_tokens")
    if total is not None and total > input_tokens + output_tokens:
        output_tokens = total - input_tokens
    if input_tokens == 0 and output_tokens == 0:
        # A protobuf with unset fields reads as zeros. No call costs nothing.
        return None
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


class TokenMeter:
    """Running token spend for one ``ask`` and the limit it is held to.

    ``limit`` of ``None`` means no budget: every call fits and the meter only
    counts. ``estimator`` turns text into a token count for calls the
    provider did not report and for the estimate before a call. ``strategy``
    and ``shape`` size the replies that have not been produced yet, see
    :mod:`loopkit.estimate`.
    """

    def __init__(
        self,
        limit: int | None = None,
        *,
        estimator: TokenEstimator = estimate_tokens,
        strategy: EstimationStrategy = EstimationStrategy.LAST_TURN,
        shape: AskShape = EMPTY_SHAPE,
    ) -> None:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1.")
        self.limit = limit
        self.estimator = estimator
        self.strategy = strategy
        self.shape = shape
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.estimated = False
        self._scale = 1.0
        self._last: dict[str, int] = {}

    @property
    def used(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remaining(self) -> int | None:
        """Tokens left under the limit, never negative; ``None`` with no limit."""

        if self.limit is None:
            return None
        return max(0, self.limit - self.used)

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self.used > self.limit

    def expected_output(self, kind: str) -> int:
        """The reply size expected for the next call of ``kind``."""

        return expected_output(self.strategy, kind, self.shape, self._last)

    def estimate_call(
        self, system_text: str, user_text: str, kind: str, *, extra_prompt_tokens: int = 0
    ) -> int:
        """What a call with these prompts and this reply kind will probably cost.

        ``extra_prompt_tokens`` is for text that will be in the prompt but
        does not exist yet, such as the draft a check will read.
        """

        prompt = self.estimator(system_text) + self.estimator(user_text) + extra_prompt_tokens
        return math.ceil(prompt * self._scale) + self.expected_output(kind)

    def fits(self, cost: int) -> bool:
        """True when a call of ``cost`` tokens stays inside the limit."""

        if self.limit is None:
            return True
        return self.used + cost <= self.limit

    def record(
        self,
        *,
        system_text: str,
        user_text: str,
        output_text: str,
        kind: str,
        reported: Usage | None,
    ) -> Usage:
        """Add one finished call and return what it was counted as.

        ``reported`` is the provider's usage when the client had it. Without
        it the call is counted from the prompts and ``output_text`` (the
        parsed reply, serialised), and :attr:`estimated` is set.
        """

        prompt = self.estimator(system_text) + self.estimator(user_text)
        if reported is None:
            usage = Usage(prompt, self.estimator(output_text), estimated=True)
            self.estimated = True
        else:
            usage = reported
            if prompt > 0 and usage.input_tokens > 0:
                self._scale = min(_MAX_SCALE, max(_MIN_SCALE, usage.input_tokens / prompt))
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self._last[kind] = usage.output_tokens
        return usage

    def describe(self) -> str:
        """``1,234 of 5,000 tokens used`` or ``1,234 tokens used``, for trace lines."""

        prefix = "about " if self.estimated else ""
        if self.limit is None:
            return f"{prefix}{self.used:,} tokens used"
        return f"{prefix}{self.used:,} of {self.limit:,} tokens used"
