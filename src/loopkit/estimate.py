"""How big a reply will be, before the call is made.

The token meter needs a number for the reply of a call that has not happened
yet. This module holds the rules for that number and nothing else: it reads
the shape of the task and what earlier calls of the same kind produced, and
it returns a token count. It makes no calls and keeps no state, so the same
inputs always give the same estimate.

:class:`EstimationStrategy` names the rule set. There is one today,
:attr:`~EstimationStrategy.LAST_TURN`:

* the first reply of a kind (the first draft, the first instruction check,
  and so on) is sized from the input alone, with :func:`first_output`;
* every later reply of that kind is assumed to be the size of the latest one
  the model actually produced, which is what the meter has observed.

``kind`` is the name of the schema a call parses into: ``Draft``,
``OneCheck``, ``GroundingCheck``, ``DraftCheck``, or ``RolePrefix``.

The same rules also give the loop a **response length** to ask for, see
:func:`response_length_target`: the expected draft size, or the length the
caller set, held under what the model can return in one call
(:func:`max_response_chars`). Whether a draft kept to it is decided by
:func:`length_compliant`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class EstimationStrategy(StrEnum):
    """Which rules size a reply that has not been produced yet."""

    LAST_TURN = "last_turn"
    """First reply of a kind from the input; later ones the size of the latest of that kind."""


@dataclass(frozen=True)
class AskShape:
    """What is known about a task before the first call, in tokens and counts.

    ``response_chars`` is a response length the caller asked for, in
    characters, or ``None`` when the length is left to the estimate.
    """

    task_tokens: int
    instruction_count: int
    passage_tokens: int
    response: str = "normal"
    mode: str = "knowledge"
    response_chars: int | None = None


EMPTY_SHAPE = AskShape(task_tokens=0, instruction_count=0, passage_tokens=0)
"""The shape of nothing: a meter built without a task sizes replies from the constants alone."""


# The structured reply wraps the text in JSON: the field names, quotes, and
# the list of cited ids.
_JSON_OVERHEAD = 30

# Characters of answer text per output token, for turning a reply size into
# a length the model can be told and the caller can read.
CHARS_PER_TOKEN = 4

# How much a draft may run over the length it was asked for and still count
# as compliant. Models do not count characters; a tenth is noise, more is not.
LENGTH_TOLERANCE = 0.10

# The smallest response length the loop will ask for. Below this an answer
# cannot say that the passages do not contain the answer and cite them.
MIN_RESPONSE_CHARS = 200

# The most a model returns in one call, in characters of answer text, by the
# start of its name. These are assumptions for the default models, not
# measured limits: about 128k output tokens for Grok and GPT, 64k for Claude.
# A model that is not listed gets DEFAULT_MAX_RESPONSE_CHARS, the size of an
# 8k-token reply, which is what most OpenAI-compatible servers allow.
#
# TODO(response-length): a response longer than one call is not produced
# yet. The target is held under the model's maximum, so a book-length answer
# on a small model is cut to what one call can return. The plan is to draft
# such answers in parts, each under the maximum, and stitch them: the draft
# prompt gets the parts so far and the section to write next, and the checks
# run on the whole. Track this under the "response length" feature.
MODEL_MAX_RESPONSE_CHARS: Mapping[str, int] = {
    "grok": 128_000 * CHARS_PER_TOKEN,
    "gpt": 128_000 * CHARS_PER_TOKEN,
    "claude": 64_000 * CHARS_PER_TOKEN,
}
DEFAULT_MAX_RESPONSE_CHARS = 8_000 * CHARS_PER_TOKEN

# First-draft rules, by mode and response kind: a constant, a share of the
# passage tokens, a share of the task tokens, and the range the result is
# held to. A concise answer restates a part of the passages that bears on
# the task; a book-length one can restate most of them with headings.
# Open mode has no passages, so the task is the only size signal.
_FIRST_DRAFT: Mapping[tuple[str, str], tuple[int, float, float, int, int]] = {
    # (mode, response): (constant, per passage token, per task token, floor, ceiling)
    ("knowledge", "normal"): (60, 0.25, 0.5, 120, 1_500),
    ("knowledge", "book"): (300, 1.2, 0.0, 1_000, 20_000),
    ("open", "normal"): (250, 0.0, 3.0, 120, 1_500),
    ("open", "book"): (3_000, 0.0, 10.0, 1_500, 20_000),
}

# First replies of the check and optimize steps. A verdict is a boolean and
# a note; the grounding audit also lists gaps; the combined check carries one
# verdict per instruction; the role is two to four sentences.
_FIRST_ONE_CHECK = 100
_FIRST_GROUNDING = 200
_FIRST_ALL_CHECK_BASE = 80
_FIRST_ALL_CHECK_PER_INSTRUCTION = 60
_FIRST_ROLE = 120
_FIRST_UNKNOWN = 300


def first_output(kind: str, shape: AskShape) -> int:
    """The expected size of the first reply of ``kind``, from the input alone."""

    if kind == "Draft":
        return first_draft(shape)
    if kind == "OneCheck":
        return _FIRST_ONE_CHECK
    if kind == "GroundingCheck":
        return _FIRST_GROUNDING
    if kind == "DraftCheck":
        return _FIRST_ALL_CHECK_BASE + _FIRST_ALL_CHECK_PER_INSTRUCTION * shape.instruction_count
    if kind == "RolePrefix":
        return _FIRST_ROLE
    return _FIRST_UNKNOWN


def first_draft(shape: AskShape) -> int:
    """The expected size of the first draft reply, from the task and the passages.

    When the caller set a response length, the reply is expected to be that
    long, plus the JSON around it.
    """

    if shape.response_chars is not None:
        return math.ceil(shape.response_chars / CHARS_PER_TOKEN) + _JSON_OVERHEAD
    mode = "open" if shape.mode == "open" else "knowledge"
    response = "book" if shape.response == "book" else "normal"
    constant, per_passage, per_task, floor, ceiling = _FIRST_DRAFT[(mode, response)]
    raw = constant + per_passage * shape.passage_tokens + per_task * shape.task_tokens
    return int(min(ceiling, max(floor, raw + _JSON_OVERHEAD)))


def max_response_chars(model: str, output_cap_tokens: int | None = None) -> int:
    """The most ``model`` returns in one call, in characters of answer text.

    Looked up by the start of the model name, see
    :data:`MODEL_MAX_RESPONSE_CHARS`. ``output_cap_tokens`` is the cap the
    client puts on a call when it has one; the answer cannot be longer than
    that either, less the JSON around it.
    """

    name = model.strip().lower()
    limit = DEFAULT_MAX_RESPONSE_CHARS
    for prefix, chars in sorted(MODEL_MAX_RESPONSE_CHARS.items(), key=lambda item: -len(item[0])):
        if name.startswith(prefix):
            limit = chars
            break
    if output_cap_tokens is not None:
        limit = min(limit, max(0, output_cap_tokens - _JSON_OVERHEAD) * CHARS_PER_TOKEN)
    return max(MIN_RESPONSE_CHARS, limit)


def response_length_target(shape: AskShape, model_max_chars: int) -> int:
    """The length, in characters, to ask the draft for.

    The caller's ``response_chars`` when set, else the expected first draft
    less its JSON; either way at least :data:`MIN_RESPONSE_CHARS` and at
    most ``model_max_chars``, so one call can return it.
    """

    if shape.response_chars is not None:
        wanted = shape.response_chars
    else:
        wanted = (first_draft(shape) - _JSON_OVERHEAD) * CHARS_PER_TOKEN
    return int(min(model_max_chars, max(MIN_RESPONSE_CHARS, wanted)))


def length_compliant(actual_chars: int, limit_chars: int) -> bool:
    """Whether a draft of ``actual_chars`` kept to ``limit_chars``, within the tolerance."""

    return actual_chars <= limit_chars * (1 + LENGTH_TOLERANCE)


def words_for(chars: int) -> int:
    """About how many English words fit in ``chars`` characters."""

    return max(1, round(chars / 6))


def expected_output(
    strategy: EstimationStrategy,
    kind: str,
    shape: AskShape,
    last: Mapping[str, int],
) -> int:
    """The tokens the next reply of ``kind`` is expected to take.

    ``last`` maps a kind to the size of the latest reply of that kind in this
    task; it is empty before the first call.
    """

    if strategy is EstimationStrategy.LAST_TURN:
        seen = last.get(kind)
        if seen is not None:
            return seen
        return first_output(kind, shape)
    raise ValueError(f"Unknown estimation strategy: {strategy!r}.")


def coerce_strategy(value: object) -> EstimationStrategy:
    """An :class:`EstimationStrategy` from the enum itself or its value; None is the default."""

    if value is None:
        return EstimationStrategy.LAST_TURN
    if isinstance(value, EstimationStrategy):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        for strategy in EstimationStrategy:
            if cleaned in {strategy.value, strategy.name.lower()}:
                return strategy
    names = ", ".join(strategy.value for strategy in EstimationStrategy)
    raise ValueError(f"estimation_strategy must be one of: {names}.")
