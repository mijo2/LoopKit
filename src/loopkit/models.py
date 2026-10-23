"""Results returned to the caller. The loop stays inside the library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OutputFormat = Literal["text", "json"]
"""The shape of the answer: prose, or one JSON document held to a schema."""

StoppedReason = Literal["passed", "max_turns", "token_budget"]
"""Why the loop stopped: every check passed, the turn budget ran out, or the
token budget was reached before the next step."""

ResponseKind = Literal["normal", "book"]
LoopMode = Literal["knowledge", "open"]


@dataclass(frozen=True)
class Citation:
    """A passage the answer actually used."""

    id: str
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class InstructionCheck:
    """Whether this draft followed one instruction."""

    number: int
    instruction: str
    followed: bool
    note: str


@dataclass(frozen=True)
class LengthCheck:
    """Whether a draft kept to the response length it was asked for.

    ``limit`` and ``actual`` are characters of answer text. ``followed`` is
    true when ``actual`` is within the tolerance of ``limit``. Measured in
    code, not by the model.
    """

    limit: int
    actual: int
    followed: bool


@dataclass(frozen=True)
class StructureCheck:
    """Whether a draft was the JSON document the output schema asks for.

    ``problems`` are ``$.path: problem`` lines from the validator, or one
    line saying the answer was not JSON. Measured in code, not by the model.
    """

    followed: bool
    problems: tuple[str, ...]


@dataclass(frozen=True)
class Check:
    """One auditor result. ``revise`` is what the loop acted on.

    ``follows_instructions`` is true only when every instruction held.
    ``length`` is the response length check when the loop was asked to
    control the length, and ``structure`` the output schema check with
    ``output_format="json"``; each is ``None`` otherwise. Neither is an
    instruction, and neither counts toward ``follows_instructions``.
    """

    turn: int
    follows_instructions: bool
    instructions: tuple[InstructionCheck, ...]
    grounded: bool
    revise: bool
    notes: str
    gaps: tuple[str, ...]
    length: LengthCheck | None = None
    structure: StructureCheck | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Tokens one ``ask`` spent across all of its model calls.

    ``estimated`` is true when at least one call was counted from the prompt
    and reply text because the client did not report usage; the built-in
    provider clients always do. ``budget`` is the soft limit the loop was
    held to, or ``None`` when there was none. ``calls`` is the number of
    model calls made.
    """

    input_tokens: int
    output_tokens: int
    calls: int
    estimated: bool
    budget: int | None

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def exceeded(self) -> bool:
        """True when the total went past the budget (the limit is soft)."""

        return self.budget is not None and self.total > self.budget


@dataclass(frozen=True)
class Answer:
    """The user-facing text plus the trace of the loop that produced it.

    ``response`` is ``normal`` for an ordinary answer and ``book`` when the
    loop was asked for a long-form response. ``mode`` is ``knowledge`` when
    the answer was drafted from passages and ``open`` when there were none
    and the model answered from its own knowledge; in open mode ``grounded``
    on each check means the accuracy audit found no false or invented claim.
    ``role`` is the prefix the optimize step wrote for this task, or ``""``
    when ``auto_optimize`` was off or the step failed. ``tokens`` is what
    the whole ``ask`` spent; with ``stopped_reason="token_budget"`` the
    loop stopped because the next step would not have fit, or the budget was
    already passed, and the latest draft is returned as it stands.
    ``response_length`` is the length, in characters, the drafts were asked
    to keep under, or ``None`` when the length was not controlled; the last
    ``Check.length`` says whether the text did. With
    ``output_format="json"``, ``text`` is the JSON document and ``data`` is
    it parsed (``None`` when the text was not JSON); the last
    ``Check.structure`` says whether it followed the schema.
    """

    text: str
    response: ResponseKind
    citations: tuple[Citation, ...]
    checks: tuple[Check, ...]
    turns: int
    stopped_reason: StoppedReason
    mode: LoopMode = "knowledge"
    role: str = ""
    tokens: TokenUsage = TokenUsage(0, 0, 0, False, None)
    response_length: int | None = None
    output_format: OutputFormat = "text"
    data: object = None

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable view of the answer, as printed by ``loopkit ask --json``."""

        return {
            "text": self.text,
            "response": self.response,
            "mode": self.mode,
            "role": self.role,
            "turns": self.turns,
            "stopped_reason": self.stopped_reason,
            "response_length": self.response_length,
            "output_format": self.output_format,
            "data": self.data,
            "tokens": {
                "input": self.tokens.input_tokens,
                "output": self.tokens.output_tokens,
                "total": self.tokens.total,
                "calls": self.tokens.calls,
                "estimated": self.tokens.estimated,
                "budget": self.tokens.budget,
            },
            "citations": [
                {
                    "id": citation.id,
                    "source": citation.source,
                    "heading": citation.heading,
                    "text": citation.text,
                }
                for citation in self.citations
            ],
            "checks": [
                {
                    "turn": check.turn,
                    "follows_instructions": check.follows_instructions,
                    "instructions": [
                        {
                            "number": item.number,
                            "instruction": item.instruction,
                            "followed": item.followed,
                            "note": item.note,
                        }
                        for item in check.instructions
                    ],
                    "grounded": check.grounded,
                    "revise": check.revise,
                    "notes": check.notes,
                    "gaps": list(check.gaps),
                    "length": None
                    if check.length is None
                    else {
                        "limit": check.length.limit,
                        "actual": check.length.actual,
                        "followed": check.length.followed,
                    },
                    "structure": None
                    if check.structure is None
                    else {
                        "followed": check.structure.followed,
                        "problems": list(check.structure.problems),
                    },
                }
                for check in self.checks
            ],
        }
