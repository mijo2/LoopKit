"""Results returned to the caller. The loop stays inside the library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StoppedReason = Literal["passed", "max_turns"]
ResponseKind = Literal["normal", "book"]


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
class Check:
    """One auditor result. ``revise`` is what the loop acted on.

    ``follows_instructions`` is true only when every instruction held.
    """

    turn: int
    follows_instructions: bool
    instructions: tuple[InstructionCheck, ...]
    grounded: bool
    revise: bool
    notes: str
    gaps: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    """The user-facing text plus the trace of the loop that produced it.

    ``response`` is ``normal`` for an ordinary answer and ``book`` when the
    loop was asked for a long-form response.
    """

    text: str
    response: ResponseKind
    citations: tuple[Citation, ...]
    checks: tuple[Check, ...]
    turns: int
    stopped_reason: StoppedReason

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "response": self.response,
            "turns": self.turns,
            "stopped_reason": self.stopped_reason,
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
                }
                for check in self.checks
            ],
        }
