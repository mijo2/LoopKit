"""Results returned to the caller. The loop stays inside the library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StoppedReason = Literal["passed", "max_turns"]


@dataclass(frozen=True)
class Citation:
    """A passage the answer actually used."""

    id: str
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class Check:
    """One auditor result. ``revise`` is what the loop acted on."""

    turn: int
    follows_instructions: bool
    grounded: bool
    revise: bool
    notes: str
    gaps: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    """The user-facing text plus the trace of the loop that produced it."""

    text: str
    citations: tuple[Citation, ...]
    checks: tuple[Check, ...]
    turns: int
    stopped_reason: StoppedReason

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
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
                    "grounded": check.grounded,
                    "revise": check.revise,
                    "notes": check.notes,
                    "gaps": list(check.gaps),
                }
                for check in self.checks
            ],
        }
