"""A model that returns scripted structured results and records the prompts."""

from __future__ import annotations

from loopkit.schemas import Draft, DraftCheck


class ScriptedModel:
    def __init__(self, replies: list[object]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def parse(self, *, system_text: str, user_text: str, schema: type[object]) -> object:
        self.calls.append({"system": system_text, "user": user_text, "schema": schema})
        if not self._replies:
            raise AssertionError("ScriptedModel ran out of replies")
        reply = self._replies.pop(0)
        if not isinstance(reply, schema):
            raise AssertionError(f"Expected {schema.__name__}, scripted {type(reply).__name__}")
        return reply


def accepted(answer: str, *ids: str) -> list[object]:
    return [
        Draft(answer=answer, cited_ids=list(ids)),
        DraftCheck(
            follows_instructions=True,
            grounded=True,
            revise=False,
            notes="",
            gaps=[],
        ),
    ]


def audit(*, revise: bool, notes: str = "", grounded: bool = True, follows: bool = True, gaps: list[str] | None = None) -> DraftCheck:
    return DraftCheck(
        follows_instructions=follows,
        grounded=grounded,
        revise=revise,
        notes=notes,
        gaps=gaps or [],
    )
