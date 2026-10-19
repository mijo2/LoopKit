"""A model that returns scripted structured results and records the prompts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict, TypeVar

from pydantic import BaseModel

from loopkit.loop import DEFAULT_INSTRUCTIONS
from loopkit.schemas import Draft, DraftCheck, GroundingCheck, InstructionVerdict, OneCheck

T = TypeVar("T", bound=BaseModel)


class Call(TypedDict):
    """One recorded model call: the two prompt halves and the schema that was requested."""

    system: str
    user: str
    schema: type[BaseModel]


class ScriptedModel:
    """Stands in for a provider client. Satisfies ``loopkit.loop.ModelClient``."""

    def __init__(self, replies: list[BaseModel]) -> None:
        self._replies = list(replies)
        self.calls: list[Call] = []

    def parse(self, *, system_text: str, user_text: str, schema: type[T]) -> T:
        self.calls.append({"system": system_text, "user": user_text, "schema": schema})
        if not self._replies:
            raise AssertionError("ScriptedModel ran out of replies")
        reply = self._replies.pop(0)
        if not isinstance(reply, schema):
            raise AssertionError(f"Expected {schema.__name__}, scripted {type(reply).__name__}")
        return reply


class SearchAwareModel(ScriptedModel):
    """A scripted model whose parse accepts ``search``, recording each flag."""

    def __init__(self, replies: list[object]) -> None:
        super().__init__(replies)
        self.search_flags: list[bool] = []

    def parse(self, *, system_text: str, user_text: str, schema: type[object], search: bool = False) -> object:
        self.search_flags.append(search)
        return super().parse(system_text=system_text, user_text=user_text, schema=schema)


def _verdicts(
    count: int | None = None,
    *,
    follows: bool = True,
    failed: Sequence[int] = (),
    note: str = "",
) -> list[InstructionVerdict]:
    total = len(DEFAULT_INSTRUCTIONS) if count is None else count
    failed_numbers = set(failed)
    if not follows:
        failed_numbers.add(1)
    verdicts: list[InstructionVerdict] = []
    for number in range(1, total + 1):
        missed = number in failed_numbers
        verdicts.append(
            InstructionVerdict(number=number, followed=not missed, note=note if missed else "")
        )
    return verdicts


def checks(
    *,
    revise: bool = False,
    notes: str = "",
    grounded: bool = True,
    follows: bool = True,
    gaps: list[str] | None = None,
    failed: Sequence[int] = (),
    instruction_count: int | None = None,
) -> list[BaseModel]:
    """One turn of per-instruction checks plus the grounding check."""

    total = len(DEFAULT_INSTRUCTIONS) if instruction_count is None else instruction_count
    failed_numbers = set(failed)
    if not follows:
        failed_numbers.add(1)
    failure_note = notes if not follows or failed else ""
    replies: list[BaseModel] = []
    for number in range(1, total + 1):
        missed = number in failed_numbers
        replies.append(OneCheck(followed=not missed, note=failure_note if missed else ""))
    replies.append(GroundingCheck(grounded=grounded, revise=revise, notes=notes, gaps=gaps or []))
    return replies


def accepted(answer: str, *ids: str, instructions: int | None = None) -> list[BaseModel]:
    return [
        Draft(answer=answer, cited_ids=list(ids)),
        *checks(instruction_count=instructions),
    ]


def audit(
    *,
    revise: bool,
    notes: str = "",
    grounded: bool = True,
    follows: bool = True,
    gaps: list[str] | None = None,
    failed: Sequence[int] = (),
    instruction_count: int | None = None,
) -> DraftCheck:
    failure_note = notes if not follows or failed else ""
    return DraftCheck(
        instructions=_verdicts(
            instruction_count,
            follows=follows,
            failed=failed,
            note=failure_note,
        ),
        grounded=grounded,
        revise=revise,
        notes=notes,
        gaps=gaps or [],
    )
