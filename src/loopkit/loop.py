"""Run retrieve, then draft, check, and revise.

The model fills in one step at a time. This module decides when to stop:
the auditor accepts the draft, or the turn budget is spent.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast

from loopkit.client import DEFAULT_MODELS, make_client
from loopkit.config import CHECK_MODES, RESPONSES, load_project
from loopkit.errors import LoopError
from loopkit.knowledge import Knowledge, KnowledgeLike, Passage
from loopkit.models import Answer, Check, Citation, InstructionCheck, ResponseKind, StoppedReason
from loopkit.prompts import (
    CHECK_ONE_SYSTEM,
    CHECK_SYSTEM,
    DRAFT_SYSTEM,
    GROUND_SYSTEM,
    check_one_prompt,
    check_prompt,
    citation_ids,
    draft_prompt,
    ground_prompt,
)
from loopkit.retrieve import select
from loopkit.schemas import Draft, DraftCheck, GroundingCheck, InstructionVerdict, OneCheck

CheckMode = Literal["each", "all"]

DEFAULT_MODEL = DEFAULT_MODELS["xai"]
DEFAULT_MAX_TURNS = 3
DEFAULT_TOP_K = 8
DEFAULT_CONTEXT_BUDGET = 16_000
DEFAULT_INSTRUCTIONS = (
    "Answer the task using only the knowledge passages.",
    "Cite supporting passages inline as [p1], [p2], and so on.",
    "If the passages do not contain the answer, say that clearly and do not guess.",
    "Be concise. Do not add facts that are not in the passages.",
)
BOOK_INSTRUCTIONS = (
    "Answer the task using only the knowledge passages.",
    "Cite supporting passages inline as [p1], [p2], and so on.",
    "If the passages do not contain the answer, say that clearly and do not guess.",
    "Write a long response with headings. It may be as long as a book when the passages support that. "
    "Do not add facts that are not in the passages.",
)
# A book-length completion. Providers that cap lower reject the request.
BOOK_MAX_OUTPUT_TOKENS = 100_000

T = TypeVar("T")


class ModelClient(Protocol):
    """The only model surface the loop needs. Tests pass a scripted stand-in."""

    def parse(self, *, system_text: str, user_text: str, schema: type[T]) -> T:
        """Return one structured result for this step."""


class Loop:
    """A knowledge base plus the loop that answers from it.

    Build it once and call :meth:`ask` for each task. Documents are read at
    construction, not on every question.
    """

    def __init__(
        self,
        knowledge: KnowledgeLike,
        instructions: str | Sequence[str] | None = None,
        *,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        top_k: int = DEFAULT_TOP_K,
        context_budget: int = DEFAULT_CONTEXT_BUDGET,
        api_key: str | None = None,
        client: ModelClient | None = None,
        trace: bool | None = None,
        check_mode: str | None = None,
        response: str | None = None,
    ) -> None:
        if max_turns < 1:
            raise LoopError("max_turns must be at least 1.")
        if top_k < 1:
            raise LoopError("top_k must be at least 1.")
        if context_budget < 1:
            raise LoopError("context_budget must be at least 1.")
        self.check_mode = cast(CheckMode, _choice(check_mode, "each", "check_mode", CHECK_MODES))
        self.response = cast(ResponseKind, _choice(response, "normal", "response", RESPONSES))
        self.knowledge = knowledge if isinstance(knowledge, Knowledge) else Knowledge.load(knowledge)
        if instructions is None and self.response == "book":
            instructions = BOOK_INSTRUCTIONS
        self.instructions = normalize_instructions(instructions)
        self.max_turns = max_turns
        self.top_k = top_k
        self.context_budget = context_budget
        self.trace = _env_flag("LOOPKIT_TRACE") if trace is None else trace
        if client is not None:
            self.provider = "custom"
            self.model = (model or os.environ.get("LOOPKIT_MODEL", "")).strip() or DEFAULT_MODEL
            self._client: ModelClient = client
        else:
            output_tokens = BOOK_MAX_OUTPUT_TOKENS if self.response == "book" else None
            built = make_client(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_output_tokens=output_tokens,
            )
            self.provider = built.provider
            self.model = built.model
            self._client = built

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: object) -> Loop:
        """Build a loop from ``loopkit.toml``."""

        project = load_project(path)
        instructions = _coerce_instructions(overrides.pop("instructions", project.instructions))
        model = overrides.pop("model", project.model)
        provider = overrides.pop("provider", project.provider)
        base_url = overrides.pop("base_url", project.base_url)
        max_turns = overrides.pop("max_turns", project.max_turns)
        top_k = overrides.pop("top_k", project.top_k)
        check_mode = overrides.pop("check_mode", project.check_mode)
        response = overrides.pop("response", project.response)
        return cls(
            project.knowledge,
            instructions,
            model=None if model is None else str(model),
            provider=None if provider is None else str(provider),
            base_url=None if base_url is None else str(base_url),
            max_turns=_override_int(max_turns, DEFAULT_MAX_TURNS, "max_turns"),
            top_k=_override_int(top_k, DEFAULT_TOP_K, "top_k"),
            check_mode=None if check_mode is None else str(check_mode),
            response=None if response is None else str(response),
            **overrides,  # type: ignore[arg-type]
        )

    def ask(self, task: str) -> Answer:
        """Answer one task from the loaded knowledge base."""

        question = task.strip()
        if not question:
            raise LoopError("The task is empty.")

        retrieval = select(
            self.knowledge.passages,
            question,
            top_k=self.top_k,
            budget=self.context_budget,
        )
        self._say(
            f"{len(self.knowledge.passages)} passages loaded, {len(retrieval.passages)} sent to the model"
        )
        if retrieval.weak:
            self._say("no lexical overlap; the drafter is told not to stretch a passage")
        known = {passage.id: passage for passage in retrieval.passages}
        checks: list[Check] = []
        previous: Draft | None = None
        feedback: str | None = None

        for turn in range(1, self.max_turns + 1):
            self._say(f"turn {turn}: draft")
            draft = self._step(
                stage=f"draft (turn {turn})",
                system_text=DRAFT_SYSTEM,
                user_text=draft_prompt(
                    instructions=self.instructions,
                    task=question,
                    passages=retrieval.passages,
                    weak=retrieval.weak,
                    previous=previous,
                    feedback=feedback,
                    response=self.response,
                ),
                schema=Draft,
            )
            audit = self._audit(turn, question, retrieval.passages, draft)
            unknown = [cited for cited in citation_ids(draft) if cited not in known]
            problem, judged = _revision_problem(draft, audit, unknown, self.instructions)
            checks.append(_record(turn, audit, problem, unknown, judged))
            missed = [str(item.number) for item in judged if not item.followed]
            if missed:
                self._say("instructions not followed: " + ", ".join(missed))
            if problem is None:
                self._say("stop: passed")
                return _finish(draft, known, checks, turn, "passed", self.response)
            if turn == self.max_turns:
                self._say("stop: max turns")
                return _finish(draft, known, checks, turn, "max_turns", self.response)
            self._say("revise")
            previous = draft
            feedback = problem

        raise LoopError("The loop ended without an answer.")

    def _audit(self, turn: int, task: str, passages: Sequence[Passage], draft: Draft) -> DraftCheck:
        """Check the draft.

        ``each`` asks about one instruction per call, then checks grounding
        once. ``all`` scores every instruction in a single call.
        """

        if self.check_mode == "all":
            self._say(f"turn {turn}: check")
            return self._step(
                stage=f"check (turn {turn})",
                system_text=CHECK_SYSTEM,
                user_text=check_prompt(
                    instructions=self.instructions,
                    task=task,
                    passages=passages,
                    draft=draft,
                    response=self.response,
                ),
                schema=DraftCheck,
            )

        verdicts: list[InstructionVerdict] = []
        for number, instruction in enumerate(self.instructions, start=1):
            self._say(f"turn {turn}: check instruction {number}")
            one = self._step(
                stage=f"check instruction {number} (turn {turn})",
                system_text=CHECK_ONE_SYSTEM,
                user_text=check_one_prompt(
                    number=number,
                    instruction=instruction,
                    task=task,
                    passages=passages,
                    draft=draft,
                ),
                schema=OneCheck,
            )
            verdicts.append(InstructionVerdict(number=number, followed=one.followed, note=one.note))
        self._say(f"turn {turn}: check grounding")
        ground = self._step(
            stage=f"check grounding (turn {turn})",
            system_text=GROUND_SYSTEM,
            user_text=ground_prompt(task=task, passages=passages, draft=draft),
            schema=GroundingCheck,
        )
        return DraftCheck(
            instructions=verdicts,
            grounded=ground.grounded,
            revise=ground.revise,
            notes=ground.notes,
            gaps=list(ground.gaps),
        )

    def _step(self, *, stage: str, system_text: str, user_text: str, schema: type[T]) -> T:
        try:
            parsed = self._client.parse(system_text=system_text, user_text=user_text, schema=schema)
        except LoopError:
            raise
        except Exception as exc:
            raise LoopError(f"The model call failed during {stage}: {exc}") from exc
        if not isinstance(parsed, schema):
            raise LoopError(
                f"The model returned {type(parsed).__name__} during {stage}, expected {schema.__name__}."
            )
        return parsed

    def _say(self, message: str) -> None:
        if self.trace:
            print(f"loopkit: {message}", file=sys.stderr, flush=True)


def ask(
    task: str,
    knowledge: KnowledgeLike | None = None,
    instructions: str | Sequence[str] | None = None,
    **kwargs: object,
) -> Answer:
    """Answer one task. With no knowledge argument, read ``loopkit.toml``."""

    if knowledge is None:
        if instructions is not None:
            kwargs["instructions"] = instructions
        return Loop.load(**kwargs).ask(task)
    return Loop(knowledge, instructions, **kwargs).ask(task)  # type: ignore[arg-type]


def normalize_instructions(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Turn instructions into one entry per rule.

    None selects the built-in list. A string is split on non-empty lines.
    A list is already separate: each item is one instruction, and line breaks
    inside an item stay part of that instruction.
    """

    if value is None:
        return DEFAULT_INSTRUCTIONS
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if not lines:
            raise LoopError("Instructions must contain at least one non-empty line.")
        return tuple(lines)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items: list[str] = []
        for piece in value:
            if not isinstance(piece, str):
                raise LoopError("Each instruction must be a string.")
            text = piece.strip()
            if not text:
                raise LoopError("Each instruction must be a non-empty string.")
            items.append(text)
        if not items:
            raise LoopError("Instructions must contain at least one non-empty line.")
        return tuple(items)
    raise LoopError("Instructions must be a string or a list of strings.")


def _coerce_instructions(value: object) -> str | Sequence[str] | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LoopError("Each instruction must be a string.")
            items.append(item)
        return items
    raise LoopError("Instructions must be a string or a list of strings.")


def _choice(value: object, default: str, name: str, allowed: tuple[str, ...]) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise LoopError(f"{name} must be one of: {', '.join(allowed)}.")
    cleaned = value.strip().lower()
    if cleaned not in allowed:
        raise LoopError(f"{name} must be one of: {', '.join(allowed)}.")
    return cleaned


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _override_int(value: object, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LoopError(f"{name} must be an integer of at least 1.")
    return value


def _revision_problem(
    draft: Draft,
    audit: DraftCheck,
    unknown: list[str],
    instructions: Sequence[str],
) -> tuple[str | None, tuple[InstructionCheck, ...]]:
    judged, instruction_problems = _judge_instructions(instructions, audit.instructions)
    problems: list[str] = []
    if not draft.answer.strip():
        problems.append("The draft was empty. Write the answer the user should read.")
    if unknown:
        listed = ", ".join(unknown)
        problems.append(
            f"These cited ids are not in the passages: {listed}. Cite only the ids you were given."
        )
    problems.extend(instruction_problems)
    # ``revise`` is the auditor's stop bit. Any instruction that was not followed,
    # and a grounding miss, also send the draft back.
    needs_work = bool(instruction_problems) or audit.revise or not audit.grounded
    if needs_work:
        note = audit.notes.strip()
        if note:
            problems.append(note)
        for gap in audit.gaps:
            cleaned = gap.strip()
            if cleaned:
                problems.append(f"Gap: {cleaned}")
        if not problems:
            problems.append(
                "The draft does not follow the instructions or is not grounded in the passages."
            )
    if not problems:
        return None, judged
    return "\n".join(problems), judged


def _judge_instructions(
    instructions: Sequence[str],
    verdicts: Sequence[InstructionVerdict],
) -> tuple[tuple[InstructionCheck, ...], list[str]]:
    """Pair each instruction with the verdict of that number.

    A missing verdict counts as not followed. When the same number is judged
    twice, a failure wins.
    """

    by_number: dict[int, InstructionVerdict] = {}
    for verdict in verdicts:
        current = by_number.get(verdict.number)
        if current is None or (current.followed and not verdict.followed):
            by_number[verdict.number] = verdict

    judged: list[InstructionCheck] = []
    problems: list[str] = []
    for number, instruction in enumerate(instructions, start=1):
        verdict = by_number.get(number)
        if verdict is not None and verdict.followed:
            judged.append(
                InstructionCheck(number=number, instruction=instruction, followed=True, note="")
            )
            continue
        if verdict is None:
            note = "The check did not judge this instruction."
            problems.append(f"Instruction {number} was not judged: {instruction}")
        else:
            note = verdict.note.strip()
            if note:
                problems.append(f"Instruction {number} was not followed: {instruction}\n{note}")
            else:
                problems.append(f"Instruction {number} was not followed: {instruction}")
        judged.append(
            InstructionCheck(number=number, instruction=instruction, followed=False, note=note)
        )
    return tuple(judged), problems


def _record(
    turn: int,
    audit: DraftCheck,
    problem: str | None,
    unknown: list[str],
    instructions: Sequence[InstructionCheck],
) -> Check:
    recorded = tuple(instructions)
    return Check(
        turn=turn,
        follows_instructions=all(item.followed for item in recorded),
        instructions=recorded,
        grounded=audit.grounded and not unknown,
        revise=problem is not None,
        notes=problem if problem is not None else audit.notes.strip(),
        gaps=tuple(gap.strip() for gap in audit.gaps if gap.strip()),
    )


def _finish(
    draft: Draft,
    known: Mapping[str, Passage],
    checks: list[Check],
    turns: int,
    reason: StoppedReason,
    response: ResponseKind,
) -> Answer:
    citations: list[Citation] = []
    for cited in citation_ids(draft):
        passage = known.get(cited)
        if passage is None:
            continue
        citations.append(
            Citation(id=passage.id, source=passage.source, heading=passage.heading, text=passage.text)
        )
    return Answer(
        text=draft.answer.strip(),
        response=response,
        citations=tuple(citations),
        checks=tuple(checks),
        turns=turns,
        stopped_reason=reason,
    )
