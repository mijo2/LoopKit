"""Run retrieve, then draft, check, and revise.

The model fills in one step at a time. This module decides when to stop:
the auditor accepts the draft, or the turn budget is spent.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeVar

from loopkit.client import DEFAULT_MODELS, make_client
from loopkit.config import load_project
from loopkit.errors import LoopError
from loopkit.knowledge import Knowledge, KnowledgeLike, Passage
from loopkit.models import Answer, Check, Citation, StoppedReason
from loopkit.prompts import CHECK_SYSTEM, DRAFT_SYSTEM, check_prompt, citation_ids, draft_prompt
from loopkit.retrieve import select
from loopkit.schemas import Draft, DraftCheck

DEFAULT_MODEL = DEFAULT_MODELS["xai"]
DEFAULT_MAX_TURNS = 3
DEFAULT_TOP_K = 8
DEFAULT_CONTEXT_BUDGET = 16_000
DEFAULT_INSTRUCTIONS = """\
Answer the task using only the knowledge passages.
Cite supporting passages inline as [p1], [p2], and so on.
If the passages do not contain the answer, say that clearly and do not guess.
Be concise. Do not add facts that are not in the passages.
"""

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
        instructions: str | None = None,
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
    ) -> None:
        if max_turns < 1:
            raise LoopError("max_turns must be at least 1.")
        if top_k < 1:
            raise LoopError("top_k must be at least 1.")
        if context_budget < 1:
            raise LoopError("context_budget must be at least 1.")
        self.knowledge = knowledge if isinstance(knowledge, Knowledge) else Knowledge.load(knowledge)
        self.instructions = (instructions or DEFAULT_INSTRUCTIONS).strip()
        self.max_turns = max_turns
        self.top_k = top_k
        self.context_budget = context_budget
        self.trace = _env_flag("LOOPKIT_TRACE") if trace is None else trace
        if client is not None:
            self.provider = "custom"
            self.model = (model or os.environ.get("LOOPKIT_MODEL", "")).strip() or DEFAULT_MODEL
            self._client: ModelClient = client
        else:
            built = make_client(provider=provider, model=model, api_key=api_key, base_url=base_url)
            self.provider = built.provider
            self.model = built.model
            self._client = built

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: object) -> Loop:
        """Build a loop from ``loopkit.toml``."""

        project = load_project(path)
        instructions = overrides.pop("instructions", project.instructions)
        model = overrides.pop("model", project.model)
        provider = overrides.pop("provider", project.provider)
        base_url = overrides.pop("base_url", project.base_url)
        max_turns = overrides.pop("max_turns", project.max_turns)
        top_k = overrides.pop("top_k", project.top_k)
        return cls(
            project.knowledge,
            None if instructions is None else str(instructions),
            model=None if model is None else str(model),
            provider=None if provider is None else str(provider),
            base_url=None if base_url is None else str(base_url),
            max_turns=_override_int(max_turns, DEFAULT_MAX_TURNS, "max_turns"),
            top_k=_override_int(top_k, DEFAULT_TOP_K, "top_k"),
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
                ),
                schema=Draft,
            )
            self._say(f"turn {turn}: check")
            audit = self._step(
                stage=f"check (turn {turn})",
                system_text=CHECK_SYSTEM,
                user_text=check_prompt(
                    instructions=self.instructions,
                    task=question,
                    passages=retrieval.passages,
                    draft=draft,
                ),
                schema=DraftCheck,
            )
            unknown = [cited for cited in citation_ids(draft) if cited not in known]
            problem = _revision_problem(draft, audit, unknown)
            checks.append(_record(turn, audit, problem, unknown))
            if problem is None:
                self._say("stop: passed")
                return _finish(draft, known, checks, turn, "passed")
            if turn == self.max_turns:
                self._say("stop: max turns")
                return _finish(draft, known, checks, turn, "max_turns")
            self._say("revise")
            previous = draft
            feedback = problem

        raise LoopError("The loop ended without an answer.")

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
    instructions: str | None = None,
    **kwargs: object,
) -> Answer:
    """Answer one task. With no knowledge argument, read ``loopkit.toml``."""

    if knowledge is None:
        if instructions is not None:
            kwargs["instructions"] = instructions
        return Loop.load(**kwargs).ask(task)
    return Loop(knowledge, instructions, **kwargs).ask(task)  # type: ignore[arg-type]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _override_int(value: object, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LoopError(f"{name} must be an integer of at least 1.")
    return value


def _revision_problem(draft: Draft, audit: DraftCheck, unknown: list[str]) -> str | None:
    problems: list[str] = []
    if not draft.answer.strip():
        problems.append("The draft was empty. Write the answer the user should read.")
    if unknown:
        listed = ", ".join(unknown)
        problems.append(
            f"These cited ids are not in the passages: {listed}. Cite only the ids you were given."
        )
    # ``revise`` is the auditor's stop bit. A failed instruction or grounding
    # flag also sends the draft back, even if the auditor left revise false.
    needs_work = audit.revise or not audit.follows_instructions or not audit.grounded
    if needs_work:
        note = audit.notes.strip()
        if note:
            problems.append(note)
        for gap in audit.gaps:
            cleaned = gap.strip()
            if cleaned:
                problems.append(f"Gap: {cleaned}")
        if not problems:
            problems.append("The draft does not follow the instructions or is not grounded in the passages.")
    if not problems:
        return None
    return "\n".join(problems)


def _record(turn: int, audit: DraftCheck, problem: str | None, unknown: list[str]) -> Check:
    return Check(
        turn=turn,
        follows_instructions=audit.follows_instructions,
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
        citations=tuple(citations),
        checks=tuple(checks),
        turns=turns,
        stopped_reason=reason,
    )
