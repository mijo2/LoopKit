"""Run retrieve, then draft, check, and revise.

The model fills in one step at a time. This module decides when to stop:
the auditor accepts the draft, or the turn budget is spent.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast

from pydantic import BaseModel

from loopkit.client import DEFAULT_MODELS, make_client
from loopkit.config import CHECK_MODES, RESPONSES, load_project
from loopkit.debuglog import block, enable_debug_log, logger, resolve_debug
from loopkit.errors import LoopError, LoopWarning
from loopkit.knowledge import Knowledge, KnowledgeLike, Passage
from loopkit.models import Answer, Check, Citation, InstructionCheck, ResponseKind, StoppedReason
from loopkit.prompts import (
    OPTIMIZE_SYSTEM,
    check_one_prompt,
    check_one_system,
    check_prompt,
    check_system,
    citation_ids,
    draft_prompt,
    draft_system,
    ground_prompt,
    optimize_prompt,
    review_system,
    with_role,
)
from loopkit.retrieve import select
from loopkit.schemas import Draft, DraftCheck, GroundingCheck, InstructionVerdict, OneCheck, RolePrefix

CheckMode = Literal["each", "all"]
Mode = Literal["knowledge", "open"]

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
    (
        "Write a long response with headings. "
        "It may be as long as a book when the passages support that. "
        "Do not add facts that are not in the passages."
    ),
)
# With web search and passages, the passages still come first.
SEARCH_INSTRUCTIONS = (
    "Answer the task from the knowledge passages first. Use web search only for what the passages "
    "do not cover, and name the web source in the answer for those facts.",
    "Cite supporting passages inline as [p1], [p2], and so on.",
    "If neither the passages nor a search result contains the answer, say that clearly and do not guess.",
    "Be concise. Do not add facts that are not in the passages or a named web source.",
)
SEARCH_BOOK_INSTRUCTIONS = (
    SEARCH_INSTRUCTIONS[0],
    SEARCH_INSTRUCTIONS[1],
    SEARCH_INSTRUCTIONS[2],
    "Write a long response with headings. It may be as long as a book when the material supports that. "
    "Do not add facts that are not in the passages or a named web source.",
)
# Open mode: no passages. The model answers from what it knows.
OPEN_INSTRUCTIONS = (
    "Answer the task directly and completely.",
    "When you are not sure of a specific fact, date, figure, name, or quotation, say so instead of presenting a guess as fact.",
    "Be concise.",
)
OPEN_BOOK_INSTRUCTIONS = (
    "Answer the task directly and completely.",
    "When you are not sure of a specific fact, date, figure, name, or quotation, say so instead of presenting a guess as fact.",
    "Write a long response with headings. It may be as long as a book when the subject supports that.",
)
# A book-length completion. Providers that cap lower reject the request.
BOOK_MAX_OUTPUT_TOKENS = 100_000


def default_instructions(mode: str, response: str, search: bool = False) -> tuple[str, ...]:
    """The built-in instruction list for a mode, a response kind, and search."""

    book = response == "book"
    if mode == "open":
        return OPEN_BOOK_INSTRUCTIONS if book else OPEN_INSTRUCTIONS
    if search:
        return SEARCH_BOOK_INSTRUCTIONS if book else SEARCH_INSTRUCTIONS
    return BOOK_INSTRUCTIONS if book else DEFAULT_INSTRUCTIONS

T = TypeVar("T", bound=BaseModel)


class ModelClient(Protocol):
    """The only model surface the loop needs. Tests pass a scripted stand-in.

    ``parse`` is called with ``search=True`` only when the loop was built
    with ``search=True``; a client that does not support it never sees it.
    """

    def parse(self, *, system_text: str, user_text: str, schema: type[T]) -> T:
        """Return one structured result for this step."""


class Loop:
    """A knowledge base plus the loop that answers from it.

    Build it once and call :meth:`ask` for each task. Documents are read at
    construction, not on every question.

    ``mode`` is ``"knowledge"`` when there are passages and ``"open"`` when
    there are none. In open mode the drafter answers from the model's own
    knowledge, the audit checks accuracy instead of grounding, and the
    built-in instructions do not ask for passages or citations. ``search``
    turns on the provider's web search for the draft and the accuracy or
    grounding audit; it is available for xAI, OpenAI, and Anthropic.

    Args:
        knowledge: What to answer from. A directory or file path, a string of
            text, a ``{name: text}`` mapping, a list of any of those, or a
            prepared :class:`~loopkit.knowledge.Knowledge`.
        instructions: Rules the checker enforces on every draft. A string is
            split into one instruction per non-empty line; a sequence is used
            item by item. ``None`` selects :data:`DEFAULT_INSTRUCTIONS` (or
            :data:`BOOK_INSTRUCTIONS` when ``response="book"``).
        model: Model name. Defaults to ``LOOPKIT_MODEL`` or the provider's
            entry in :data:`~loopkit.client.DEFAULT_MODELS`.
        provider: ``"xai"``, ``"openai"``, ``"anthropic"``, or ``"compatible"``.
            Needed only when more than one API key is set; otherwise inferred
            from the environment (or ``LOOPKIT_PROVIDER``).
        base_url: An OpenAI-compatible server to call instead of a hosted
            provider (or ``LOOPKIT_BASE_URL``). Implies ``provider="compatible"``
            unless a provider is set explicitly.
        max_turns: Draft/check/revise rounds before the loop gives up and
            returns the latest draft with ``stopped_reason="max_turns"``.
        top_k: Passages kept by retrieval when the knowledge base exceeds
            ``context_budget``.
        context_budget: Character budget for passages sent to the model. A
            knowledge base that fits is sent whole, skipping retrieval;
            otherwise retrieval keeps the best matches up to this budget.
        api_key: Overrides the provider's environment variable. Never read
            from ``loopkit.toml``.
        client: Any object with a ``parse(system_text=, user_text=, schema=)``
            method (see :class:`ModelClient`). Replaces the built-in provider
            clients; tests use this to script the model.
        trace: Print each loop step to stderr. ``None`` reads ``LOOPKIT_TRACE``.
        check_mode: ``"each"`` judges one instruction per model call then runs
            a grounding check; ``"all"`` scores every instruction in one call.
        response: ``"normal"`` for a concise answer, ``"book"`` for a long-form
            answer with headings and a raised output-token cap.

    Raises:
        LoopError: On an unusable knowledge base, an invalid option, or a
            missing API key.
    """

    def __init__(
        self,
        knowledge: KnowledgeLike | None = None,
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
        debug: str | Path | bool | None = None,
        search: bool | None = None,
        auto_optimize: bool | None = None,
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
        self.mode: Mode = "open" if self.knowledge.is_empty else "knowledge"
        self.search = _env_flag("LOOPKIT_SEARCH") if search is None else bool(search)
        self.auto_optimize = (
            _env_flag("LOOPKIT_AUTO_OPTIMIZE", default=True) if auto_optimize is None else bool(auto_optimize)
        )
        if self.knowledge.is_empty:
            warnings.warn(self.knowledge.warnings[0], LoopWarning, stacklevel=2)
        if instructions is None:
            instructions = default_instructions(self.mode, self.response, self.search)
        self.instructions = normalize_instructions(instructions)
        self.max_turns = max_turns
        self.top_k = top_k
        self.context_budget = context_budget
        self.trace = _env_flag("LOOPKIT_TRACE") if trace is None else trace
        self._calls = 0
        self.debug_path: Path | None = resolve_debug(debug)
        if self.debug_path is not None:
            self.debug_path = enable_debug_log(self.debug_path)
        if client is not None:
            if self.search and not _accepts_search(client):
                raise LoopError(
                    "search=True needs a client whose parse() accepts a search keyword. "
                    "The built-in xai, openai, and anthropic clients do."
                )
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
                search=self.search,
            )
            self.provider = built.provider
            self.model = built.model
            self._client = built

    @classmethod
    def load(cls, path: str | Path | None = None, /, **overrides: object) -> Loop:
        """Build a loop from ``loopkit.toml``."""

        project = load_project(path, missing_ok=True)
        knowledge: KnowledgeLike | None = project.knowledge
        if knowledge is None:
            knowledge = Knowledge.empty(project.note)
        instructions = _coerce_instructions(overrides.pop("instructions", project.instructions))
        model = overrides.pop("model", project.model)
        provider = overrides.pop("provider", project.provider)
        base_url = overrides.pop("base_url", project.base_url)
        max_turns = overrides.pop("max_turns", project.max_turns)
        top_k = overrides.pop("top_k", project.top_k)
        check_mode = overrides.pop("check_mode", project.check_mode)
        response = overrides.pop("response", project.response)
        search = overrides.pop("search", project.search)
        if search is not None and not isinstance(search, bool):
            raise LoopError("search must be true or false.")
        auto_optimize = overrides.pop("auto_optimize", project.auto_optimize)
        if auto_optimize is not None and not isinstance(auto_optimize, bool):
            raise LoopError("auto_optimize must be true or false.")
        return cls(
            knowledge,
            instructions,
            model=None if model is None else str(model),
            provider=None if provider is None else str(provider),
            base_url=None if base_url is None else str(base_url),
            max_turns=_override_int(max_turns, DEFAULT_MAX_TURNS, "max_turns"),
            top_k=_override_int(top_k, DEFAULT_TOP_K, "top_k"),
            check_mode=None if check_mode is None else str(check_mode),
            response=None if response is None else str(response),
            search=search,
            auto_optimize=auto_optimize,
            **overrides,  # type: ignore[arg-type]
        )

    def ask(self, task: str) -> Answer:
        """Answer one task from the loaded knowledge base.

        Runs retrieval, then up to ``max_turns`` rounds of draft, check, and
        revise. Returns the latest draft even when the turn budget runs out;
        inspect :attr:`Answer.stopped_reason` and :attr:`Answer.checks` to see
        whether it passed.

        Raises:
            LoopError: When ``task`` is empty, or when a model call fails
                (the provider error is chained as ``__cause__``).
        """

        question = task.strip()
        if not question:
            raise LoopError("The task is empty.")

        retrieval = select(
            self.knowledge.passages,
            question,
            top_k=self.top_k,
            budget=self.context_budget,
        )
        if self.mode == "open":
            self._say("open mode: no passages; the model answers from its own knowledge" + (" and web search" if self.search else ""))
        else:
            self._say(
                f"{len(self.knowledge.passages)} passages loaded, {len(retrieval.passages)} sent to the model"
            )
            if retrieval.is_weak:
                self._say("no lexical overlap; the drafter is told not to stretch a passage")
        known = {passage.id: passage for passage in retrieval.passages}
        checks: list[Check] = []
        previous: Draft | None = None
        feedback: str | None = None
        self._calls = 0
        block(
            "ASK",
            ("TASK", question),
            ("INSTRUCTIONS", [f"{n}. {text}" for n, text in enumerate(self.instructions, start=1)]),
            (
                "RETRIEVAL",
                {
                    "passages_loaded": len(self.knowledge.passages),
                    "passages_sent": [
                        {"id": p.id, "source": p.source, "heading": p.heading, "chars": len(p.text)}
                        for p in retrieval.passages
                    ],
                    "weak_match": retrieval.is_weak,
                    "knowledge_warnings": list(self.knowledge.warnings),
                },
            ),
            meta={
                "provider": self.provider,
                "model": self.model,
                "mode": self.mode,
                "search": self.search,
                "auto_optimize": self.auto_optimize,
                "check_mode": self.check_mode,
                "response": self.response,
                "max_turns": self.max_turns,
            },
        )
        role = self._optimize(question, retrieval.passages) if self.auto_optimize else ""

        for turn in range(1, self.max_turns + 1):
            self._say(f"turn {turn}: draft")
            draft = self._step(
                stage=f"draft (turn {turn})",
                system_text=with_role(role, draft_system(self.mode, search=self.search)),
                user_text=draft_prompt(
                    instructions=self.instructions,
                    task=question,
                    passages=retrieval.passages,
                    is_weak=retrieval.is_weak,
                    previous=previous,
                    feedback=feedback,
                    response=self.response,
                    mode=self.mode,
                ),
                schema=Draft,
                search=self.search,
            )
            audit = self._audit(turn, question, retrieval.passages, draft, role)
            unknown = [cited for cited in citation_ids(draft) if cited not in known]
            problem, judged = _revision_problem(draft, audit, unknown, self.instructions, self.mode)
            checks.append(_record(turn, audit, problem, unknown, judged))
            missed = [str(item.number) for item in judged if not item.followed]
            if missed:
                self._say("instructions not followed: " + ", ".join(missed))
            if problem is None:
                decision = "passed"
            elif turn == self.max_turns:
                decision = "max_turns"
            else:
                decision = "revise"
            block(
                f"TURN {turn} DECISION: {decision}",
                ("DRAFT", draft.answer),
                (
                    "INSTRUCTION VERDICTS",
                    [
                        {"number": item.number, "followed": item.followed, "instruction": item.instruction, "note": item.note}
                        for item in judged
                    ],
                ),
                (
                    "GROUNDING",
                    {
                        "grounded": audit.grounded,
                        "revise": audit.revise,
                        "notes": audit.notes,
                        "gaps": list(audit.gaps),
                        "unknown_citations": unknown,
                    },
                ),
                ("REVISION NOTES SENT TO NEXT DRAFT", problem if decision == "revise" else "(none)"),
                meta={"turn": turn, "of": self.max_turns, "model_calls_so_far": self._calls},
            )
            if decision == "passed":
                self._say("stop: passed")
                answer = _finish(draft, known, checks, turn, "passed", self.response, self.mode, role)
                self._log_answer(answer)
                return answer
            if decision == "max_turns":
                self._say("stop: max turns")
                answer = _finish(draft, known, checks, turn, "max_turns", self.response, self.mode, role)
                self._log_answer(answer)
                return answer
            self._say("revise")
            previous = draft
            feedback = problem

        raise LoopError("The loop ended without an answer.")

    def _optimize(self, task: str, passages: Sequence[Passage]) -> str:
        """One call before the loop: the role the model should take for this task.

        A failure here is logged and the loop goes on without a role, so a
        quality step cannot by itself stop an answer.
        """

        self._say("optimize: writing the role for this task")
        try:
            result = self._step(
                stage="optimize",
                system_text=OPTIMIZE_SYSTEM,
                user_text=optimize_prompt(
                    task=task,
                    instructions=self.instructions,
                    passages=passages,
                    mode=self.mode,
                    response=self.response,
                ),
                schema=RolePrefix,
            )
        except LoopError as exc:
            logger.warning("optimize step failed, continuing without a role: %s", exc)
            self._say(f"optimize failed, continuing without a role: {exc}")
            return ""
        role = " ".join(result.role.split())
        if not role:
            self._say("optimize returned no role")
            return ""
        shown = role if len(role) <= 160 else role[:157] + "..."
        self._say(f"role: {shown}")
        return role

    def _log_answer(self, answer: Answer) -> None:
        block(
            "ANSWER",
            ("ROLE", answer.role or "(none)"),
            ("TEXT", answer.text),
            ("CITATIONS", [f"[{c.id}] {c.source}" + (f" — {c.heading}" if c.heading else "") for c in answer.citations]),
            meta={
                "mode": answer.mode,
                "stopped_reason": answer.stopped_reason,
                "turns": answer.turns,
                "model_calls": self._calls,
            },
        )

    def _audit(self, turn: int, task: str, passages: Sequence[Passage], draft: Draft, role: str = "") -> DraftCheck:
        """Check the draft.

        ``each`` asks about one instruction per call, then checks grounding
        once. ``all`` scores every instruction in a single call.
        """

        if self.check_mode == "all":
            self._say(f"turn {turn}: check")
            return self._step(
                stage=f"check (turn {turn})",
                system_text=with_role(role, check_system(self.mode, search=self.search)),
                user_text=check_prompt(
                    instructions=self.instructions,
                    task=task,
                    passages=passages,
                    draft=draft,
                    response=self.response,
                    mode=self.mode,
                ),
                schema=DraftCheck,
                search=self.search,
            )

        verdicts: list[InstructionVerdict] = []
        for number, instruction in enumerate(self.instructions, start=1):
            self._say(f"turn {turn}: check instruction {number}")
            one = self._step(
                stage=f"check instruction {number} (turn {turn})",
                system_text=with_role(role, check_one_system(self.mode)),
                user_text=check_one_prompt(
                    number=number,
                    instruction=instruction,
                    task=task,
                    passages=passages,
                    draft=draft,
                    mode=self.mode,
                ),
                schema=OneCheck,
            )
            verdicts.append(InstructionVerdict(number=number, followed=one.followed, note=one.note))
        review = "accuracy" if self.mode == "open" else "grounding"
        self._say(f"turn {turn}: check {review}")
        ground = self._step(
            stage=f"check {review} (turn {turn})",
            system_text=with_role(role, review_system(self.mode, search=self.search)),
            user_text=ground_prompt(task=task, passages=passages, draft=draft, mode=self.mode),
            schema=GroundingCheck,
            search=self.search,
        )
        return DraftCheck(
            instructions=verdicts,
            grounded=ground.grounded,
            revise=ground.revise,
            notes=ground.notes,
            gaps=list(ground.gaps),
        )

    def _step(
        self,
        *,
        stage: str,
        system_text: str,
        user_text: str,
        schema: type[T],
        search: bool = False,
    ) -> T:
        self._calls += 1
        number = self._calls
        block(
            f"CALL {number}: {stage}",
            ("SYSTEM PROMPT", system_text),
            ("USER PROMPT", user_text),
            meta={"schema": schema.__name__, "provider": self.provider, "model": self.model, "search": search},
        )
        started = time.monotonic()
        try:
            if search:
                parsed = self._client.parse(  # type: ignore[call-arg]
                    system_text=system_text, user_text=user_text, schema=schema, search=True
                )
            else:
                parsed = self._client.parse(system_text=system_text, user_text=user_text, schema=schema)
        except Exception as exc:
            self._log_result(number, stage, schema, started, parsed=None, error=exc)
            if isinstance(exc, LoopError):
                raise
            raise LoopError(f"The model call failed during {stage}: {exc}") from exc
        if not isinstance(parsed, schema):
            error = LoopError(
                f"The model returned {type(parsed).__name__} during {stage}, expected {schema.__name__}."
            )
            self._log_result(number, stage, schema, started, parsed=parsed, error=error)
            raise error
        self._log_result(number, stage, schema, started, parsed=parsed, error=None)
        return parsed

    def _log_result(
        self,
        number: int,
        stage: str,
        schema: type[object],
        started: float,
        *,
        parsed: object,
        error: Exception | None,
    ) -> None:
        if not logger.isEnabledFor(10):  # logging.DEBUG
            return
        last = getattr(self._client, "last_call", None)
        record = last if isinstance(last, dict) else {}
        sections: list[tuple[str, object]] = []
        if record.get("request") is not None:
            request = dict(cast(Mapping[str, object], record["request"]))
            # The prompts are already in the CALL block, and the JSON schema is
            # the same on every call of a stage. Keep the rest of the request.
            for key in ("input", "messages", "system", "instructions"):
                request.pop(key, None)
            sections.append(("REQUEST PARAMETERS", _compact_schema(request, schema.__name__)))
        sections.append(("RAW MODEL TEXT", record.get("raw_text")))
        if record.get("response") is not None:
            sections.append(("RAW RESPONSE BODY", record["response"]))
        if error is None:
            sections.append(("PARSED", parsed))
        else:
            sections.append(("ERROR", f"{type(error).__name__}: {error}"))
        block(
            f"RESULT {number}: {stage} -> {'error' if error else 'ok'}",
            *sections,
            meta={"schema": schema.__name__, "seconds": time.monotonic() - started},
        )

    def _say(self, message: str) -> None:
        if self.trace:
            print(f"loopkit: {message}", file=sys.stderr, flush=True)


def ask(
    task: str,
    knowledge: KnowledgeLike | None = None,
    instructions: str | Sequence[str] | None = None,
    **kwargs: object,
) -> Answer:
    """Answer one task with a throwaway :class:`Loop`.

    With no ``knowledge`` argument the project file ``loopkit.toml`` is read.
    Extra keyword arguments are passed to :class:`Loop`. For more than one
    question, build a :class:`Loop` once and call :meth:`Loop.ask` instead, so
    documents are read only once.
    """

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
    if isinstance(value, Sequence):
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


def _accepts_search(client: object) -> bool:
    import inspect

    parse = getattr(client, "parse", None)
    if parse is None:
        return False
    try:
        parameters = inspect.signature(parse).parameters
    except (TypeError, ValueError):
        return True
    return "search" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


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
    mode: str = "knowledge",
) -> tuple[str | None, tuple[InstructionCheck, ...]]:
    judged, instruction_problems = _judge_instructions(instructions, audit.instructions)
    problems: list[str] = []
    if not draft.answer.strip():
        problems.append("The draft was empty. Write the answer the user should read.")
    if unknown:
        listed = ", ".join(unknown)
        if mode == "open":
            problems.append(
                f"Remove the bracketed reference markers ({listed}). They do not point to anything the reader can see."
            )
        else:
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
                if mode != "open"
                else "The draft does not follow the instructions or contains claims the check believes are wrong."
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
        found = by_number.get(number)
        if found is not None and found.followed:
            judged.append(
                InstructionCheck(number=number, instruction=instruction, followed=True, note="")
            )
            continue
        if found is None:
            note = "The check did not judge this instruction."
            problems.append(f"Instruction {number} was not judged: {instruction}")
        else:
            note = found.note.strip()
            if note:
                problems.append(f"Instruction {number} was not followed: {instruction}\n{note}")
            else:
                problems.append(f"Instruction {number} was not followed: {instruction}")
        judged.append(
            InstructionCheck(number=number, instruction=instruction, followed=False, note=note)
        )
    return tuple(judged), problems


def _compact_schema(node: object, name: str) -> object:
    """Replace any nested ``schema`` object with a one-line stand-in."""

    if isinstance(node, dict):
        out: dict[str, object] = {}
        for key, value in node.items():
            if key == "schema" and isinstance(value, dict):
                count = len(value.get("properties", {})) if isinstance(value.get("properties"), dict) else 0
                out[key] = f"<JSON schema for {name}: {count} required properties, additionalProperties=false>"
            else:
                out[key] = _compact_schema(value, name)
        return out
    if isinstance(node, list):
        return [_compact_schema(item, name) for item in node]
    return node


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
    mode: str = "knowledge",
    role: str = "",
) -> Answer:
    citations: list[Citation] = []
    for cited in citation_ids(draft):
        passage = known.get(cited)
        if passage is None:
            continue
        citations.append(
            Citation(
                id=passage.id, source=passage.source, heading=passage.heading, text=passage.text
            )
        )
    return Answer(
        text=draft.answer.strip(),
        response=response,
        mode=cast(Literal["knowledge", "open"], mode),
        role=role,
        citations=tuple(citations),
        checks=tuple(checks),
        turns=turns,
        stopped_reason=reason,
    )
