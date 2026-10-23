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

from loopkit.client import DEFAULT_MODELS, PROVIDER_NAMES, make_client
from loopkit.config import CHECK_MODES, OUTPUT_FORMATS, RESPONSES, TOKEN_BUDGET_MODES, load_project
from loopkit.debuglog import block, enable_debug_log, logger, resolve_debug
from loopkit.errors import LoopError, LoopWarning
from loopkit.estimate import (
    AskShape,
    EstimationStrategy,
    coerce_strategy,
    length_compliant,
    max_response_chars,
    response_length_target,
    words_for,
)
from loopkit.knowledge import Knowledge, KnowledgeLike, Passage
from loopkit.models import (
    Answer,
    Check,
    Citation,
    InstructionCheck,
    LengthCheck,
    OutputFormat,
    ResponseKind,
    StoppedReason,
    StructureCheck,
    TokenUsage,
)
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
from loopkit.structure import check_structure, coerce_schema, render_schema
from loopkit.schemas import Draft, DraftCheck, GroundingCheck, InstructionVerdict, OneCheck, RolePrefix
from loopkit.tokens import TokenEstimator, TokenMeter, Usage, estimate_tokens, usage_from_response

CheckMode = Literal["each", "all"]
Mode = Literal["knowledge", "open"]
TokenBudgetMode = Literal["soft", "hard"]
ResponseLength = int | Literal["auto"] | None

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
# The note on an instruction the token budget did not leave room to check.
NOT_CHECKED_NOTE = "Not checked: the token budget ran out."


# With JSON output the passages are named in cited_ids, not marked inline.
JSON_CITE_INSTRUCTION = "Name every passage you used by id (p1, p2, and so on) in cited_ids."


def default_instructions(
    mode: str, response: str, search: bool = False, output: str = "text"
) -> tuple[str, ...]:
    """The built-in instruction list for a mode, a response kind, search, and output format."""

    book = response == "book"
    if mode == "open":
        return OPEN_BOOK_INSTRUCTIONS if book else OPEN_INSTRUCTIONS
    if search:
        chosen = SEARCH_BOOK_INSTRUCTIONS if book else SEARCH_INSTRUCTIONS
    else:
        chosen = BOOK_INSTRUCTIONS if book else DEFAULT_INSTRUCTIONS
    if output == "json":
        return tuple(
            JSON_CITE_INSTRUCTION if item == DEFAULT_INSTRUCTIONS[1] else item for item in chosen
        )
    return chosen

T = TypeVar("T", bound=BaseModel)


class ModelClient(Protocol):
    """The only model surface the loop needs. Tests pass a scripted stand-in.

    ``parse`` is called with ``search=True`` only when the loop was built
    with ``search=True``; a client that does not support it never sees it.

    A client may also keep a ``last_call`` dict. When it holds a ``usage``
    entry (a :class:`~loopkit.tokens.Usage`, or a dict with the provider's
    ``input_tokens``/``output_tokens`` or ``prompt_tokens``/``completion_tokens``),
    the loop counts that against the token budget instead of estimating.
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
        check_model: A second model for the checks: every instruction check,
            the combined check, and the grounding or accuracy audit. The
            optimize step and the drafts stay on ``model``. A smaller or
            cheaper model often judges one instruction well; a different
            one judges with different blind spots. Defaults to
            ``LOOPKIT_CHECK_MODEL``; unset means the checks use the same
            client as the drafts. On its own it means the same provider,
            key, and server as ``model``.
        check_provider: The provider for the check model, when it is not
            the draft's (``LOOPKIT_CHECK_PROVIDER``). Its key comes from
            that provider's environment variable, and its model is the
            provider's default unless ``check_model`` says otherwise.
        check_base_url: An OpenAI-compatible server for the checks
            (``LOOPKIT_CHECK_BASE_URL``); needs ``check_model``.
        check_api_key: The key for the check provider, when it is not in
            the environment.
        check_client: A client object for the checks, like ``client``.
        trace: Print each loop step to stderr. ``None`` reads ``LOOPKIT_TRACE``.
        check_mode: ``"each"`` judges one instruction per model call then runs
            a grounding check; ``"all"`` scores every instruction in one call.
        response: ``"normal"`` for a concise answer, ``"book"`` for a long-form
            answer with headings and a raised output-token cap.
        token_budget: A limit on the tokens one :meth:`ask` may spend across
            all of its model calls, input and output together. ``None``
            reads ``LOOPKIT_TOKEN_BUDGET``; unset means no limit. When it
            runs out the loop stops and returns what it has with
            ``stopped_reason="token_budget"``. A call is never cut short.
        token_budget_mode: How the limit is held. ``"soft"`` (the default,
            or ``LOOPKIT_TOKEN_BUDGET_MODE``) works a turn at a time: before
            a turn the loop estimates the whole turn, draft and checks, and
            does not start it when it would not fit; a turn that has
            started runs to the end, and the first turn always runs, so the
            answer is always a fully checked draft and the total can end
            over the limit by up to one turn. ``"hard"`` works a call at a
            time: every call, the first draft included, is estimated first
            and not made when it would not fit, and the loop stops the
            moment the total is over the limit. The answer is then the
            draft as it stands, with the verdicts the check had reached, or
            no draft at all when even the first one would not fit.
        token_estimator: A function from text to a token count, used for
            the estimate before a call and for calls whose client does not
            report usage. Defaults to :func:`~loopkit.tokens.estimate_tokens`,
            a dependency-free approximation; pass a real tokenizer for
            tighter accounting.
        estimation_strategy: The :class:`~loopkit.estimate.EstimationStrategy`
            that sizes a reply before the call is made. The one strategy
            today, ``"last_turn"``, sizes the first reply of each kind from
            the input and every later one like the latest of that kind.
        response_length: Control the length of the answer. ``"auto"`` asks
            each draft to keep under the length the estimate step expects
            for this task; an integer asks for that many characters. Either
            is held under what the model can return in one call (see
            :func:`~loopkit.estimate.max_response_chars`). The length is
            measured in code after every draft; a draft over it is sent
            back with a note, like a missed instruction. ``None`` reads
            ``LOOPKIT_RESPONSE_LENGTH``; unset leaves the length to the
            instructions.
        output_format: ``"text"`` (the default) for prose, ``"json"`` for
            one JSON document held to ``output_schema``. With ``"json"`` the
            drafter is shown the schema, every draft is parsed and validated
            in code, and a draft that is not the document the schema asks
            for is sent back with the exact problems; the built-in citation
            instruction then asks for ids in ``cited_ids`` rather than
            inline. The answer's ``text`` is the document and ``data`` is it
            parsed.
        output_schema: The shape for ``output_format="json"``: a JSON Schema
            object, an example of the output (a schema is inferred from it:
            every key required, no others allowed, types from the values),
            or either as JSON text. Giving it implies ``output_format="json"``.

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
        check_model: str | None = None,
        check_provider: str | None = None,
        check_base_url: str | None = None,
        check_api_key: str | None = None,
        check_client: ModelClient | None = None,
        trace: bool | None = None,
        check_mode: str | None = None,
        response: str | None = None,
        debug: str | Path | bool | None = None,
        search: bool | None = None,
        auto_optimize: bool | None = None,
        token_budget: int | None = None,
        token_budget_mode: str | None = None,
        token_estimator: TokenEstimator | None = None,
        estimation_strategy: EstimationStrategy | str | None = None,
        response_length: int | str | None = None,
        output_format: str | None = None,
        output_schema: Mapping[str, object] | Sequence[object] | str | None = None,
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
            instructions = default_instructions(
                self.mode,
                self.response,
                self.search,
                "json" if output_format == "json" or output_schema is not None else "text",
            )
        self.instructions = normalize_instructions(instructions)
        self.max_turns = max_turns
        self.top_k = top_k
        self.context_budget = context_budget
        self.trace = _env_flag("LOOPKIT_TRACE") if trace is None else trace
        self.token_budget = (
            _env_int("LOOPKIT_TOKEN_BUDGET") if token_budget is None else _budget(token_budget)
        )
        if token_budget_mode is None:
            token_budget_mode = os.environ.get("LOOPKIT_TOKEN_BUDGET_MODE", "").strip() or None
        self.token_budget_mode = cast(
            TokenBudgetMode,
            _choice(token_budget_mode, "soft", "token_budget_mode", TOKEN_BUDGET_MODES),
        )
        if token_estimator is not None and not callable(token_estimator):
            raise LoopError("token_estimator must be a function from text to a token count.")
        self.token_estimator: TokenEstimator = (
            estimate_tokens if token_estimator is None else token_estimator
        )
        try:
            self.estimation_strategy = coerce_strategy(estimation_strategy)
        except ValueError as exc:
            raise LoopError(str(exc)) from None
        if response_length is None:
            response_length = os.environ.get("LOOPKIT_RESPONSE_LENGTH", "").strip() or None
        self.response_length: ResponseLength = _response_length(response_length)
        if output_format is None and output_schema is not None:
            output_format = "json"
        self.output_format = cast(
            OutputFormat, _choice(output_format, "text", "output_format", OUTPUT_FORMATS)
        )
        self.output_schema: dict[str, object] | None = None
        if self.output_format == "json":
            if output_schema is None:
                raise LoopError(
                    'output_format="json" needs output_schema: a JSON Schema, or an example of the output.'
                )
            self.output_schema = coerce_schema(output_schema)
        elif output_schema is not None:
            raise LoopError('output_schema needs output_format="json".')
        self._calls = 0
        self._meter = TokenMeter(None, estimator=self.token_estimator)
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
        self._check_client: ModelClient
        self.check_provider: str
        self.check_model: str
        self._check_client, self.check_provider, self.check_model = self._build_check_client(
            model=check_model,
            provider=check_provider,
            base_url=check_base_url,
            api_key=check_api_key,
            client=check_client,
            main_api_key=api_key,
            main_base_url=base_url,
        )

    def _build_check_client(
        self,
        *,
        model: str | None,
        provider: str | None,
        base_url: str | None,
        api_key: str | None,
        client: ModelClient | None,
        main_api_key: str | None,
        main_base_url: str | None,
    ) -> tuple[ModelClient, str, str]:
        """The client for the checks: the draft's unless a check_* option says otherwise."""

        if client is not None:
            if self.search and not _accepts_search(client):
                raise LoopError(
                    "search=True needs a check_client whose parse() accepts a search keyword."
                )
            name = (model or os.environ.get("LOOPKIT_CHECK_MODEL", "")).strip() or self.model
            return client, "custom", name
        model = (model or os.environ.get("LOOPKIT_CHECK_MODEL", "")).strip() or None
        provider = (provider or os.environ.get("LOOPKIT_CHECK_PROVIDER", "")).strip() or None
        base_url = (base_url or os.environ.get("LOOPKIT_CHECK_BASE_URL", "")).strip() or None
        if model is None and provider is None and base_url is None and api_key is None:
            return self._client, self.provider, self.model
        if provider is None and base_url is not None:
            provider = "compatible"
        if provider is None and self.provider in PROVIDER_NAMES:
            provider = self.provider
        same = provider is not None and provider == self.provider
        if same and base_url is None:
            base_url = main_base_url
        if same and api_key is None:
            api_key = main_api_key
        if model is None:
            # LOOPKIT_MODEL is the draft's. The check model is the draft's on
            # the same provider, else that provider's default.
            model = self.model if same else DEFAULT_MODELS.get(provider or "")
            if not model:
                raise LoopError(
                    "check_model is needed with check_base_url or check_provider='compatible': "
                    "the server has no default model."
                )
        built = make_client(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            search=self.search,
        )
        # The built clients accept a search keyword the protocol does not name.
        return cast(ModelClient, built), built.provider, built.model

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
        check_model = overrides.pop("check_model", project.check_model)
        check_provider = overrides.pop("check_provider", project.check_provider)
        check_base_url = overrides.pop("check_base_url", project.check_base_url)
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
        token_budget = _budget(overrides.pop("token_budget", project.token_budget))
        token_budget_mode = overrides.pop("token_budget_mode", project.token_budget_mode)
        response_length = overrides.pop("response_length", project.response_length)
        output_format = overrides.pop("output_format", project.output_format)
        output_schema = overrides.pop("output_schema", project.output_schema)
        return cls(
            knowledge,
            instructions,
            model=None if model is None else str(model),
            provider=None if provider is None else str(provider),
            base_url=None if base_url is None else str(base_url),
            check_model=None if check_model is None else str(check_model),
            check_provider=None if check_provider is None else str(check_provider),
            check_base_url=None if check_base_url is None else str(check_base_url),
            max_turns=_override_int(max_turns, DEFAULT_MAX_TURNS, "max_turns"),
            top_k=_override_int(top_k, DEFAULT_TOP_K, "top_k"),
            check_mode=None if check_mode is None else str(check_mode),
            response=None if response is None else str(response),
            search=search,
            auto_optimize=auto_optimize,
            token_budget=token_budget,
            token_budget_mode=None if token_budget_mode is None else str(token_budget_mode),
            response_length=_response_length(response_length),
            output_format=None if output_format is None else str(output_format),
            output_schema=output_schema,  # type: ignore[arg-type]
            **overrides,  # type: ignore[arg-type]
        )

    def ask(self, task: str, *, token_budget: int | None = None) -> Answer:
        """Answer one task from the loaded knowledge base.

        Runs retrieval, then up to ``max_turns`` rounds of draft, check, and
        revise. Returns the latest draft even when the turn budget runs out;
        inspect :attr:`Answer.stopped_reason` and :attr:`Answer.checks` to see
        whether it passed.

        ``token_budget`` is a limit on the tokens this one call may spend,
        overriding the loop's ``token_budget`` for this task only and held
        the way ``token_budget_mode`` says. When it runs out the answer is
        what the loop has with ``stopped_reason="token_budget"``;
        :attr:`Answer.tokens` says what was spent.

        Raises:
            LoopError: When ``task`` is empty, or when a model call fails
                (the provider error is chained as ``__cause__``).
        """

        question = task.strip()
        if not question:
            raise LoopError("The task is empty.")
        limit = self.token_budget if token_budget is None else _budget(token_budget)
        hard = self.token_budget_mode == "hard"

        retrieval = select(
            self.knowledge.passages,
            question,
            top_k=self.top_k,
            budget=self.context_budget,
        )
        # The estimate step: size the task before any call is made. The meter
        # uses this shape for the first reply of each kind, then what it sees.
        shape = AskShape(
            task_tokens=self.token_estimator(question),
            instruction_count=len(self.instructions),
            passage_tokens=sum(self.token_estimator(p.text) for p in retrieval.passages),
            response=self.response,
            mode=self.mode,
            response_chars=self.response_length if isinstance(self.response_length, int) else None,
        )
        meter = TokenMeter(
            limit, estimator=self.token_estimator, strategy=self.estimation_strategy, shape=shape
        )
        self._meter = meter
        length_limit: int | None = None
        model_max = max_response_chars(self.model, getattr(self._client, "max_output_tokens", None))
        if self.response_length is not None:
            length_limit = response_length_target(shape, model_max)
            if isinstance(self.response_length, int) and self.response_length > length_limit:
                self._say(
                    f"response length {self.response_length:,} is over what {self.model} returns in "
                    f"one call; held to {length_limit:,} characters"
                )
        if self.mode == "open":
            self._say("open mode: no passages; the model answers from its own knowledge" + (" and web search" if self.search else ""))
        else:
            self._say(
                f"{len(self.knowledge.passages)} passages loaded, {len(retrieval.passages)} sent to the model"
            )
            if retrieval.is_weak:
                self._say("no lexical overlap; the drafter is told not to stretch a passage")
        if self._check_client is not self._client:
            self._say(
                f"models: {self.provider} {self.model} drafts; "
                f"{self.check_provider} {self.check_model} checks"
            )
        if limit is not None:
            self._say(
                f"token budget: {limit:,} ({self.token_budget_mode}); "
                f"first draft reply estimated at {meter.expected_output('Draft'):,} tokens"
            )
        if length_limit is not None:
            self._say(
                f"response length: under {length_limit:,} characters, about {words_for(length_limit):,} words"
            )
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
                "check_provider": self.check_provider,
                "check_model": self.check_model,
                "mode": self.mode,
                "search": self.search,
                "auto_optimize": self.auto_optimize,
                "check_mode": self.check_mode,
                "response": self.response,
                "max_turns": self.max_turns,
                "token_budget": limit,
                "token_budget_mode": self.token_budget_mode,
                "estimation_strategy": self.estimation_strategy.value,
                "shape": {
                    "task_tokens": shape.task_tokens,
                    "instruction_count": shape.instruction_count,
                    "passage_tokens": shape.passage_tokens,
                    "first_draft_reply": meter.expected_output("Draft"),
                },
                "response_length": length_limit,
                "model_max_response_chars": model_max,
                "output_format": self.output_format,
            },
        )
        role = self._optimize(question, retrieval.passages) if self.auto_optimize else ""
        draft_system_text = with_role(
            role,
            draft_system(
                self.mode,
                search=self.search,
                length=None if length_limit is None else (length_limit, words_for(length_limit)),
                output_schema=None
                if self.output_schema is None
                else render_schema(self.output_schema),
            ),
        )

        def draft_user_text(previous: Draft | None, feedback: str | None) -> str:
            return draft_prompt(
                instructions=self.instructions,
                task=question,
                passages=retrieval.passages,
                is_weak=retrieval.is_weak,
                previous=previous,
                feedback=feedback,
                response=self.response,
                mode=self.mode,
            )

        def turn_fits(previous: Draft | None, feedback: str | None) -> bool:
            """Soft mode looks a whole turn ahead; hard mode one call, the draft."""

            user_text = draft_user_text(previous, feedback)
            if hard:
                return self._fits(draft_system_text, user_text, Draft)
            cost = self._turn_cost(draft_system_text, user_text, role, question, retrieval.passages)
            return meter.fits(cost)

        for turn in range(1, self.max_turns + 1):
            user_text = draft_user_text(previous, feedback)
            if turn == 1 and limit is not None and not turn_fits(previous, feedback):
                if hard:
                    self._say("the first draft would not fit the token budget; nothing was drafted")
                    self._say(_STOP_MESSAGES["token_budget"])
                    self._say(f"tokens: {meter.describe()}")
                    answer = _finish(
                        None,
                        known,
                        checks,
                        0,
                        "token_budget",
                        self.response,
                        self.mode,
                        role,
                        _usage(meter),
                        length_limit,
                        self.output_format,
                    )
                    self._log_answer(answer)
                    return answer
                cost = self._turn_cost(
                    draft_system_text, user_text, role, question, retrieval.passages
                )
                self._say(
                    f"the first turn is estimated at {cost:,} tokens, over the budget of {limit:,}; "
                    "it runs anyway so there is a checked answer to return"
                )
            self._say(f"turn {turn}: draft")
            draft = self._step(
                stage=f"draft (turn {turn})",
                system_text=draft_system_text,
                user_text=user_text,
                schema=Draft,
                search=self.search,
            )
            length = _measure(draft, length_limit)
            if length is not None:
                verdict = "over the limit" if not length.followed else "within the limit"
                self._say(
                    f"turn {turn}: response length {length.actual:,} of {length.limit:,} characters, {verdict}"
                )
            data, structure = self._structure(draft)
            if structure is not None:
                if structure.followed:
                    self._say(f"turn {turn}: output follows the schema")
                else:
                    count = len(structure.problems)
                    self._say(
                        f"turn {turn}: output does not follow the schema: "
                        f"{count} problem{'s' if count != 1 else ''}"
                    )
            audit, complete = self._audit(turn, question, retrieval.passages, draft, role)
            unknown = [cited for cited in citation_ids(draft) if cited not in known]
            problem, judged = _revision_problem(
                draft,
                audit,
                unknown,
                self.instructions,
                self.mode,
                missing_note=None if complete else NOT_CHECKED_NOTE,
                length=length,
                structure=structure,
            )
            checks.append(_record(turn, audit, problem, unknown, judged, length, structure))
            unchecked = [str(item.number) for item in judged if item.note == NOT_CHECKED_NOTE]
            missed = [
                str(item.number)
                for item in judged
                if not item.followed and item.note != NOT_CHECKED_NOTE
            ]
            if missed:
                self._say("instructions not followed: " + ", ".join(missed))
            if unchecked:
                self._say("instructions not checked: " + ", ".join(unchecked))
            decision: StoppedReason | Literal["revise"]
            if not complete:
                decision = "token_budget"
            elif problem is None:
                decision = "passed"
            elif turn == self.max_turns:
                decision = "max_turns"
            elif not turn_fits(draft, problem):
                self._say(
                    f"revision skipped: it would not fit the token budget ({meter.describe()})"
                )
                decision = "token_budget"
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
                (
                    "RESPONSE LENGTH",
                    "(not controlled)"
                    if length is None
                    else {
                        "limit": length.limit,
                        "actual": length.actual,
                        "followed": length.followed,
                    },
                ),
                (
                    "OUTPUT STRUCTURE",
                    "(text)"
                    if structure is None
                    else {"followed": structure.followed, "problems": list(structure.problems)},
                ),
                ("REVISION NOTES SENT TO NEXT DRAFT", problem if decision == "revise" else "(none)"),
                meta={
                    "turn": turn,
                    "of": self.max_turns,
                    "model_calls_so_far": self._calls,
                    "tokens_so_far": meter.used,
                    "token_budget": limit,
                },
            )
            if decision != "revise":
                self._say(_STOP_MESSAGES[decision])
                if limit is not None:
                    self._say(f"tokens: {meter.describe()}")
                answer = _finish(
                    draft,
                    known,
                    checks,
                    turn,
                    decision,
                    self.response,
                    self.mode,
                    role,
                    _usage(meter),
                    length_limit,
                    self.output_format,
                    data,
                )
                self._log_answer(answer)
                return answer
            self._say("revise")
            previous = draft
            feedback = problem

        raise LoopError("The loop ended without an answer.")

    def _structure(self, draft: Draft) -> tuple[object, StructureCheck | None]:
        """Parse and validate a JSON answer in code; ``(None, None)`` for text output."""

        if self.output_schema is None:
            return None, None
        data, problems = check_structure(draft.answer, self.output_schema)
        return data, StructureCheck(followed=not problems, problems=tuple(problems))

    def _fits(self, system_text: str, user_text: str, schema: type[BaseModel]) -> bool:
        """Whether a call with these prompts is expected to stay inside the token budget."""

        return self._meter.fits(self._meter.estimate_call(system_text, user_text, schema.__name__))

    def _turn_cost(
        self,
        draft_system_text: str,
        draft_user_text: str,
        role: str,
        task: str,
        passages: Sequence[Passage],
    ) -> int:
        """The estimated tokens of one whole turn: the draft, then every check of it.

        The checks read the draft, which does not exist yet; its expected
        size stands in for it in their prompts.
        """

        meter = self._meter
        draft_tokens = meter.expected_output(Draft.__name__)
        cost = meter.estimate_call(draft_system_text, draft_user_text, Draft.__name__)
        placeholder = Draft(answer="", cited_ids=[])
        if self.check_mode == "all":
            return cost + meter.estimate_call(
                with_role(
                    role, check_system(self.mode, search=self.search, output=self.output_format)
                ),
                check_prompt(
                    instructions=self.instructions,
                    task=task,
                    passages=passages,
                    draft=placeholder,
                    response=self.response,
                    mode=self.mode,
                ),
                DraftCheck.__name__,
                extra_prompt_tokens=draft_tokens,
            )
        for number, instruction in enumerate(self.instructions, start=1):
            cost += meter.estimate_call(
                with_role(role, check_one_system(self.mode, output=self.output_format)),
                check_one_prompt(
                    number=number,
                    instruction=instruction,
                    task=task,
                    passages=passages,
                    draft=placeholder,
                    mode=self.mode,
                ),
                OneCheck.__name__,
                extra_prompt_tokens=draft_tokens,
            )
        return cost + meter.estimate_call(
            with_role(
                role, review_system(self.mode, search=self.search, output=self.output_format)
            ),
            ground_prompt(task=task, passages=passages, draft=placeholder, mode=self.mode),
            GroundingCheck.__name__,
            extra_prompt_tokens=draft_tokens,
        )

    def _optimize(self, task: str, passages: Sequence[Passage]) -> str:
        """One call before the loop: the role the model should take for this task.

        A failure here is logged and the loop goes on without a role, so a
        quality step cannot by itself stop an answer.
        """

        user_text = optimize_prompt(
            task=task,
            instructions=self.instructions,
            passages=passages,
            mode=self.mode,
            response=self.response,
        )
        if not self._fits(OPTIMIZE_SYSTEM, user_text, RolePrefix):
            self._say("optimize skipped: it would not fit the token budget")
            return ""
        self._say("optimize: writing the role for this task")
        try:
            result = self._step(
                stage="optimize",
                system_text=OPTIMIZE_SYSTEM,
                user_text=user_text,
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
                "tokens": _tokens_meta(answer.tokens),
                "response_length": answer.response_length,
                "output_format": answer.output_format,
            },
        )

    def _audit(
        self, turn: int, task: str, passages: Sequence[Passage], draft: Draft, role: str = ""
    ) -> tuple[DraftCheck, bool]:
        """Check the draft.

        ``each`` asks about one instruction per call, then checks grounding
        once. ``all`` scores every instruction in a single call.

        The second value is false when a hard token budget stopped the check
        part way. The verdicts gathered so far are returned; the rest are
        missing, and ``grounded`` is false because it was not checked. In
        soft mode a check that has started always runs to the end.
        """

        review = "accuracy" if self.mode == "open" else "grounding"
        guarded = self.token_budget_mode == "hard"
        if self.check_mode == "all":
            system_text = with_role(
                role, check_system(self.mode, search=self.search, output=self.output_format)
            )
            user_text = check_prompt(
                instructions=self.instructions,
                task=task,
                passages=passages,
                draft=draft,
                response=self.response,
                mode=self.mode,
            )
            if guarded and not self._fits(system_text, user_text, DraftCheck):
                self._say(f"turn {turn}: check skipped: token budget ({self._meter.describe()})")
                return _unchecked([], "The token budget ran out before the check."), False
            self._say(f"turn {turn}: check")
            audit = self._step(
                stage=f"check (turn {turn})",
                system_text=system_text,
                user_text=user_text,
                schema=DraftCheck,
                search=self.search,
                client=self._check_client,
            )
            return audit, True

        verdicts: list[InstructionVerdict] = []
        for number, instruction in enumerate(self.instructions, start=1):
            system_text = with_role(role, check_one_system(self.mode, output=self.output_format))
            user_text = check_one_prompt(
                number=number,
                instruction=instruction,
                task=task,
                passages=passages,
                draft=draft,
                mode=self.mode,
            )
            if guarded and not self._fits(system_text, user_text, OneCheck):
                self._say(
                    f"turn {turn}: check instruction {number} skipped: token budget "
                    f"({self._meter.describe()})"
                )
                return _unchecked(
                    verdicts,
                    f"The token budget ran out before the check of instruction {number}; "
                    f"{review} was not checked either.",
                ), False
            self._say(f"turn {turn}: check instruction {number}")
            one = self._step(
                stage=f"check instruction {number} (turn {turn})",
                system_text=system_text,
                user_text=user_text,
                schema=OneCheck,
                client=self._check_client,
            )
            verdicts.append(InstructionVerdict(number=number, followed=one.followed, note=one.note))
        system_text = with_role(
            role, review_system(self.mode, search=self.search, output=self.output_format)
        )
        user_text = ground_prompt(task=task, passages=passages, draft=draft, mode=self.mode)
        if guarded and not self._fits(system_text, user_text, GroundingCheck):
            self._say(
                f"turn {turn}: check {review} skipped: token budget ({self._meter.describe()})"
            )
            return _unchecked(
                verdicts, f"The token budget ran out before the {review} check."
            ), False
        self._say(f"turn {turn}: check {review}")
        ground = self._step(
            stage=f"check {review} (turn {turn})",
            system_text=system_text,
            user_text=user_text,
            schema=GroundingCheck,
            search=self.search,
            client=self._check_client,
        )
        audit = DraftCheck(
            instructions=verdicts,
            grounded=ground.grounded,
            revise=ground.revise,
            notes=ground.notes,
            gaps=list(ground.gaps),
        )
        return audit, True

    def _step(
        self,
        *,
        stage: str,
        system_text: str,
        user_text: str,
        schema: type[T],
        search: bool = False,
        client: ModelClient | None = None,
    ) -> T:
        """One model call. ``client`` is the check client for the checks, else the draft's."""

        if client is None:
            client = self._client
        checking = client is self._check_client and client is not self._client
        provider = self.check_provider if checking else self.provider
        model = self.check_model if checking else self.model
        self._calls += 1
        number = self._calls
        block(
            f"CALL {number}: {stage}",
            ("SYSTEM PROMPT", system_text),
            ("USER PROMPT", user_text),
            meta={"schema": schema.__name__, "provider": provider, "model": model, "search": search},
        )
        started = time.monotonic()
        try:
            if search:
                parsed = client.parse(  # type: ignore[call-arg]
                    system_text=system_text, user_text=user_text, schema=schema, search=True
                )
            else:
                parsed = client.parse(system_text=system_text, user_text=user_text, schema=schema)
        except Exception as exc:
            self._log_result(number, stage, schema, started, parsed=None, error=exc, client=client)
            if isinstance(exc, LoopError):
                raise
            raise LoopError(f"The model call failed during {stage}: {exc}") from exc
        if not isinstance(parsed, schema):
            error = LoopError(
                f"The model returned {type(parsed).__name__} during {stage}, expected {schema.__name__}."
            )
            self._log_result(
                number, stage, schema, started, parsed=parsed, error=error, client=client
            )
            raise error
        usage = self._count(system_text, user_text, parsed, schema, client)
        self._log_result(
            number, stage, schema, started, parsed=parsed, error=None, usage=usage, client=client
        )
        return parsed

    def _count(
        self,
        system_text: str,
        user_text: str,
        parsed: BaseModel,
        schema: type[T],
        client: ModelClient,
    ) -> Usage:
        """Charge one finished call to the meter: reported usage when the client has it."""

        reported: Usage | None = None
        last = getattr(client, "last_call", None)
        if isinstance(last, Mapping):
            raw = last.get("usage")
            reported = raw if isinstance(raw, Usage) else usage_from_response(raw)
        return self._meter.record(
            system_text=system_text,
            user_text=user_text,
            output_text=parsed.model_dump_json(),
            kind=schema.__name__,
            reported=reported,
        )

    def _log_result(
        self,
        number: int,
        stage: str,
        schema: type[object],
        started: float,
        *,
        parsed: object,
        error: Exception | None,
        usage: Usage | None = None,
        client: ModelClient | None = None,
    ) -> None:
        if not logger.isEnabledFor(10):  # logging.DEBUG
            return
        last = getattr(self._client if client is None else client, "last_call", None)
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
        meta: dict[str, object] = {"schema": schema.__name__, "seconds": time.monotonic() - started}
        if usage is not None:
            meta["tokens"] = {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "estimated": usage.estimated,
                "used_so_far": self._meter.used,
                "budget": self._meter.limit,
            }
        block(f"RESULT {number}: {stage} -> {'error' if error else 'ok'}", *sections, meta=meta)

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


def _budget(value: object, name: str = "token_budget") -> int | None:
    """A token budget: ``None`` for no limit, else an integer of at least 1."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LoopError(f"{name} must be an integer of at least 1.")
    return value


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw.replace("_", "").replace(",", ""))
    except ValueError:
        raise LoopError(f"{name} must be an integer of at least 1.") from None
    return _budget(value, name)


_STOP_MESSAGES: Mapping[str, str] = {
    "passed": "stop: passed",
    "max_turns": "stop: max turns",
    "token_budget": "stop: token budget",
}


def _usage(meter: TokenMeter) -> TokenUsage:
    return TokenUsage(
        input_tokens=meter.input_tokens,
        output_tokens=meter.output_tokens,
        calls=meter.calls,
        estimated=meter.estimated,
        budget=meter.limit,
    )


def _tokens_meta(tokens: TokenUsage) -> dict[str, object]:
    return {
        "input": tokens.input_tokens,
        "output": tokens.output_tokens,
        "total": tokens.total,
        "estimated": tokens.estimated,
        "budget": tokens.budget,
    }


def _unchecked(verdicts: Sequence[InstructionVerdict], notes: str) -> DraftCheck:
    """The audit of a turn the token budget cut short: judged so far, nothing more."""

    return DraftCheck(
        instructions=list(verdicts), grounded=False, revise=True, notes=notes, gaps=[]
    )


def _revision_problem(
    draft: Draft,
    audit: DraftCheck,
    unknown: list[str],
    instructions: Sequence[str],
    mode: str = "knowledge",
    missing_note: str | None = None,
    length: LengthCheck | None = None,
    structure: StructureCheck | None = None,
) -> tuple[str | None, tuple[InstructionCheck, ...]]:
    judged, instruction_problems = _judge_instructions(instructions, audit.instructions, missing_note)
    problems: list[str] = []
    if not draft.answer.strip():
        problems.append("The draft was empty. Write the answer the user should read.")
    if length is not None and not length.followed:
        problems.append(
            f"The answer is {length.actual:,} characters; keep it under {length.limit:,} characters, "
            f"about {words_for(length.limit):,} words. Cut what the task does not need."
        )
    if structure is not None and not structure.followed:
        listed = "\n".join(f"- {item}" for item in structure.problems)
        problems.append(
            "The answer must be one JSON document that follows the OUTPUT SCHEMA, with nothing "
            f"outside it. Problems:\n{listed}"
        )
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
    # and a grounding miss, also send the draft back. A length miss is already
    # in ``problems`` and sends it back on its own.
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
    missing_note: str | None = None,
) -> tuple[tuple[InstructionCheck, ...], list[str]]:
    """Pair each instruction with the verdict of that number.

    A missing verdict counts as not followed, with ``missing_note`` as its
    note when one is given (the token budget stopped the check) and
    otherwise a note that the check skipped it. When the same number is
    judged twice, a failure wins.
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
            note = missing_note or "The check did not judge this instruction."
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
    length: LengthCheck | None = None,
    structure: StructureCheck | None = None,
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
        length=length,
        structure=structure,
    )


def _measure(draft: Draft, limit: int | None) -> LengthCheck | None:
    """The response length check of a draft, in code; None when the length is not controlled."""

    if limit is None:
        return None
    actual = len(draft.answer.strip())
    return LengthCheck(limit=limit, actual=actual, followed=length_compliant(actual, limit))


def _response_length(value: object) -> ResponseLength:
    """``None``, ``"auto"``, or a character count of at least 1."""

    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower().replace(",", "").replace("_", "")
        if cleaned == "auto":
            return "auto"
        if cleaned.isdigit() and int(cleaned) >= 1:
            return int(cleaned)
    elif isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    raise LoopError('response_length must be "auto" or an integer of at least 1 (characters).')


def _finish(
    draft: Draft | None,
    known: Mapping[str, Passage],
    checks: list[Check],
    turns: int,
    reason: StoppedReason,
    response: ResponseKind,
    mode: str = "knowledge",
    role: str = "",
    tokens: TokenUsage | None = None,
    response_length: int | None = None,
    output_format: OutputFormat = "text",
    data: object = None,
) -> Answer:
    citations: list[Citation] = []
    for cited in citation_ids(draft) if draft is not None else ():
        passage = known.get(cited)
        if passage is None:
            continue
        citations.append(
            Citation(
                id=passage.id, source=passage.source, heading=passage.heading, text=passage.text
            )
        )
    return Answer(
        text=draft.answer.strip() if draft is not None else "",
        response=response,
        mode=cast(Literal["knowledge", "open"], mode),
        role=role,
        citations=tuple(citations),
        checks=tuple(checks),
        turns=turns,
        stopped_reason=reason,
        tokens=TokenUsage(0, 0, 0, False, None) if tokens is None else tokens,
        response_length=response_length,
        output_format=output_format,
        data=data,
    )
