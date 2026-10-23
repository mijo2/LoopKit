"""Command line for trying a loop without writing a script. The library is Loop.ask."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

from loopkit import __version__
from loopkit.debuglog import DEFAULT_DEBUG_FILE, logger
from loopkit.errors import LoopError, LoopWarning
from loopkit.loop import DEFAULT_MODEL, Loop, ModelClient
from loopkit.models import Answer
from loopkit.serve import serve

SAMPLE_KNOWLEDGE = """\
# Sample knowledge

Replace these notes with your own documents. loop-kit reads the text files in this folder.

## Refunds

Sample Co refunds a purchase within 30 days. After 30 days the sale is final.

## Support

Email help@example.com when this folder does not answer a billing question.
"""

PROJECT_FILE = """\
# loopkit.toml: the settings for this knowledge base, kept out of code.
#
# loop-kit reads it when you build a Loop without knowledge=, call loopkit.ask()
# without knowledge, or run `loopkit ask` without --knowledge. Paths are relative
# to this file. Set one API key in the environment, never here:
# XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.

knowledge = "knowledge"

# instructions = [
#   "You are the support assistant for this knowledge base.",
#   "Answer in under 120 words.",
# ]

# provider = "openai"
# model = "gpt-6.1-sol"
# base_url = "http://localhost:11434/v1"
# check_model = "gpt-6.1-mini"  # a second model for the checks; the drafts stay on model
# check_provider = "anthropic"  # its provider, when it is not the draft's
# check_base_url = "http://localhost:11434/v1"
# max_turns = 3
# top_k = 8
# check_mode = "each"  # or "all"
# response = "normal"  # or "book"
# search = true        # provider web search for the draft and the audit
# auto_optimize = false  # skip the role-writing call before the loop
# token_budget = 20000  # cap on tokens per ask; the loop stops early when reached
# token_budget_mode = "soft"  # soft: a turn at a time, always a checked answer. hard: a call at a time
# response_length = "auto"  # or a number of characters; each draft is held to it
# output_format = "json"    # one JSON document held to output_schema, instead of text
# output_schema = "schema.json"  # a JSON Schema or an example of the output; a file path or a table
"""


def main(argv: list[str] | None = None, *, client: ModelClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loopkit",
        description="Try a loop-kit knowledge base from the terminal. Programs call Loop.ask.",
    )
    parser.add_argument("--version", action="version", version=f"loopkit {__version__}")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--knowledge",
        "-k",
        help="File, directory, or a sentence of source text. Omit to use loopkit.toml.",
    )
    shared.add_argument(
        "--instructions",
        "-i",
        action="append",
        help="One instruction. Repeat for another. Replaces the built-in instructions.",
    )
    shared.add_argument(
        "--instructions-file",
        help="Read instructions from this file, one per non-empty line.",
    )
    shared.add_argument(
        "--model",
        help=f"Model id. Defaults to the provider's model, or LOOPKIT_MODEL. xAI default: {DEFAULT_MODEL}.",
    )
    shared.add_argument(
        "--provider",
        help="xai, openai, anthropic, or compatible. Default: the one API key that is set.",
    )
    shared.add_argument(
        "--base-url", help="OpenAI-compatible base URL, for example http://localhost:11434/v1."
    )
    shared.add_argument(
        "--check-model",
        help=(
            "A second model for the checks: each instruction check and the grounding or accuracy "
            "audit. Drafts and the optimize step stay on --model. On its own, the same provider "
            "as --model. Same as LOOPKIT_CHECK_MODEL or check_model in loopkit.toml."
        ),
    )
    shared.add_argument(
        "--check-provider",
        help=(
            "The provider for --check-model when it is not the draft's: xai, openai, anthropic, or "
            "compatible. Its key comes from that provider's environment variable. Same as "
            "LOOPKIT_CHECK_PROVIDER."
        ),
    )
    shared.add_argument(
        "--check-base-url",
        help="An OpenAI-compatible server for the checks; needs --check-model. Same as LOOPKIT_CHECK_BASE_URL.",
    )
    shared.add_argument("--max-turns", type=int, help="Draft and check rounds. Default: 3.")
    shared.add_argument(
        "--token-budget",
        type=int,
        metavar="N",
        help=(
            "Cap on the tokens one ask may spend across all of its model calls, input and output "
            "together. When it runs out the loop stops and prints what it has "
            "(stopped_reason: token_budget). Same as LOOPKIT_TOKEN_BUDGET=N or token_budget = N "
            "in loopkit.toml. Default: no cap."
        ),
    )
    shared.add_argument(
        "--token-budget-mode",
        choices=("soft", "hard"),
        help=(
            "How --token-budget is held. soft estimates a whole turn before starting it and lets a "
            "started turn finish, so the answer is always a checked draft and the total can end over "
            "the cap by up to one turn; the first turn always runs. hard estimates every call, the "
            "first draft included, makes none that would not fit, and stops the moment the cap is "
            "exceeded, printing the draft as it stands with the verdicts reached so far, or nothing "
            "when even the first draft would not fit. Same as LOOPKIT_TOKEN_BUDGET_MODE or "
            "token_budget_mode in loopkit.toml. Default: soft."
        ),
    )
    shared.add_argument(
        "--check-mode",
        choices=("each", "all"),
        help="each checks one instruction per call. all checks every instruction in one call. Default: each.",
    )
    shared.add_argument(
        "--response",
        choices=("normal", "book"),
        help="normal is a direct answer. book asks for a long response. Default: normal.",
    )
    shared.add_argument(
        "--search",
        action="store_true",
        help=(
            "Let the model use the provider's web search for the draft and the grounding or accuracy check "
            "(xai, openai, anthropic). Same as LOOPKIT_SEARCH=1 or search = true in loopkit.toml."
        ),
    )
    shared.add_argument(
        "--no-auto-optimize",
        action="store_true",
        help=(
            "Skip the optimize step. By default one call before the loop writes a role for the task "
            "(\"You are a ...\") that prefixes every draft and check. Same as LOOPKIT_AUTO_OPTIMIZE=0."
        ),
    )
    shared.add_argument(
        "--response-length",
        metavar="auto|N",
        help=(
            "Control the length of the answer. auto asks each draft to keep under the length the "
            "estimate step expects for the task; N asks for N characters. Either is held under what "
            "the model returns in one call. The length is measured in code after every draft, and a "
            "draft over it is revised. Same as LOOPKIT_RESPONSE_LENGTH or response_length in "
            "loopkit.toml. Default: the length is left to the instructions."
        ),
    )
    shared.add_argument(
        "--output-format",
        choices=("text", "json"),
        help=(
            "text is prose, the default. json asks for one JSON document held to --output-schema: "
            "every draft is parsed and validated in code, and one that does not follow the schema "
            "is revised with the exact problems. Same as output_format in loopkit.toml."
        ),
    )
    shared.add_argument(
        "--output-schema",
        metavar="FILE|JSON",
        help=(
            "The shape for --output-format json: a JSON Schema, or an example of the output from "
            "which a schema is inferred (every key required, no others, types from the values). "
            "A file path or inline JSON. Implies --output-format json. Same as output_schema in "
            "loopkit.toml."
        ),
    )
    shared.add_argument(
        "--debug",
        nargs="?",
        const=DEFAULT_DEBUG_FILE,
        metavar="FILE",
        help=(
            "Append every prompt, raw model reply, and loop decision to FILE "
            f"(default: {DEFAULT_DEBUG_FILE}). Same as LOOPKIT_DEBUG=FILE."
        ),
    )

    commands = parser.add_subparsers(dest="command", required=True)
    ask_command = commands.add_parser(
        "ask",
        parents=[shared],
        help="Answer one task",
        epilog=(
            "The task can also come from --task-file, or from stdin when TASK is '-' or omitted "
            "(for example: echo 'What does `ls` do?' | loopkit ask). That avoids shell quoting: "
            "in bash and zsh a backtick inside double quotes runs a command, and in PowerShell "
            "the backtick is the escape character. Single quotes are safe in bash and zsh."
        ),
    )
    ask_command.add_argument(
        "task",
        nargs="?",
        help="Question or task. Use '-' or leave it out to read the task from stdin.",
    )
    ask_command.add_argument(
        "--task-file",
        metavar="FILE",
        help="Read the task from this file instead of the command line.",
    )
    ask_command.add_argument("--json", action="store_true", help="Print the answer object as JSON.")
    ask_command.add_argument("--quiet", action="store_true", help="Print the answer only.")

    serve_command = commands.add_parser(
        "serve", parents=[shared], help="Try the loaded loop in a browser"
    )
    serve_command.add_argument("--host", default="127.0.0.1", help="Default: 127.0.0.1.")
    serve_command.add_argument("--port", type=int, default=8080)
    serve_command.add_argument(
        "--trace", action="store_true", help="Print each loop step to stderr."
    )

    commands.add_parser("init", help="Write loopkit.toml and a sample knowledge file")

    args = parser.parse_args(argv)
    # The CLI prints knowledge warnings itself, in its own format.
    warnings.simplefilter("ignore", LoopWarning)
    _log_warnings_to_stderr()
    try:
        if args.command == "init":
            _init()
            return 0
        if args.command == "ask":
            task = _task(args)
            trace = not args.quiet
            loop = _loop_from_args(args, client, trace=trace)
            _print_warnings(loop)
            answer = loop.ask(task)
            if args.json:
                print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
            elif answer.text:
                print(answer.text)
            else:
                print("loopkit: no answer: " + _why_empty(answer), file=sys.stderr)
            return 0
        if args.command == "serve":
            loop = _loop_from_args(args, client, trace=True if args.trace else None)
            _print_warnings(loop)
            if args.host not in {"127.0.0.1", "localhost", "::1"}:
                print(
                    "loopkit: listening beyond localhost with no authentication.",
                    file=sys.stderr,
                )
            serve(loop, args.host, args.port)
            return 0
    except LoopError as exc:
        print(f"loopkit: {exc}", file=sys.stderr)
        return 2
    return 2


def _loop_from_args(
    args: argparse.Namespace, client: ModelClient | None, *, trace: bool | None
) -> Loop:
    instructions = _instructions(args)
    kwargs: dict[str, object] = {}
    if args.model:
        kwargs["model"] = args.model
    if args.provider:
        kwargs["provider"] = args.provider
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.check_model:
        kwargs["check_model"] = args.check_model
    if args.check_provider:
        kwargs["check_provider"] = args.check_provider
    if args.check_base_url:
        kwargs["check_base_url"] = args.check_base_url
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns
    if args.token_budget is not None:
        kwargs["token_budget"] = args.token_budget
    if args.token_budget_mode:
        kwargs["token_budget_mode"] = args.token_budget_mode
    if args.response_length:
        kwargs["response_length"] = args.response_length
    if args.output_format:
        kwargs["output_format"] = args.output_format
    if args.output_schema:
        kwargs["output_schema"] = _output_schema(args.output_schema)
    if args.check_mode:
        kwargs["check_mode"] = args.check_mode
    if args.response:
        kwargs["response"] = args.response
    if args.debug:
        kwargs["debug"] = args.debug
    if args.search:
        kwargs["search"] = True
    if args.no_auto_optimize:
        kwargs["auto_optimize"] = False
    if trace is not None:
        kwargs["trace"] = trace
    if client is not None:
        kwargs["client"] = client
    if instructions is not None:
        kwargs["instructions"] = instructions
    if args.knowledge:
        return Loop(args.knowledge, **kwargs)  # type: ignore[arg-type]
    return Loop.load(**kwargs)


def _output_schema(value: str) -> str:
    """The schema argument: the contents of a file when it names one, else the text itself."""

    text = value.strip()
    path = Path(text)
    if not text.startswith(("{", "[")) and path.is_file():
        return path.read_text(encoding="utf-8")
    return text


def _why_empty(answer: Answer) -> str:
    if answer.stopped_reason == "token_budget" and answer.turns == 0:
        return "the token budget was too small for the first draft."
    return "the model returned an empty draft."


def _instructions(args: argparse.Namespace) -> list[str] | None:
    if args.instructions and args.instructions_file:
        raise LoopError("Pass either --instructions or --instructions-file.")
    if args.instructions_file:
        path = Path(args.instructions_file)
        if not path.is_file():
            raise LoopError(f"No instructions file at {path}.")
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise LoopError(f"Instructions file {path} is empty.")
        return lines
    if args.instructions:
        return list(args.instructions)
    return None


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stderr`` is at the time of the message."""

    loopkit_stderr = True

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> object:
        return sys.stderr

    @stream.setter
    def stream(self, value: object) -> None:
        pass


def _log_warnings_to_stderr() -> None:
    """Show retries and other warnings from the loop on stderr, once per process."""

    if any(getattr(handler, "loopkit_stderr", False) for handler in logger.handlers):
        return
    handler = _StderrHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("loopkit: %(message)s"))
    logger.addHandler(handler)


def _task(args: argparse.Namespace) -> str:
    """The task from the argument, a file, or stdin."""

    if args.task_file:
        if args.task not in (None, "-"):
            raise LoopError("Pass either a task argument or --task-file, not both.")
        path = Path(args.task_file)
        if not path.is_file():
            raise LoopError(f"No task file at {path}.")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise LoopError(f"Task file {path} is empty.")
        return text
    if args.task is not None and args.task != "-":
        return args.task
    if sys.stdin is None or sys.stdin.isatty():
        raise LoopError(
            "Give a task: loopkit ask 'your question', or --task-file FILE, "
            "or pipe the task on stdin."
        )
    text = sys.stdin.read().strip()
    if not text:
        raise LoopError("The task on stdin is empty.")
    return text


def _print_warnings(loop: Loop) -> None:
    for warning in loop.knowledge.warnings:
        print(f"loopkit: {warning}", file=sys.stderr)
    if loop.debug_path is not None:
        print(f"loopkit: debug log: {loop.debug_path}", file=sys.stderr)


def _init() -> None:
    project = Path("loopkit.toml")
    folder = Path("knowledge")
    sample = folder / "start.md"
    if project.exists():
        print("loopkit.toml already exists. Left it in place.", file=sys.stderr)
    else:
        project.write_text(PROJECT_FILE, encoding="utf-8")
        print(f"Wrote {project}", file=sys.stderr)
    if folder.exists() and not folder.is_dir():
        print("knowledge exists and is not a directory. Left it in place.", file=sys.stderr)
        return
    if sample.exists():
        print("knowledge/start.md already exists. Left it in place.", file=sys.stderr)
        return
    folder.mkdir(parents=True, exist_ok=True)
    sample.write_text(SAMPLE_KNOWLEDGE, encoding="utf-8")
    print(f"Wrote {sample}", file=sys.stderr)
    print("Next: set XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.", file=sys.stderr)
    print('Try it: loopkit ask "What is the refund window?"', file=sys.stderr)
    print('In code: Loop(knowledge="knowledge").ask("What is the refund window?")', file=sys.stderr)
