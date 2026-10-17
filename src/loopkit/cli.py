"""Command line for trying a loop without writing a script. The library is Loop.ask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loopkit import __version__
from loopkit.errors import LoopError
from loopkit.loop import DEFAULT_MODEL, Loop, ModelClient
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
# loop-kit reads this file when you do not pass knowledge= to Loop().
# Set one API key: XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.

knowledge = "knowledge"

# instructions = [
#   "You are the support assistant for this knowledge base.",
#   "Answer in under 120 words.",
# ]

# provider = "openai"
# model = "gpt-6.1-sol"
# base_url = "http://localhost:11434/v1"
# max_turns = 3
# top_k = 8
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
    shared.add_argument("--model", help=f"Model id. Defaults to the provider's model, or LOOPKIT_MODEL. xAI default: {DEFAULT_MODEL}.")
    shared.add_argument("--provider", help="xai, openai, anthropic, or compatible. Default: the one API key that is set.")
    shared.add_argument("--base-url", help="OpenAI-compatible base URL, for example http://localhost:11434/v1.")
    shared.add_argument("--max-turns", type=int, help="Draft and check rounds. Default: 3.")

    commands = parser.add_subparsers(dest="command", required=True)
    ask_command = commands.add_parser("ask", parents=[shared], help="Answer one task")
    ask_command.add_argument("task", help="Question or task, in quotes if it contains spaces.")
    ask_command.add_argument("--json", action="store_true", help="Print the answer object as JSON.")
    ask_command.add_argument("--quiet", action="store_true", help="Print the answer only.")

    serve_command = commands.add_parser("serve", parents=[shared], help="Try the loaded loop in a browser")
    serve_command.add_argument("--host", default="127.0.0.1", help="Default: 127.0.0.1.")
    serve_command.add_argument("--port", type=int, default=8080)
    serve_command.add_argument("--trace", action="store_true", help="Print each loop step to stderr.")

    commands.add_parser("init", help="Write loopkit.toml and a sample knowledge file")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            _init()
            return 0
        if args.command == "ask":
            trace = False if args.quiet else True
            loop = _loop_from_args(args, client, trace=trace)
            _print_warnings(loop)
            answer = loop.ask(args.task)
            if args.json:
                print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(answer.text)
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


def _loop_from_args(args: argparse.Namespace, client: ModelClient | None, *, trace: bool | None) -> Loop:
    instructions = _instructions(args)
    kwargs: dict[str, object] = {}
    if args.model:
        kwargs["model"] = args.model
    if args.provider:
        kwargs["provider"] = args.provider
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns
    if trace is not None:
        kwargs["trace"] = trace
    if client is not None:
        kwargs["client"] = client
    if instructions is not None:
        kwargs["instructions"] = instructions
    if args.knowledge:
        return Loop(args.knowledge, **kwargs)  # type: ignore[arg-type]
    return Loop.load(**kwargs)


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


def _print_warnings(loop: Loop) -> None:
    for warning in loop.knowledge.warnings:
        print(f"loopkit: {warning}", file=sys.stderr)


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
