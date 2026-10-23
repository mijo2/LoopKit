# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) — while the major version is `0`,
minor releases may contain breaking changes, and they are called out below.

## [Unreleased]

### Changed

- `xai-sdk` is now an optional extra: `pip install "loop-kit[xai]"`. OpenAI,
  Anthropic, and OpenAI-compatible servers need no extra. Selecting the xAI
  provider without the extra raises a `LoopError` that names it.
- `Loop.load(path, ...)`: `path` is positional-only, so keyword overrides can
  never collide with it.
- `XAIClient.parse` validates that the SDK returned an instance of the
  requested schema and raises `LoopError` otherwise.

### Added

- `loopkit serve` is now a playground for the loop. The page is a chat with
  tabs. It sends the earlier turns of a tab as `history`, and the server
  (`loopkit.serve.compose_task`) folds them into the one task the loop
  receives (`User asked: ...; Previous conversation: [(ask, answer), ...]`),
  so the library itself stays memoryless. An **Advanced options** panel exposes every
  `Loop(...)` argument per tab: knowledge source (the server's, none for open
  mode, or pasted text), instructions, provider, model, base URL, check mode,
  response, max turns, top k, context budget, search, and auto-optimize. Each
  answer shows its audit trail: role, citations, per-turn verdicts, and the
  settings used.
- `GET /config` reports what the server loaded and which provider keys are
  set (never the values). `POST /ask` accepts optional `history` and
  `options`; options build a `Loop` for that request on the knowledge already
  in memory. The response gains `task` (the string the loop received) and
  `loop` (the settings that produced it).
  `loopkit.serve.loop_for`, `settings`, and `config` are the functions behind
  those routes.
- `loopkit serve --session [FILE]` (and `make serve SESSION=FILE`) keeps the
  page's chats in a JSON file: tabs, transcripts with their audit trails,
  options, and the turns sent as `history`. A refresh loads the file first
  and shows the same state. `GET /session` and `PUT /session` are the routes;
  `loopkit.serve.SessionStore` validates the shape and writes atomically.
  Without the flag nothing is written, as before.
- The page lives in `loopkit/ui/index.html` and ships with the package.
- `loopkit.serve.server_url(server)` returns the URL a server is bound to.
- Strict type checking (`mypy --strict`) over the package and the tests;
  `loopkit` ships `py.typed`.
- `CITATION.cff`, contribution guidelines, security policy, issue and pull
  request templates, CI for Python 3.11–3.13 on Linux, macOS, and Windows,
  and a tag-driven release workflow.

### Fixed

- `tests/test_knowledge.py` no longer depends on the working directory.

## [0.1.0] - 2026-01-01

### Added

- `Loop` and `ask`: answer from a knowledge base with a draft, check, and
  revise loop; one check per instruction (`check_mode="each"`) or all at once
  (`"all"`); `response="book"` for long-form answers.
- Knowledge loading from directories, files, strings, mappings, and lists,
  with markdown-aware chunking and a local BM25 retriever.
- Providers: xAI (Grok), OpenAI (Responses API), Anthropic (Messages API), and
  any OpenAI-compatible server via `base_url`.
- `loopkit.toml` project file, and the `loopkit ask`, `loopkit serve`, and
  `loopkit init` commands.

[Unreleased]: https://github.com/OWNER/loop-kit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/loop-kit/releases/tag/v0.1.0
