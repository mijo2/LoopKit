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
