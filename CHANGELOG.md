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

- A token budget for one `ask`. `Loop(token_budget=N)`,
  `loop.ask(task, token_budget=N)`, `LOOPKIT_TOKEN_BUDGET`, `token_budget` in
  `loopkit.toml`, `loopkit ask --token-budget N`, and `"token_budget"` in the
  body of `POST /ask`. When it runs out the loop stops and returns what it
  has with the new `stopped_reason` `"token_budget"`.
- `token_budget_mode` (`--token-budget-mode`, `LOOPKIT_TOKEN_BUDGET_MODE`,
  `loopkit.toml`): `"soft"`, the default, estimates a whole turn before
  starting it and lets a started turn finish, so the answer is always a fully
  checked draft; the first turn always runs. `"hard"` estimates every call,
  the first draft included, makes none that would not fit, and stops the
  moment the budget is exceeded: the answer is the draft as it stands with
  the verdicts reached so far (the rest carry the note
  `Not checked: the token budget ran out.`), or an empty answer with
  `turns == 0` when even the first draft would not fit.
- `loopkit.estimate`: the deterministic rules that size a reply before the
  call, behind `EstimationStrategy` (`Loop(estimation_strategy=...)`). The
  one strategy, `LAST_TURN`, sizes the first reply of each kind from the
  task, instructions, and passages, and each later one like the latest reply
  of that kind.
- `response_length` (`--response-length`, `LOOPKIT_RESPONSE_LENGTH`,
  `loopkit.toml`): hold each draft to a length in characters, `"auto"` for
  the length the estimate step expects. The target is capped at the model's
  assumed maximum response size (`loopkit.estimate.MODEL_MAX_RESPONSE_CHARS`,
  by model name) and the client's output cap. Each draft prompt ends with a
  `RESPONSE LIMIT` line; the length is measured in code after every draft
  and a draft over it (beyond a tenth of tolerance) is revised with a note.
  `Check.length` (`LengthCheck`: `limit`, `actual`, `followed`) and
  `Answer.response_length` report it, also in `to_dict()` and `--json`.
  Responses longer than one call are not produced yet; see
  `TODO(response-length)` in `src/loopkit/estimate.py`.
- `output_format` (`--output-format`, `loopkit.toml`): `"text"` (the
  default) or `"json"`, with `output_schema` (`--output-schema FILE|JSON`,
  a table or file path in `loopkit.toml`): a JSON Schema, or an example of
  the output from which a closed schema is inferred. The schema goes in the
  drafter's system prompt; every draft is parsed and validated in code by
  `loopkit.structure` (a JSON Schema subset, no dependency) and one off the
  schema is revised with the problems by path. `Check.structure`
  (`StructureCheck`), `Answer.output_format`, and `Answer.data` (the parsed
  document) report it, also in `to_dict()` and `--json`. With the built-in
  instructions the inline-citation rule becomes "name the passages in
  cited_ids".
- The `RESPONSE LIMIT` rule of `response_length` is in the drafter's system
  prompt rather than the task prompt.
- A second model for the checks: `check_model`, `check_provider`,
  `check_base_url`, `check_api_key`, and `check_client` on `Loop`;
  `--check-model`, `--check-provider`, `--check-base-url`;
  `LOOPKIT_CHECK_MODEL`, `LOOPKIT_CHECK_PROVIDER`, `LOOPKIT_CHECK_BASE_URL`;
  and the same three names in `loopkit.toml`. Every instruction check, the
  combined check, and the grounding or accuracy audit go to it; the optimize
  step and the drafts stay on `model`. `check_model` alone means the draft's
  provider; `check_provider` means that provider's key and default model.
  `loop.check_provider` and `loop.check_model` say what was resolved, the
  trace prints both models when they differ, and the debug log names the
  model on every call.
- `Answer.tokens` (`TokenUsage`): input, output, and total tokens, the number
  of calls, whether any were estimated, and the budget; also under `"tokens"`
  in `to_dict()` and `--json`. The provider clients record reported usage in
  `last_call["usage"]`, and a custom client can do the same to be counted
  exactly rather than estimated.
- `loopkit.estimate_tokens`, a dependency-free token estimate, and
  `Loop(token_estimator=...)` to supply a real tokenizer instead.
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
