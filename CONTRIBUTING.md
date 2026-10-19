# Contributing to loop-kit

Thanks for taking the time. This page covers how to set up a development
environment, what the checks are, and what a good pull request looks like.
Questions that are not bugs go to [Discussions](https://github.com/OWNER/loop-kit/discussions);
security reports go through [SECURITY.md](SECURITY.md), not the issue tracker.

## Scope

loop-kit is deliberately small: documents in, a draft/check/revise loop, a
grounded answer out. Contributions that fit:

- bug fixes, clearer errors, better chunking or retrieval within the BM25 design
- new OpenAI-compatible providers, or fixes to the existing provider clients
- tests, documentation, examples

Things that are out of scope by design (see the README): a vector database or
embedding model, an agent graph, a second product built around the CLI. Open a
Discussion first if you think one of those should change; please do not open a
PR for it cold.

For anything beyond a small fix, **open an issue before writing code** so we
can agree on the approach. It saves everyone's time.

## Development setup

```bash
git clone https://github.com/OWNER/loop-kit.git
cd loop-kit
make setup                     # creates .venv, installs ".[dev]", registers pre-commit
```

`make` needs GNU make and bash (Windows: Git Bash or WSL). The manual
equivalent is:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # package + every provider extra + tooling
pre-commit install             # runs the checks below on each commit
```

Python 3.11 or newer. No API key is needed for development; the test suite
never calls a provider.

## Checks

CI runs exactly these commands. Run them locally before pushing.

| What | `make` | Command |
| --- | --- | --- |
| Lint + format check | `make lint` | `ruff check src tests examples && ruff format --check src tests examples` |
| Auto-fix and reformat | `make fmt` | `ruff check --fix src tests examples && ruff format src tests examples` |
| Types | `make typecheck` | `mypy` (strict; covers `src` and `tests`) |
| Tests | `make test` (`ARGS="-k name"`) | `pytest` |
| Coverage report | `make cov` | `pytest --cov --cov-report=html` |
| All of the above | `make check` | |
| Everything, the pre-commit way | | `pre-commit run --all-files` |

Configuration for all of them lives in `pyproject.toml`. If a rule is wrong for
this codebase, change the configuration in its own commit and say why, rather
than sprinkling `noqa` / `type: ignore`. When a suppression is the right call,
add a short comment on the same line explaining it.

## Tests

- **Never call a real API.** `tests/fakes.py` has `ScriptedModel`, a stand-in
  that returns the structured results you script and records every prompt.
  Provider HTTP clients are tested by faking `urllib.request.urlopen`
  (see `tests/test_providers.py`); the xAI SDK is faked via `sys.modules`
  (see `tests/test_client.py`).
- **Do not depend on the working directory.** Use pytest's `tmp_path` and
  `monkeypatch.chdir(...)` for anything that touches the filesystem.
- **Write files with `encoding="utf-8"`.** CI runs on Windows too.
- A bug fix should come with a test that fails without the fix.

## Making a change

1. Branch from `main`: `git switch -c fix/short-description`.
2. Keep the change focused. Formatting-only or rename-only changes go in their
   own commit (or their own PR) so the meaningful diff stays readable.
3. Commit messages: an imperative subject line under 72 characters, then a
   blank line, then *why* the change is needed if it is not obvious. Prefixes
   like `ci:`, `deps:`, `docs:` are welcome but not enforced.
4. If the change is visible to users (behaviour, CLI, config file, public API,
   dependencies), add a line under **Unreleased** in `CHANGELOG.md` and update
   the README if it describes the old behaviour.
5. Push and open a pull request. The template asks for a summary, the linked
   issue, and a checklist; fill it in honestly — "not applicable" is a fine answer.

## Pull request review

- CI has to be green. The single required check is **"CI passed"**; open the
  run to see which job failed.
- A maintainer reviews within a few days. Review comments are about the code,
  not the author; please read them the same way.
- PRs are squash-merged, so the PR title becomes the commit subject on `main`.
  Write it as you would a commit subject.
- PRs with no response to review feedback for a month may be closed; reopen
  any time by replying.

## Adding a provider

Providers that speak the OpenAI chat-completions or Responses API usually need
no code: pass `base_url` and `model`. A genuinely different API means:

- a client class in `src/loopkit/httpapi.py` (HTTP) or `src/loopkit/client.py`
  (SDK) with a `parse(system_text=, user_text=, schema=)` method returning an
  instance of `schema`;
- wiring in `make_client` / `resolve_provider` and a default model in
  `DEFAULT_MODELS`;
- tests in `tests/test_providers.py` that assert the exact request body and
  headers against a faked `urlopen`;
- a row in the README provider table. If the provider needs an SDK, make it an
  optional extra in `pyproject.toml` like `xai` rather than a hard dependency.

## Reporting a security issue

Please do not open a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE). There is no CLA.
