# loop-kit developer entry points.
#
#   make help            list targets
#   make ask Q="..."     answer a question (creates the venv and installs on first use)
#   make serve           local web page on http://127.0.0.1:8080
#   make check           everything CI runs: lint, format, types, tests
#
# The virtual environment lives in .venv/ and is created on demand. A stamp
# file records when it was last installed, so repeated `make ask` / `make test`
# calls do not reinstall unless pyproject.toml changed. `make distclean` resets.
#
# API keys: export XAI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY in your
# shell, or put them in a `.env` file next to this Makefile (see .env.example).
# `.env` is gitignored and loaded automatically.
#
# Needs GNU make and bash (macOS and Linux have both; on Windows use Git Bash
# or WSL, or run the commands from CONTRIBUTING.md directly).

SHELL := bash

# ---------------------------------------------------------------------------
# Settings (override on the command line: make test PYTHON=python3.13)
# ---------------------------------------------------------------------------
PYTHON  ?= python3
VENV    ?= .venv
EXTRAS  ?= dev
Q       ?= What is the refund window?
HOST    ?= 127.0.0.1
PORT    ?= 8080
ARGS    ?=

ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
else
BIN := $(VENV)/bin
endif

STAMP := $(VENV)/.installed
SRC   := src tests examples

# Load API keys from .env if present. Lines must be KEY=value with no quotes.
ifneq (,$(wildcard .env))
include .env
export
endif

.DEFAULT_GOAL := help
.PHONY: help setup ask serve init example shell doctor \
        check lint fmt typecheck test cov build clean distclean

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
$(STAMP): pyproject.toml
	@test -d $(VENV) || { echo ">> creating $(VENV) with $(PYTHON)"; $(PYTHON) -m venv $(VENV); }
	@echo ">> installing loop-kit[$(EXTRAS)] into $(VENV)"
	@$(BIN)/python -m pip install --quiet --upgrade pip
	@$(BIN)/python -m pip install --quiet -e ".[$(EXTRAS)]"
	@if [ -d .git ] && [ -x $(BIN)/pre-commit ]; then $(BIN)/pre-commit install >/dev/null; fi
	@touch $@

setup: $(STAMP) ## Create .venv and install the package with dev tooling (idempotent)

shell: $(STAMP) ## Print the command that activates the venv in your shell
	@echo "source $(BIN)/activate"

doctor: $(STAMP) ## Show interpreter, version, and which provider keys are set
	@echo "python:    $$($(BIN)/python --version) at $(BIN)/python"
	@echo "loop-kit:  $$($(BIN)/python -c 'import loopkit; print(loopkit.__version__)')"
	@for v in XAI_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY LOOPKIT_API_KEY; do \
	  if [ -n "$${!v}" ]; then echo "$$v: set ($${!v:0:4}...)"; else echo "$$v: not set"; fi; \
	done
	@echo "LOOPKIT_PROVIDER=$${LOOPKIT_PROVIDER:-<unset>}  LOOPKIT_MODEL=$${LOOPKIT_MODEL:-<unset>}  LOOPKIT_BASE_URL=$${LOOPKIT_BASE_URL:-<unset>}"
	@test -f loopkit.toml && echo "project:   ./loopkit.toml" || echo "project:   none (run 'make init')"

# ---------------------------------------------------------------------------
# Use the loop
# ---------------------------------------------------------------------------
ask: $(STAMP) ## Ask the project's knowledge base: make ask Q="..." [ARGS="--quiet"]
	@$(BIN)/loopkit ask "$(Q)" $(ARGS)

serve: $(STAMP) ## Serve the local page: make serve [HOST=127.0.0.1] [PORT=8080]
	@$(BIN)/loopkit serve --host $(HOST) --port $(PORT) $(ARGS)

init: $(STAMP) ## Write loopkit.toml and knowledge/start.md in the current directory
	@$(BIN)/loopkit init

example: $(STAMP) ## Run the Northwind example: make example [Q="..."]
	@$(BIN)/python examples/northwind/ask.py $(Q)

# ---------------------------------------------------------------------------
# Checks (the same commands CI runs; config lives in pyproject.toml)
# ---------------------------------------------------------------------------
check: lint typecheck test ## Run everything CI runs

lint: $(STAMP) ## ruff check + ruff format --check
	@$(BIN)/ruff check $(SRC)
	@$(BIN)/ruff format --check $(SRC)

fmt: $(STAMP) ## Auto-fix lint findings and reformat
	@$(BIN)/ruff check --fix $(SRC)
	@$(BIN)/ruff format $(SRC)

typecheck: $(STAMP) ## mypy --strict over src and tests
	@$(BIN)/mypy

test: $(STAMP) ## pytest (pass extra flags with ARGS="-k name -x")
	@$(BIN)/pytest $(ARGS)

cov: $(STAMP) ## pytest with a coverage report; writes htmlcov/index.html
	@$(BIN)/pytest --cov --cov-report=term-missing --cov-report=html $(ARGS)
	@echo ">> open htmlcov/index.html"

build: $(STAMP) ## Build sdist + wheel into dist/ and verify them
	@rm -rf dist
	@$(BIN)/python -m build
	@$(BIN)/twine check --strict dist/*

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean: ## Remove build artefacts and caches (keeps .venv)
	@rm -rf dist build htmlcov coverage.xml .coverage .pytest_cache .mypy_cache .ruff_cache
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find src -name "*.egg-info" -type d -prune -exec rm -rf {} + 2>/dev/null || true

distclean: clean ## Remove .venv as well; next make target recreates it
	@rm -rf $(VENV)

help: ## Show this help
	@echo "loop-kit - make targets"
	@echo
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "First run of any target creates $(VENV) and installs the package."
	@echo "Keys: export XAI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY, or put them in .env"
