"""A second model for the checks: routing, resolution, and the knobs that set it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopkit.cli import main
from loopkit.config import load_project
from loopkit.errors import LoopError
from loopkit.httpapi import AnthropicClient, OpenAIClient, OpenAICompatibleClient
from loopkit.loop import DEFAULT_INSTRUCTIONS, Loop
from loopkit.schemas import Draft, DraftCheck, GroundingCheck, OneCheck, RolePrefix
from tests.fakes import ScriptedModel, SearchAwareModel, accepted, audit, checks
from tests.test_providers import _clear

POLICY = "Refunds are available for 14 days after the charge. Keys are shown once."


def _split() -> tuple[ScriptedModel, ScriptedModel]:
    """A drafter with only draft replies and a checker with only check replies."""

    drafter = ScriptedModel([Draft(answer="Refunds last 14 days. [p1]", cited_ids=["p1"])])
    checker = ScriptedModel(checks())
    return drafter, checker


def test_without_check_options_the_checks_share_the_draft_client() -> None:
    model = ScriptedModel(accepted("Refunds last 14 days. [p1]", "p1"))
    loop = Loop(knowledge=POLICY, client=model, trace=False)
    assert loop._check_client is loop._client
    assert loop.check_provider == loop.provider == "custom"
    assert loop.check_model == loop.model
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"
    assert len(model.calls) == 1 + len(DEFAULT_INSTRUCTIONS) + 1


def test_checks_go_to_the_check_client_and_drafts_stay(capsys: pytest.CaptureFixture[str]) -> None:
    drafter, checker = _split()
    loop = Loop(
        knowledge=POLICY, client=drafter, check_client=checker, check_model="small", trace=True
    )
    answer = loop.ask("What is the refund window?")

    assert answer.stopped_reason == "passed"
    assert [call["schema"] for call in drafter.calls] == [Draft]
    assert [call["schema"] for call in checker.calls] == [OneCheck] * len(DEFAULT_INSTRUCTIONS) + [
        GroundingCheck
    ]
    assert loop.check_model == "small"
    assert loop.check_provider == "custom"
    assert answer.tokens.calls == 6
    err = capsys.readouterr().err
    assert "models: custom grok-4.7 drafts; custom small checks" in err


def test_the_optimize_step_stays_on_the_draft_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPKIT_AUTO_OPTIMIZE", "1")
    drafter = ScriptedModel(
        [
            RolePrefix(role="You are a billing specialist."),
            Draft(answer="Refunds last 14 days. [p1]", cited_ids=["p1"]),
        ]
    )
    checker = ScriptedModel(checks())
    loop = Loop(knowledge=POLICY, client=drafter, check_client=checker, trace=False)
    answer = loop.ask("What is the refund window?")
    assert answer.role == "You are a billing specialist."
    assert [call["schema"] for call in drafter.calls] == [RolePrefix, Draft]
    assert all(call["schema"] is not Draft for call in checker.calls)


def test_check_mode_all_and_revisions_route_the_same_way() -> None:
    drafter = ScriptedModel(
        [
            Draft(answer="Refunds last 14 days.", cited_ids=[]),
            Draft(answer="Refunds last 14 days. [p1]", cited_ids=["p1"]),
        ]
    )
    checker = ScriptedModel(
        [audit(revise=True, notes="Cite it.", follows=False), audit(revise=False)]
    )
    loop = Loop(
        knowledge=POLICY, client=drafter, check_client=checker, check_mode="all", trace=False
    )
    answer = loop.ask("What is the refund window?")
    assert answer.turns == 2
    assert answer.stopped_reason == "passed"
    assert [call["schema"] for call in drafter.calls] == [Draft, Draft]
    assert [call["schema"] for call in checker.calls] == [DraftCheck, DraftCheck]
    assert "Cite it." in str(drafter.calls[1]["user"])


def test_search_needs_a_check_client_that_accepts_it() -> None:
    with pytest.raises(LoopError, match="check_client whose parse\\(\\) accepts a search keyword"):
        Loop(
            knowledge=POLICY,
            client=SearchAwareModel([]),  # type: ignore[arg-type]
            check_client=ScriptedModel([]),
            search=True,
        )
    drafter = SearchAwareModel([Draft(answer="Refunds last 14 days. [p1]", cited_ids=["p1"])])
    checker = SearchAwareModel(list(checks()))
    loop = Loop(knowledge=POLICY, client=drafter, check_client=checker, search=True, trace=False)  # type: ignore[arg-type]
    loop.ask("What is the refund window?")
    assert drafter.search_flags == [True]
    # Per-instruction checks never search; the grounding audit does.
    assert checker.search_flags == [False] * len(DEFAULT_INSTRUCTIONS) + [True]


def test_check_model_alone_means_the_same_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    loop = Loop(knowledge=POLICY, check_model="gpt-6.1-mini", trace=False)
    assert loop.provider == "openai"
    assert loop.model == "gpt-6.1-sol"
    assert loop.check_provider == "openai"
    assert loop.check_model == "gpt-6.1-mini"
    assert isinstance(loop._check_client, OpenAIClient)
    assert loop._check_client is not loop._client
    assert loop._check_client._api_key == "sk-main"

    # An explicit key and server are inherited on the same provider.
    loop = Loop(
        knowledge=POLICY,
        provider="compatible",
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        api_key="local-key",
        check_model="llama3.1:8b",
        trace=False,
    )
    assert isinstance(loop._check_client, OpenAICompatibleClient)
    assert loop._check_client._api_key == "local-key"
    assert loop._check_client._url == loop._client._url  # type: ignore[attr-defined]
    assert loop.check_model == "llama3.1:8b"


def test_check_provider_uses_that_providers_key_and_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("LOOPKIT_MODEL", "gpt-6.1-pro")
    loop = Loop(knowledge=POLICY, provider="openai", check_provider="anthropic", trace=False)
    assert loop.model == "gpt-6.1-pro"
    assert loop.check_provider == "anthropic"
    # LOOPKIT_MODEL is the draft's; the check model is the provider's default.
    assert loop.check_model == "claude-sonnet-5-5"
    assert isinstance(loop._check_client, AnthropicClient)
    assert loop._check_client._api_key == "sk-ant"

    loop = Loop(
        knowledge=POLICY,
        provider="openai",
        check_provider="anthropic",
        check_model="claude-haiku-5",
        check_api_key="sk-other",
        trace=False,
    )
    assert loop.check_model == "claude-haiku-5"
    assert isinstance(loop._check_client, AnthropicClient)
    assert loop._check_client._api_key == "sk-other"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(LoopError, match="ANTHROPIC_API_KEY"):
        Loop(knowledge=POLICY, provider="openai", check_provider="anthropic", trace=False)


def test_check_base_url_is_a_compatible_server_that_needs_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    with pytest.raises(LoopError, match="check_model is needed with check_base_url"):
        Loop(knowledge=POLICY, check_base_url="http://localhost:11434/v1", trace=False)
    loop = Loop(
        knowledge=POLICY,
        check_base_url="http://localhost:11434/v1",
        check_model="llama3.1",
        trace=False,
    )
    assert loop.provider == "openai"
    assert loop.check_provider == "compatible"
    assert loop.check_model == "llama3.1"
    assert isinstance(loop._check_client, OpenAICompatibleClient)
    # A compatible check server cannot run the grounding audit's web search.
    with pytest.raises(LoopError, match="search needs a provider with a web search tool"):
        Loop(
            knowledge=POLICY,
            check_base_url="http://localhost:11434/v1",
            check_model="llama3.1",
            search=True,
            trace=False,
        )


def test_a_custom_draft_client_can_still_have_a_provider_check_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    drafter = ScriptedModel([])
    loop = Loop(knowledge=POLICY, client=drafter, check_model="claude-haiku-5", trace=False)
    assert loop.provider == "custom"
    assert loop.check_provider == "anthropic"
    assert loop.check_model == "claude-haiku-5"
    assert isinstance(loop._check_client, AnthropicClient)


def test_environment_and_project_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("LOOPKIT_PROVIDER", "openai")
    monkeypatch.setenv("LOOPKIT_CHECK_PROVIDER", "anthropic")
    monkeypatch.setenv("LOOPKIT_CHECK_MODEL", "claude-haiku-5")
    loop = Loop(knowledge=POLICY, trace=False)
    assert (loop.check_provider, loop.check_model) == ("anthropic", "claude-haiku-5")
    monkeypatch.delenv("LOOPKIT_CHECK_PROVIDER")
    monkeypatch.delenv("LOOPKIT_CHECK_MODEL")
    monkeypatch.setenv("LOOPKIT_CHECK_BASE_URL", "http://localhost:11434/v1")
    with pytest.raises(LoopError, match="check_model is needed"):
        Loop(knowledge=POLICY, trace=False)
    monkeypatch.delenv("LOOPKIT_CHECK_BASE_URL")

    (tmp_path / "knowledge.md").write_text(POLICY, encoding="utf-8")
    toml = tmp_path / "loopkit.toml"
    toml.write_text(
        'knowledge = "knowledge.md"\nprovider = "openai"\ncheck_provider = "anthropic"\n'
        'check_model = "claude-haiku-5"\n',
        encoding="utf-8",
    )
    project = load_project(toml)
    assert (project.check_provider, project.check_model, project.check_base_url) == (
        "anthropic",
        "claude-haiku-5",
        None,
    )
    monkeypatch.chdir(tmp_path)
    loop = Loop.load(trace=False)
    assert (loop.check_provider, loop.check_model) == ("anthropic", "claude-haiku-5")
    loop = Loop.load(trace=False, check_provider="openai", check_model="gpt-6.1-mini")
    assert (loop.check_provider, loop.check_model) == ("openai", "gpt-6.1-mini")
    toml.write_text('knowledge = "knowledge.md"\ncheck_model = 3\n', encoding="utf-8")
    with pytest.raises(LoopError, match=r"check_model in loopkit\.toml must be a non-empty string"):
        load_project(toml)


def test_cli_flags_reach_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    seen: dict[str, object] = {}
    real = Loop.__init__

    def spy(self: Loop, *args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Loop, "__init__", spy)
    code = main(
        [
            "ask",
            "Refunds?",
            "-k",
            POLICY,
            "--provider",
            "openai",
            "--check-provider",
            "anthropic",
            "--check-model",
            "claude-haiku-5",
            "--check-base-url",
            "http://localhost:9999/v1",
            "--quiet",
        ],
        client=ScriptedModel([]),
    )
    assert seen["check_provider"] == "anthropic"
    assert seen["check_model"] == "claude-haiku-5"
    assert seen["check_base_url"] == "http://localhost:9999/v1"
    # anthropic with a base URL is a real configuration; the ask then fails on the empty script.
    assert code == 2


def test_json_output_reports_nothing_new_but_the_debug_log_names_both(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drafter, checker = _split()
    log = tmp_path / "run.log"
    loop = Loop(
        knowledge=POLICY,
        client=drafter,
        check_client=checker,
        check_model="small",
        debug=log,
        trace=False,
    )
    answer = loop.ask("What is the refund window?")
    assert answer.stopped_reason == "passed"
    text = log.read_text(encoding="utf-8")
    assert "check_model=small" in text
    assert "model=small" in text
    assert "model=grok-4.7" in text
    payload = json.dumps(answer.to_dict())
    assert "small" not in payload
