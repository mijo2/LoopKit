"""Build the model client for whichever provider the caller configured.

xAI uses the official SDK. OpenAI and Anthropic use their HTTP APIs. Any other
OpenAI-compatible server uses chat completions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from loopkit.errors import LoopError
from loopkit.httpapi import (
    AnthropicClient,
    OpenAIClient,
    OpenAICompatibleClient,
    is_local_base_url,
)
from loopkit.retry import Transient, retry_call

# gRPC status names the xAI SDK raises when the service is busy or broke
# on its side. These are worth another attempt; anything else is not.
_GRPC_TRANSIENT = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "ABORTED", "DEADLINE_EXCEEDED"}

# grok-4.7 is a reasoning model. The xAI text guide uses a one-hour client timeout.
DEFAULT_TIMEOUT_SECONDS = 3600

DEFAULT_MODELS = {
    "xai": "grok-4.7",
    "openai": "gpt-6.1-sol",
    "anthropic": "claude-sonnet-5-5",
}

_KEY_ENV = {
    "xai": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_KEY_HELP = {
    "xai": "Create one at https://console.x.ai/team/default/api-keys",
    "openai": "Create one at https://platform.openai.com/api-keys",
    "anthropic": "Create one at https://console.anthropic.com/settings/keys",
}

_DEFAULT_BASE = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}

_ALIASES = {
    "grok": "xai",
    "x.ai": "xai",
    "claude": "anthropic",
    "gpt": "openai",
    "openai-compatible": "compatible",
    "openai_compatible": "compatible",
}

_PROVIDER_NAMES = ("xai", "openai", "anthropic", "compatible")


@dataclass(frozen=True)
class ProviderChoice:
    name: str
    default_model: str
    key_env: str
    base_url: str | None


class XAIClient:
    """Call Grok on the xAI API and parse the response into ``schema``."""

    provider = "xai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens: int | None = None,
        search: bool = False,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("XAI_API_KEY", "")
        api_key = api_key.strip()
        if not api_key:
            raise LoopError(_missing_key("xai"))
        try:
            from xai_sdk import Client
        except ImportError as exc:
            raise LoopError(
                "The xAI provider needs the xAI SDK, which is an optional extra. "
                'Install it with: pip install "loop-kit[xai]"'
            ) from exc

        self.model = model
        self._max_output_tokens = max_output_tokens
        self._sdk = Client(api_key=api_key, timeout=timeout)
        self.search = search
        self.last_call: dict[str, object] | None = None

    def parse(
        self, *, system_text: str, user_text: str, schema: type[object], search: bool = False
    ) -> object:
        from xai_sdk.chat import system, user

        # Knowledge bases are often private. Do not store the prompt on the server.
        # The request itself is still sent to the API.
        params: dict[str, object] = {"model": self.model, "store_messages": False}
        if self._max_output_tokens is not None:
            params["max_tokens"] = self._max_output_tokens
        record: dict[str, object] = {"request": dict(params), "response": None, "raw_text": None}
        if search:
            record["request"] = {**params, "tools": ["web_search"]}
        self.last_call = record

        def attempt() -> object:
            if search:
                from xai_sdk.tools import web_search

                chat = self._sdk.chat.create(**params, tools=[web_search()])
            else:
                chat = self._sdk.chat.create(**params)
            chat.append(system(system_text))
            chat.append(user(user_text))
            try:
                response, parsed = chat.parse(schema)
            except Exception as exc:
                status = _grpc_status(exc)
                if status in _GRPC_TRANSIENT:
                    raise Transient(f"xAI {status}: {_brief(exc)}", cause=exc) from exc
                raise
            record["raw_text"] = getattr(response, "content", None)
            record["response"] = {
                name: getattr(response, name)
                for name in ("id", "finish_reason", "usage")
                if getattr(response, name, None) is not None
            }
            return parsed

        return retry_call(attempt, what=f"xAI {self.model}")


def _grpc_status(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if not callable(code):
        return ""
    try:
        return str(getattr(code(), "name", ""))
    except Exception:
        return ""


def _brief(exc: Exception) -> str:
    details = getattr(exc, "details", None)
    text = details() if callable(details) else str(exc)
    text = str(text).strip().replace("\n", " ")
    return text[:200] + ("..." if len(text) > 200 else "")


def make_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_tokens: int | None = None,
    search: bool = False,
) -> XAIClient | OpenAIClient | AnthropicClient | OpenAICompatibleClient:
    choice = resolve_provider(provider, base_url=base_url)
    chosen_model = _choose_model(choice, model)
    key = _choose_key(choice, api_key)
    if choice.name == "xai":
        if choice.base_url:
            raise LoopError(
                "The xAI provider uses the xAI SDK. "
                "For a custom base URL, set provider to 'compatible'."
            )
        return XAIClient(
            model=chosen_model, api_key=key, timeout=timeout, max_output_tokens=max_output_tokens, search=search
        )
    if choice.name == "openai":
        return OpenAIClient(
            model=chosen_model,
            api_key=key,
            base_url=choice.base_url or _DEFAULT_BASE["openai"],
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            search=search,
        )
    if choice.name == "anthropic":
        return AnthropicClient(
            model=chosen_model,
            api_key=key,
            base_url=choice.base_url or _DEFAULT_BASE["anthropic"],
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            search=search,
        )
    if not choice.base_url:
        raise LoopError("provider 'compatible' needs a base_url, for example http://localhost:11434/v1.")
    if search:
        raise LoopError(
            "search needs a provider with a web search tool: xai, openai, or anthropic. "
            "The 'compatible' provider sends plain chat completions."
        )
    return OpenAICompatibleClient(
        model=chosen_model,
        api_key=key,
        base_url=choice.base_url,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
    )


def resolve_provider(provider: str | None, *, base_url: str | None = None) -> ProviderChoice:
    name = (provider or os.environ.get("LOOPKIT_PROVIDER", "")).strip().lower()
    name = _ALIASES.get(name, name)
    url = (base_url or os.environ.get("LOOPKIT_BASE_URL", "")).strip() or None
    if name and name not in _PROVIDER_NAMES:
        names = ", ".join(_PROVIDER_NAMES)
        raise LoopError(f"Unknown provider {name!r}. Use one of: {names}.")
    if not name and url:
        name = "compatible"
    if not name:
        found = [
            item
            for item in ("xai", "openai", "anthropic")
            if os.environ.get(_KEY_ENV[item], "").strip()
        ]
        if len(found) == 1:
            name = found[0]
        elif not found:
            raise LoopError(_no_key_message())
        else:
            listed = ", ".join(found)
            raise LoopError(
                f"Found API keys for {listed}. Set provider or LOOPKIT_PROVIDER to one of: {listed}."
            )
    if name == "xai" and url:
        raise LoopError(
            "The xAI provider uses the xAI SDK. For a custom base URL, set provider to 'compatible'."
        )
    if name == "compatible" and not url:
        raise LoopError(
            "provider 'compatible' needs a base_url, for example http://localhost:11434/v1."
        )
    return ProviderChoice(
        name=name,
        default_model=DEFAULT_MODELS.get(name, ""),
        key_env=_KEY_ENV.get(name, "LOOPKIT_API_KEY"),
        base_url=url or _DEFAULT_BASE.get(name),
    )


def _choose_model(choice: ProviderChoice, model: str | None) -> str:
    chosen = (model or "").strip() or os.environ.get("LOOPKIT_MODEL", "").strip()
    if chosen:
        return chosen
    if choice.default_model:
        return choice.default_model
    raise LoopError("Set model or LOOPKIT_MODEL for this server.")


def _choose_key(choice: ProviderChoice, api_key: str | None) -> str:
    if api_key is not None and api_key.strip():
        return api_key.strip()
    if choice.name == "compatible":
        for variable in ("LOOPKIT_API_KEY", "OPENAI_API_KEY"):
            value = os.environ.get(variable, "").strip()
            if value:
                return value
        if choice.base_url and is_local_base_url(choice.base_url):
            return "local"
        raise LoopError("Set LOOPKIT_API_KEY for this base URL.")
    value = os.environ.get(choice.key_env, "").strip()
    if not value:
        raise LoopError(_missing_key(choice.name))
    return value


def _missing_key(provider: str) -> str:
    variable = _KEY_ENV.get(provider, "LOOPKIT_API_KEY")
    help_text = _KEY_HELP.get(provider, "")
    if help_text:
        return f"Set {variable} to your API key. {help_text}"
    return f"Set {variable} to your API key."


def _no_key_message() -> str:
    return (
        "Set an API key: XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY. "
        "For another OpenAI-compatible server, pass base_url and model."
    )
