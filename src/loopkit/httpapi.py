"""HTTP calls for OpenAI, Anthropic, and OpenAI-compatible servers.

The xAI provider uses the official SDK. These clients speak HTTP so the
library does not take a second vendor SDK.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from loopkit.errors import LoopError

T = TypeVar("T")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def strict_json_schema(model: type[BaseModel]) -> dict[str, object]:
    """JSON Schema with every object closed and every property required.

    OpenAI strict mode and Anthropic structured outputs both require this.
    """

    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    constrained = _constrain(schema, definitions)
    if not isinstance(constrained, dict):
        raise LoopError(f"Could not build a JSON schema for {model.__name__}.")
    return constrained


def _constrain(node: object, definitions: dict[str, object]) -> object:
    if isinstance(node, list):
        return [_constrain(item, definitions) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = str(node["$ref"]).rsplit("/", 1)[-1]
        target = definitions.get(name)
        if not isinstance(target, dict):
            raise LoopError(f"The schema refers to unknown definition {name}.")
        return _constrain(target, definitions)
    cleaned: dict[str, object] = {}
    for key, value in node.items():
        if key in {"title", "$schema", "default"}:
            continue
        cleaned[key] = _constrain(value, definitions)
    if cleaned.get("type") == "object" or "properties" in cleaned:
        properties = cleaned.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            cleaned["properties"] = properties
        cleaned["type"] = "object"
        cleaned["additionalProperties"] = False
        cleaned["required"] = list(properties)
    return cleaned


class OpenAICompatibleClient:
    """Chat completions with a JSON schema. Used for non-OpenAI servers."""

    provider = "compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = _join(base_url, "chat/completions")
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens

    def parse(self, *, system_text: str, user_text: str, schema: type[T]) -> T:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(schema),
                    "strict": True,
                    "schema": strict_json_schema(schema),
                },
            },
        }
        if self._max_output_tokens is not None:
            payload["max_tokens"] = self._max_output_tokens
        body = post_json(
            self._url,
            payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        return _validate(schema, _chat_text(body))


class OpenAIClient:
    """OpenAI Responses API. ``store`` is false because the API stores by default."""

    provider = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = _join(base_url, "responses")
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens

    def parse(self, *, system_text: str, user_text: str, schema: type[T]) -> T:
        payload: dict[str, object] = {
            "model": self.model,
            "store": False,
            "input": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": _schema_name(schema),
                    "strict": True,
                    "schema": strict_json_schema(schema),
                }
            },
        }
        if self._max_output_tokens is not None:
            payload["max_output_tokens"] = self._max_output_tokens
        body = post_json(
            self._url,
            payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        return _validate(schema, _responses_text(body))


class AnthropicClient:
    """Anthropic Messages API with JSON-schema output."""

    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = _anthropic_url(base_url)
        self._timeout = timeout
        self._max_output_tokens = 8192 if max_output_tokens is None else max_output_tokens

    def parse(self, *, system_text: str, user_text: str, schema: type[T]) -> T:
        payload = {
            "model": self.model,
            "max_tokens": self._max_output_tokens,
            "system": system_text,
            "messages": [{"role": "user", "content": user_text}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": strict_json_schema(schema),
                }
            },
        }
        body = post_json(
            self._url,
            payload,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        return _validate(schema, _anthropic_text(body))


def post_json(url: str, payload: object, *, headers: dict[str, str], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        suffix = f": {detail}" if detail else ""
        raise LoopError(f"The model API returned HTTP {exc.code}{suffix}") from exc
    except urllib.error.URLError as exc:
        raise LoopError(f"Could not reach the model API at {url}: {exc.reason}") from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LoopError("The model API returned a response that is not JSON.") from exc
    if not isinstance(body, dict):
        raise LoopError("The model API returned a response that is not a JSON object.")
    return body


def is_local_base_url(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    return host in _LOCAL_HOSTS


def _join(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    tail = suffix.strip("/")
    if base.endswith("/" + tail):
        return base
    return f"{base}/{tail}"


def _anthropic_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _schema_name(schema: type[object]) -> str:
    name = schema.__name__
    cleaned = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in name)
    return (cleaned or "Result")[:64]


def _validate(schema: type[T], text: str) -> T:
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise LoopError("The loop expected a Pydantic schema.")
    try:
        return schema.model_validate_json(_strip_fence(text))
    except ValidationError as exc:
        raise LoopError(f"The model returned JSON that does not match {schema.__name__}.") from exc


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: body.rstrip().rfind("```")]
    return body.strip()


def _chat_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LoopError("The model returned no text.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LoopError("The model returned no text.")
    content = message.get("content")
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise LoopError("The model returned no text.")
    return content


def _responses_text(payload: dict[str, object]) -> str:
    output = payload.get("output")
    parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "output_text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    if not parts:
        raise LoopError("The model returned no text.")
    return "".join(parts)


def _anthropic_text(payload: dict[str, object]) -> str:
    content = payload.get("content")
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
    if not parts:
        raise LoopError("The model returned no text.")
    return "\n".join(parts)
