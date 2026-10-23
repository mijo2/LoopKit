"""HTTP calls for OpenAI, Anthropic, and OpenAI-compatible servers.

The xAI provider uses the official SDK. These clients speak HTTP so the
library does not take a second vendor SDK.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from loopkit.errors import LoopError
from loopkit.retry import RETRY_STATUSES, Transient, parse_retry_after, retry_call
from loopkit.tokens import usage_from_response

T = TypeVar("T", bound=BaseModel)

# Network failures worth one more try. A refused connection is not one of
# them: the server is not there, and waiting does not start it.
_TRANSIENT_NETWORK = (
    ConnectionResetError,
    BrokenPipeError,
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
)

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


# Provider-native web search. The provider runs the search; the model reads
# the results before it answers. Nothing about it is handled in this library.
OPENAI_SEARCH_TOOL: dict[str, object] = {"type": "web_search"}
ANTHROPIC_SEARCH_TOOL: dict[str, object] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}


class _HttpClient:
    """What the three HTTP clients share: one POST, recorded for the debug log.

    ``last_call`` holds the request payload, the response body, the text
    that was parsed, and the token usage the server reported, for the most
    recent call. The loop reads it when writing the debug log, including
    when the call failed part way, and when counting tokens against a budget.
    """

    last_call: dict[str, object] | None = None
    _timeout: float

    def _post(
        self,
        url: str,
        payload: dict[str, object],
        *,
        headers: dict[str, str],
        extract: Callable[[dict[str, object]], list[str]],
        schema: type[T],
    ) -> T:
        record: dict[str, object] = {
            "url": url,
            "request": payload,
            "response": None,
            "raw_text": None,
            "usage": None,
        }
        self.last_call = record
        body = post_json(url, payload, headers=headers, timeout=self._timeout)
        record["response"] = body
        record["usage"] = usage_from_response(body)
        parts = extract(body)
        record["raw_text"] = "\n".join(parts)
        return _validate_parts(schema, parts)


class OpenAICompatibleClient(_HttpClient):
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
        self.max_output_tokens = max_output_tokens

    def parse(self, *, system_text: str, user_text: str, schema: type[T], search: bool = False) -> T:
        if search:
            raise LoopError("The compatible provider has no web search tool.")
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
        return self._post(
            self._url,
            payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            extract=_chat_text,
            schema=schema,
        )


class OpenAIClient(_HttpClient):
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
        search: bool = False,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = _join(base_url, "responses")
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens
        self.max_output_tokens = max_output_tokens
        self.search = search

    def parse(self, *, system_text: str, user_text: str, schema: type[T], search: bool = False) -> T:
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
        if search:
            payload["tools"] = [dict(OPENAI_SEARCH_TOOL)]
        return self._post(
            self._url,
            payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            extract=_responses_text,
            schema=schema,
        )


class AnthropicClient(_HttpClient):
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
        search: bool = False,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = _anthropic_url(base_url)
        self._timeout = timeout
        self._max_output_tokens = 8192 if max_output_tokens is None else max_output_tokens
        self.max_output_tokens: int | None = self._max_output_tokens
        self.search = search

    def parse(self, *, system_text: str, user_text: str, schema: type[T], search: bool = False) -> T:
        payload: dict[str, object] = {
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
        if search:
            payload["tools"] = [dict(ANTHROPIC_SEARCH_TOOL)]
        return self._post(
            self._url,
            payload,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            extract=_anthropic_text,
            schema=schema,
        )


def post_json(
    url: str,
    payload: object,
    *,
    headers: dict[str, str],
    timeout: float,
    retries: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """POST JSON and return the JSON object the server sent back.

    A 429, a 5xx, or a dropped connection is retried with backoff, see
    :mod:`loopkit.retry`. Any other failure raises :class:`LoopError` at once
    with the status and the start of the server's message.
    """

    data = json.dumps(payload).encode("utf-8")

    def attempt() -> dict[str, object]:
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if len(detail) > 500:
                detail = detail[:500] + "..."
            suffix = f": {detail}" if detail else ""
            error = LoopError(f"The model API returned HTTP {exc.code}{suffix}")
            if exc.code in RETRY_STATUSES:
                header = getattr(exc, "headers", None)
                retry_after = parse_retry_after(header.get("Retry-After")) if header else None
                raise Transient(f"HTTP {exc.code}{suffix[:200]}", cause=error, retry_after=retry_after) from exc
            raise error from exc
        except urllib.error.URLError as exc:
            error = LoopError(f"Could not reach the model API at {url}: {exc.reason}")
            if isinstance(exc.reason, _TRANSIENT_NETWORK):
                raise Transient(f"connection dropped: {exc.reason}", cause=error) from exc
            raise error from exc
        except _TRANSIENT_NETWORK as exc:
            error = LoopError(f"The connection to {url} dropped: {exc}")
            raise Transient(f"connection dropped: {exc}", cause=error) from exc
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LoopError("The model API returned a response that is not JSON.") from exc
        if not isinstance(body, dict):
            raise LoopError("The model API returned a response that is not a JSON object.")
        return body

    return retry_call(attempt, what=f"POST {url}", retries=retries, sleep=sleep)


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
    cleaned = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_" for character in name
    )
    return (cleaned or "Result")[:64]


def _validate_parts(schema: type[T], parts: list[str]) -> T:
    """Validate the joined text; failing that, the last part on its own.

    With a web search tool the reply can hold text the model wrote before it
    searched, then the structured answer as the final block. The final block
    is what the schema describes.
    """

    joined = "\n".join(parts)
    if len(parts) <= 1:
        return _validate(schema, joined)
    try:
        return _validate(schema, joined)
    except LoopError:
        last = parts[-1]
        if not last.strip():
            raise
        return _validate(schema, last)


def _validate(schema: type[T], text: str) -> T:
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise LoopError("The loop expected a Pydantic schema.")
    try:
        return schema.model_validate_json(_strip_fence(text))
    except ValidationError as exc:
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        raise LoopError(
            f"The model returned JSON that does not match {schema.__name__}. "
            f"It said: {snippet!r}. Set --debug to log the full reply."
        ) from exc


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: body.rstrip().rfind("```")]
    return body.strip()


def _chat_text(payload: dict[str, object]) -> list[str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LoopError("The model returned no text.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LoopError("The model returned no text.")
    content = message.get("content")
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise LoopError("The model returned no text.")
    return [content]


def _responses_text(payload: dict[str, object]) -> list[str]:
    """Text of each message item, in order. Tool calls and reasoning are skipped."""

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
                if (
                    isinstance(block, dict)
                    and block.get("type") == "output_text"
                    and isinstance(block.get("text"), str)
                ):
                    parts.append(block["text"])
    if not parts:
        raise LoopError(_no_text_reason(payload))
    return parts


def _anthropic_text(payload: dict[str, object]) -> list[str]:
    """Each text block, in order. Tool use and search result blocks are skipped."""

    content = payload.get("content")
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
    if not parts:
        raise LoopError(_no_text_reason(payload))
    return parts


def _no_text_reason(payload: dict[str, object]) -> str:
    """Say why there was no text when the reply explains it."""

    for key in ("stop_reason", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value not in {"end_turn", "stop", "completed"}:
            return f"The model returned no text ({key}: {value})."
    details = payload.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason"):
        return f"The model returned no text (incomplete: {details['reason']})."
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for block in item.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "refusal":
                        return f"The model refused: {block.get('refusal', '')}"
    return "The model returned no text."
