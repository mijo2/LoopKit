"""Retry a model call after a rate limit or a server error.

One policy is shared by the HTTP clients and the xAI SDK client. A call is
retried when the provider says it is busy (HTTP 429, 529) or failed on its
side (HTTP 5xx), and when the connection dropped before a reply arrived. A
bad request, a bad key, or a missing model is not retried: another attempt
would get the same answer.
"""

from __future__ import annotations

import email.utils
import os
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

from loopkit.debuglog import logger

T = TypeVar("T")

# Attempts after the first one. ``LOOPKIT_MAX_RETRIES`` overrides this.
DEFAULT_MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
# A Retry-After longer than this is not waited for; the backoff is used instead.
MAX_RETRY_AFTER_SECONDS = 60.0

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})


class Transient(Exception):
    """Raised by a call to say the attempt failed in a way worth retrying.

    ``retry_after`` is the number of seconds the provider asked for, when it
    said. ``cause`` is the error to raise when no attempts are left.
    """

    def __init__(self, message: str, *, cause: Exception, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.cause = cause
        self.retry_after = retry_after


def max_retries() -> int:
    raw = os.environ.get("LOOPKIT_MAX_RETRIES", "").strip()
    if not raw:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_RETRIES
    return max(0, value)


def retry_call(
    attempt: Callable[[], T],
    *,
    what: str,
    retries: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``attempt`` until it returns, or until the retry budget is spent.

    ``attempt`` raises :class:`Transient` to ask for another try. Any other
    exception propagates at once. The wait doubles from one second, with
    jitter, and is capped at :data:`MAX_BACKOFF_SECONDS`. When the provider
    sent ``Retry-After`` and it is not unreasonable, that wait is used instead.
    """

    budget = max_retries() if retries is None else max(0, retries)
    delay = INITIAL_BACKOFF_SECONDS
    for done in range(budget + 1):
        try:
            return attempt()
        except Transient as exc:
            remaining = budget - done
            if remaining <= 0:
                logger.warning("%s failed after %d attempts: %s", what, done + 1, exc)
                raise exc.cause from exc
            wait = delay * (0.5 + random.random())
            if exc.retry_after is not None and 0 <= exc.retry_after <= MAX_RETRY_AFTER_SECONDS:
                wait = exc.retry_after
            logger.warning(
                "%s: %s. Retrying in %.1fs (%d of %d retries left).",
                what,
                exc,
                wait,
                remaining,
                budget,
            )
            sleep(wait)
            delay = min(delay * 2, MAX_BACKOFF_SECONDS)
    raise AssertionError("unreachable")


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a ``Retry-After`` header, which is a delay or an HTTP date."""

    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())
