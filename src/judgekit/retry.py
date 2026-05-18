from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import httpx

from judgekit.clients.base import RateLimitExhausted

P = ParamSpec("P")
R = TypeVar("R")

MAX_RETRIES = 5
BASE_BACKOFF = 1.0
JITTER = 0.5


def with_retry(fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = BASE_BACKOFF * (2**attempt) + random.uniform(0, JITTER)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
        raise RateLimitExhausted(f"Exhausted {MAX_RETRIES} retries") from last_exc

    return wrapper
