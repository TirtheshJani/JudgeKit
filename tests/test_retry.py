"""Tests for the retry decorator. Phase 1."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from judgekit.clients.base import RateLimitExhausted
from judgekit.retry import MAX_RETRIES, with_retry


def make_http_error(status_code: int, retry_after: str | None = None) -> httpx.HTTPStatusError:
    headers = {}
    if retry_after:
        headers["Retry-After"] = retry_after
    response = httpx.Response(status_code, headers=headers)
    return httpx.HTTPStatusError(
        "error", request=httpx.Request("POST", "http://test"), response=response
    )


class TestHappyPath:
    """Test 1: Happy path - function succeeds on first call."""

    def test_success_on_first_call_returns_value(self):
        fn = MagicMock(return_value="ok")
        decorated = with_retry(fn)
        result = decorated("arg1", key="val")
        assert result == "ok"
        fn.assert_called_once_with("arg1", key="val")


class Test429ThenSuccess:
    """Test 2: 429 then success - function is called twice, sleep called once."""

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_429_then_success_calls_fn_twice(self, mock_uniform, mock_sleep):
        err = make_http_error(429)
        fn = MagicMock(side_effect=[err, "ok"])
        decorated = with_retry(fn)
        result = decorated()
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()


class TestRetryAfterHeader:
    """Test 3: Retry-After header overrides exponential backoff."""

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_retry_after_header_sleeps_specified_duration(self, mock_uniform, mock_sleep):
        err = make_http_error(429, retry_after="2")
        fn = MagicMock(side_effect=[err, "ok"])
        decorated = with_retry(fn)
        result = decorated()
        assert result == "ok"
        mock_sleep.assert_called_once_with(2.0)


class TestExhaustRetries:
    """Test 4: All retries exhausted raises RateLimitExhausted."""

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_always_429_raises_rate_limit_exhausted(self, mock_uniform, mock_sleep):
        err = make_http_error(429)
        fn = MagicMock(side_effect=err)
        decorated = with_retry(fn)
        with pytest.raises(RateLimitExhausted):
            decorated()
        assert fn.call_count == MAX_RETRIES
