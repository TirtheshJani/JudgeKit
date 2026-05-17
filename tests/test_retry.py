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


class Test5xxRetries:
    """Test 5: 5xx errors trigger retry same as 429."""

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_500_triggers_retry(self, mock_uniform, mock_sleep):
        err = make_http_error(500)
        fn = MagicMock(side_effect=[err, "ok"])
        decorated = with_retry(fn)
        result = decorated()
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_502_triggers_retry(self, mock_uniform, mock_sleep):
        err = make_http_error(502)
        fn = MagicMock(side_effect=[err, "ok"])
        decorated = with_retry(fn)
        result = decorated()
        assert result == "ok"
        assert fn.call_count == 2

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_503_triggers_retry(self, mock_uniform, mock_sleep):
        err = make_http_error(503)
        fn = MagicMock(side_effect=[err, "ok"])
        decorated = with_retry(fn)
        result = decorated()
        assert result == "ok"
        assert fn.call_count == 2

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_504_triggers_retry(self, mock_uniform, mock_sleep):
        err = make_http_error(504)
        fn = MagicMock(side_effect=[err, "ok"])
        decorated = with_retry(fn)
        result = decorated()
        assert result == "ok"
        assert fn.call_count == 2


class TestNonRetriable4xx:
    """Test 6: Non-retriable 4xx raises immediately without retry."""

    @patch("time.sleep")
    def test_400_raises_immediately_no_retry(self, mock_sleep):
        err = make_http_error(400)
        fn = MagicMock(side_effect=err)
        decorated = with_retry(fn)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            decorated()
        assert exc_info.value.response.status_code == 400
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_403_raises_immediately_no_retry(self, mock_sleep):
        err = make_http_error(403)
        fn = MagicMock(side_effect=err)
        decorated = with_retry(fn)
        with pytest.raises(httpx.HTTPStatusError):
            decorated()
        fn.assert_called_once()
        mock_sleep.assert_not_called()


class TestExponentialBackoffAmounts:
    """Test 7: Verify exponential backoff sleep amounts per attempt."""

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_attempt_0_backoff(self, mock_uniform, mock_sleep):
        """On attempt 0: sleep = BASE_BACKOFF * 2^0 + jitter = 1.0 + 0.0 = 1.0"""
        err = make_http_error(429)
        fn = MagicMock(side_effect=[err, "ok"])
        decorated = with_retry(fn)
        decorated()
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep")
    @patch("random.uniform", return_value=0.0)
    def test_attempt_1_backoff(self, mock_uniform, mock_sleep):
        """On attempt 1: sleep = BASE_BACKOFF * 2^1 + jitter = 2.0 + 0.0 = 2.0"""
        err = make_http_error(429)
        fn = MagicMock(side_effect=[err, err, "ok"])
        decorated = with_retry(fn)
        decorated()
        assert mock_sleep.call_count == 2
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0  # attempt 0: 1.0 * 2^0 = 1.0
        assert calls[1][0][0] == 2.0  # attempt 1: 1.0 * 2^1 = 2.0
