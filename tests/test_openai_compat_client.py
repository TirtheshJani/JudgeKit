"""Tests for OpenAICompatClient. Phase 1."""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from judgekit.clients.base import JudgeResponse, JudgeResponseError
from judgekit.clients.openai_compat import OpenAICompatClient
from tests.fixtures.mock_api_responses import (
    openai_compat_200_bytes,
    openai_compat_429,
    openai_compat_500,
    openai_compat_malformed_json,
    openai_compat_missing_choices,
)

BASE_URL = "https://api.example.com/v1"
API_KEY = "test-key-abc"
MODEL = "llama-3.3-70b-versatile"
VENDOR = "groq"


def make_client() -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        vendor=VENDOR,
    )


class TestOpenAICompatClientSuccess:
    def test_successful_200_returns_correct_judge_response_shape(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """A 200 response with valid JSON yields a fully populated JudgeResponse."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            status_code=200,
            content=openai_compat_200_bytes(
                content="The answer is B.",
                prompt_tokens=42,
                completion_tokens=17,
                model=MODEL,
            ),
            headers={"Content-Type": "application/json"},
        )

        client = make_client()
        response = client.judge("Which answer is correct?")

        assert isinstance(response, JudgeResponse)
        assert response.text == "The answer is B."
        assert response.prompt_tokens == 42
        assert response.completion_tokens == 17
        assert response.vendor == VENDOR
        assert response.model == MODEL
        assert response.latency_ms > 0

    def test_vendor_and_model_fields_match_constructor_args(self, httpx_mock: HTTPXMock) -> None:
        """vendor and model on JudgeResponse reflect what was passed to the constructor."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            status_code=200,
            content=openai_compat_200_bytes(model="custom-model"),
            headers={"Content-Type": "application/json"},
        )
        client = OpenAICompatClient(
            base_url=BASE_URL,
            api_key=API_KEY,
            model="custom-model",
            vendor="cerebras",
        )
        response = client.judge("hello")

        assert response.vendor == "cerebras"
        assert response.model == "custom-model"

    def test_request_sends_bearer_auth_and_correct_body(self, httpx_mock: HTTPXMock) -> None:
        """The outgoing request contains the Authorization header and expected JSON body."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            status_code=200,
            content=openai_compat_200_bytes(),
            headers={"Content-Type": "application/json"},
        )
        client = make_client()
        client.judge("Tell me something.", max_tokens=256)

        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        req = requests[0]

        assert req.headers["Authorization"] == f"Bearer {API_KEY}"

        import json

        body = json.loads(req.content)
        assert body["model"] == MODEL
        assert body["max_tokens"] == 256
        assert body["temperature"] == 0
        assert body["top_p"] == 1
        assert body["messages"] == [{"role": "user", "content": "Tell me something."}]


class TestOpenAICompatClientErrors:
    def test_malformed_json_raises_judge_response_error(self, httpx_mock: HTTPXMock) -> None:
        """Non-JSON response body raises JudgeResponseError."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            status_code=200,
            content=openai_compat_malformed_json(),
            headers={"Content-Type": "application/json"},
        )
        client = make_client()
        with pytest.raises(JudgeResponseError):
            client.judge("test prompt")

    def test_missing_choices_field_raises_judge_response_error(self, httpx_mock: HTTPXMock) -> None:
        """Response missing 'choices' key raises JudgeResponseError."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            status_code=200,
            content=openai_compat_missing_choices(),
            headers={"Content-Type": "application/json"},
        )
        client = make_client()
        with pytest.raises(JudgeResponseError):
            client.judge("test prompt")

    def test_429_raises_httpx_http_status_error(self, httpx_mock: HTTPXMock) -> None:
        """A 429 response propagates as httpx.HTTPStatusError (not caught by the client)."""
        import json

        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            status_code=429,
            content=json.dumps(openai_compat_429()).encode(),
            headers={"Content-Type": "application/json"},
        )
        client = make_client()
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.judge("test prompt")

        assert exc_info.value.response.status_code == 429

    def test_500_raises_httpx_http_status_error(self, httpx_mock: HTTPXMock) -> None:
        """A 500 response propagates as httpx.HTTPStatusError (not caught by the client)."""
        import json

        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            status_code=500,
            content=json.dumps(openai_compat_500()).encode(),
            headers={"Content-Type": "application/json"},
        )
        client = make_client()
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.judge("test prompt")

        assert exc_info.value.response.status_code == 500
