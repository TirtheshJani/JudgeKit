"""Tests for AnthropicClient. Phase 1."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic

from judgekit.clients.anthropic_client import MODEL_ID, AnthropicClient
from judgekit.clients.base import JudgeResponse


def _make_mock_anthropic_response(
    text: str = "Option C is correct.",
    input_tokens: int = 55,
    output_tokens: int = 12,
) -> MagicMock:
    """Build a MagicMock that mimics an anthropic Messages response.

    The first content block is specced to anthropic.types.TextBlock so that
    isinstance() checks in the client implementation pass correctly.
    """
    mock_block = MagicMock(spec=anthropic.types.TextBlock)
    mock_block.text = text

    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    return mock_response


class TestAnthropicClientSuccess:
    def test_successful_call_yields_correct_judge_response_shape(self) -> None:
        """A successful SDK call returns a fully populated JudgeResponse."""
        mock_response = _make_mock_anthropic_response(
            text="Option C is correct.",
            input_tokens=55,
            output_tokens=12,
        )

        with patch("judgekit.clients.anthropic_client.anthropic.Anthropic") as MockAnthropic:
            mock_sdk = MockAnthropic.return_value
            mock_sdk.messages.create.return_value = mock_response

            client = AnthropicClient(api_key="test-key", model=MODEL_ID)
            response = client.judge("Which option is correct?")

        assert isinstance(response, JudgeResponse)
        assert response.text == "Option C is correct."
        assert response.prompt_tokens == 55
        assert response.completion_tokens == 12
        assert response.vendor == "anthropic"
        assert response.model == MODEL_ID
        assert response.latency_ms > 0

    def test_token_counts_come_from_usage_fields(self) -> None:
        """prompt_tokens maps to usage.input_tokens and completion_tokens to usage.output_tokens."""
        mock_response = _make_mock_anthropic_response(
            input_tokens=100,
            output_tokens=33,
        )

        with patch("judgekit.clients.anthropic_client.anthropic.Anthropic") as MockAnthropic:
            mock_sdk = MockAnthropic.return_value
            mock_sdk.messages.create.return_value = mock_response

            client = AnthropicClient(api_key="sk-ant-dummy")
            response = client.judge("How many tokens?")

        assert response.prompt_tokens == 100
        assert response.completion_tokens == 33

    def test_default_model_is_claude_sonnet_4_5(self) -> None:
        """Default model ID is the pinned claude-sonnet-4-5."""
        mock_response = _make_mock_anthropic_response()

        with patch("judgekit.clients.anthropic_client.anthropic.Anthropic") as MockAnthropic:
            mock_sdk = MockAnthropic.return_value
            mock_sdk.messages.create.return_value = mock_response

            client = AnthropicClient(api_key="sk-ant-dummy")
            response = client.judge("test")

        assert response.model == "claude-sonnet-4-5"

    def test_custom_model_is_forwarded_to_sdk(self) -> None:
        """A custom model string is forwarded to messages.create and appears in JudgeResponse."""
        mock_response = _make_mock_anthropic_response()

        with patch("judgekit.clients.anthropic_client.anthropic.Anthropic") as MockAnthropic:
            mock_sdk = MockAnthropic.return_value
            mock_sdk.messages.create.return_value = mock_response

            client = AnthropicClient(api_key="sk-ant-dummy", model="claude-opus-4-5")
            response = client.judge("test")

        assert response.model == "claude-opus-4-5"
        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-5"

    def test_max_tokens_forwarded_to_sdk(self) -> None:
        """max_tokens kwarg is passed through to the Anthropic SDK call."""
        mock_response = _make_mock_anthropic_response()

        with patch("judgekit.clients.anthropic_client.anthropic.Anthropic") as MockAnthropic:
            mock_sdk = MockAnthropic.return_value
            mock_sdk.messages.create.return_value = mock_response

            client = AnthropicClient(api_key="sk-ant-dummy")
            client.judge("A prompt.", max_tokens=1024)

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1024

    def test_sdk_receives_user_message_with_correct_content(self) -> None:
        """The messages list sent to the SDK has exactly one user message with the prompt."""
        mock_response = _make_mock_anthropic_response()

        with patch("judgekit.clients.anthropic_client.anthropic.Anthropic") as MockAnthropic:
            mock_sdk = MockAnthropic.return_value
            mock_sdk.messages.create.return_value = mock_response

            client = AnthropicClient(api_key="sk-ant-dummy")
            client.judge("Rate this response.")

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "Rate this response."}]

    def test_api_key_passed_to_sdk_constructor(self) -> None:
        """The api_key is forwarded to anthropic.Anthropic(api_key=...)."""
        mock_response = _make_mock_anthropic_response()

        with patch("judgekit.clients.anthropic_client.anthropic.Anthropic") as MockAnthropic:
            mock_sdk = MockAnthropic.return_value
            mock_sdk.messages.create.return_value = mock_response

            AnthropicClient(api_key="my-secret-key").judge("hi")

        MockAnthropic.assert_called_once_with(api_key="my-secret-key")
