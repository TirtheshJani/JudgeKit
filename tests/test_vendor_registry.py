from __future__ import annotations

import pytest

from judgekit.clients.anthropic_client import AnthropicClient
from judgekit.clients.openai_compat import OpenAICompatClient
from judgekit.clients.registry import build_judge


def test_build_groq_returns_openai_compat_client() -> None:
    client = build_judge("groq", "llama-3.3-70b-versatile")
    assert isinstance(client, OpenAICompatClient)


def test_build_groq_uses_correct_base_url() -> None:
    client = build_judge("groq", "llama-3.3-70b-versatile")
    assert isinstance(client, OpenAICompatClient)
    assert client._base_url == "https://api.groq.com/openai/v1"


def test_build_cerebras_uses_correct_base_url() -> None:
    client = build_judge("cerebras", "llama-3.3-70b")
    assert isinstance(client, OpenAICompatClient)
    assert client._base_url == "https://api.cerebras.ai/v1"


def test_build_sambanova_uses_correct_base_url() -> None:
    client = build_judge("sambanova", "DeepSeek-R1")
    assert isinstance(client, OpenAICompatClient)
    assert client._base_url == "https://api.sambanova.ai/v1"


def test_build_openrouter_uses_correct_base_url() -> None:
    client = build_judge("openrouter", "some-model")
    assert isinstance(client, OpenAICompatClient)
    assert client._base_url == "https://openrouter.ai/api/v1"


def test_build_judge_base_url_override() -> None:
    client = build_judge("groq", "llama-3.3-70b-versatile", base_url="http://custom:9999/v1")
    assert isinstance(client, OpenAICompatClient)
    assert client._base_url == "http://custom:9999/v1"


def test_build_anthropic_returns_anthropic_client() -> None:
    client = build_judge("anthropic", "claude-sonnet-4-5")
    assert isinstance(client, AnthropicClient)


def test_build_judge_sets_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    client = build_judge("groq", "llama-3.3-70b-versatile")
    assert isinstance(client, OpenAICompatClient)
    assert client._api_key == "test-key-123"


def test_build_vllm_no_key_required() -> None:
    client = build_judge("vllm", "Qwen/Qwen3-8B")
    assert isinstance(client, OpenAICompatClient)
    assert client._base_url == "http://localhost:8000/v1"
    assert client._api_key == ""
