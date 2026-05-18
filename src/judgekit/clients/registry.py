from __future__ import annotations

import os

from judgekit.clients.anthropic_client import AnthropicClient
from judgekit.clients.base import Judge
from judgekit.clients.openai_compat import OpenAICompatClient

_VENDOR_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "vllm": "http://localhost:8000/v1",
}

_VENDOR_ENV_KEYS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def build_judge(vendor: str, model: str, base_url: str | None = None) -> Judge:
    if vendor == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicClient(api_key=api_key, model=model)

    env_key = _VENDOR_ENV_KEYS.get(vendor, "")
    api_key = os.environ.get(env_key, "") if env_key else ""
    url = base_url or _VENDOR_BASE_URLS.get(vendor, "")
    return OpenAICompatClient(base_url=url, api_key=api_key, model=model, vendor=vendor)
