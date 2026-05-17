"""Canned HTTP responses for pytest-httpx: 200/429/5xx per vendor shape."""

from __future__ import annotations

import json


def openai_compat_200(
    content: str = "The answer is B.",
    prompt_tokens: int = 42,
    completion_tokens: int = 17,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """Return a well-formed OpenAI-compatible chat completion response body."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1716000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def openai_compat_200_bytes(**kwargs) -> bytes:
    return json.dumps(openai_compat_200(**kwargs)).encode()


def openai_compat_missing_choices() -> bytes:
    """Response body where 'choices' key is absent."""
    body = {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    return json.dumps(body).encode()


def openai_compat_malformed_json() -> bytes:
    """Not valid JSON."""
    return b"{ this is not json }"


def openai_compat_429() -> dict:
    return {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}


def openai_compat_500() -> dict:
    return {"error": {"message": "Internal server error"}}
