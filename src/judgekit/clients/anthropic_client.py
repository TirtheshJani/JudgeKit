from __future__ import annotations

import time

import anthropic

from judgekit.clients.base import Judge, JudgeResponse, JudgeResponseError

MODEL_ID = "claude-sonnet-4-5"


class AnthropicClient(Judge):
    def __init__(self, api_key: str, model: str = MODEL_ID) -> None:
        self._api_key = api_key
        self._model = model
        self._sdk = anthropic.Anthropic(api_key=api_key)

    def judge(self, prompt: str, *, max_tokens: int = 512) -> JudgeResponse:
        t0 = time.monotonic()
        response = self._sdk.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        first_block = response.content[0]
        if not isinstance(first_block, anthropic.types.TextBlock):
            raise JudgeResponseError(
                f"Expected TextBlock as first content block, got {type(first_block).__name__}"
            )
        text: str = first_block.text
        prompt_tokens: int = response.usage.input_tokens
        completion_tokens: int = response.usage.output_tokens

        return JudgeResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            vendor="anthropic",
            model=self._model,
            latency_ms=latency_ms,
        )
