from __future__ import annotations

from judgekit.clients.base import Judge, JudgeResponse

MODEL_ID = "claude-sonnet-4-5"


class AnthropicClient(Judge):
    def __init__(self, api_key: str, model: str = MODEL_ID) -> None:
        self._api_key = api_key
        self._model = model

    def judge(self, prompt: str, *, max_tokens: int = 512) -> JudgeResponse:
        raise NotImplementedError
