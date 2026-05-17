from __future__ import annotations

from judgekit.clients.base import Judge, JudgeResponse


class OpenAICompatClient(Judge):
    def __init__(self, base_url: str, api_key: str, model: str, vendor: str) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._vendor = vendor

    def judge(self, prompt: str, *, max_tokens: int = 512) -> JudgeResponse:
        raise NotImplementedError
