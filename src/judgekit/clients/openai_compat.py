from __future__ import annotations

import time

import httpx

from judgekit.clients.base import Judge, JudgeResponse, JudgeResponseError


class OpenAICompatClient(Judge):
    def __init__(self, base_url: str, api_key: str, model: str, vendor: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._vendor = vendor
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

    def judge(self, prompt: str, *, max_tokens: int = 512) -> JudgeResponse:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
        }

        t0 = time.monotonic()
        response = self._http.post(url, json=payload)
        latency_ms = (time.monotonic() - t0) * 1000.0

        # Let 429 / 5xx propagate so the retry decorator can handle them.
        response.raise_for_status()

        try:
            data = response.json()
        except Exception as exc:
            raise JudgeResponseError(f"Response body is not valid JSON: {exc}") from exc

        try:
            text = data["choices"][0]["message"]["content"]
            prompt_tokens: int = data["usage"]["prompt_tokens"]
            completion_tokens: int = data["usage"]["completion_tokens"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeResponseError(f"Unexpected response shape: {exc}") from exc

        return JudgeResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            vendor=self._vendor,
            model=self._model,
            latency_ms=latency_ms,
        )
