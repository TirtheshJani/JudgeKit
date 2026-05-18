from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgeResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    vendor: str
    model: str
    latency_ms: float


class RateLimitExhausted(Exception):
    pass


class JudgeResponseError(Exception):
    pass


class BudgetExceededError(Exception):
    pass


class Judge(ABC):
    @abstractmethod
    def judge(self, prompt: str, *, max_tokens: int = 512) -> JudgeResponse: ...
