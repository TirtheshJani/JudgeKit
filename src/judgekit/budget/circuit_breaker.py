from __future__ import annotations

from judgekit.budget.pricing import lookup
from judgekit.budget.tracker import BudgetTracker
from judgekit.clients.base import BudgetExceededError


class BudgetCircuitBreaker:
    def __init__(self, tracker: BudgetTracker) -> None:
        self._tracker = tracker

    def check(self, vendor: str, model: str, prompt_tokens: int, max_tokens: int) -> None:
        """Raise BudgetExceededError if the estimated call cost would exceed the cap."""
        pricing = lookup(vendor, model)
        est = (
            prompt_tokens / 1000 * pricing.input_per_1k + max_tokens / 1000 * pricing.output_per_1k
        )
        if self._tracker.total_usd() + est > self._tracker.cap_usd:
            raise BudgetExceededError(
                f"Estimated call cost ${est:.4f} would push total past "
                f"cap ${self._tracker.cap_usd:.2f}"
            )
