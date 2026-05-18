from __future__ import annotations

import os
import threading
from dataclasses import dataclass


@dataclass
class VendorSpend:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class BudgetTracker:
    def __init__(self, cap_usd: float | None = None) -> None:
        env_cap = os.environ.get("JUDGEKIT_ANTHROPIC_BUDGET_USD")
        self._cap = cap_usd if cap_usd is not None else (float(env_cap) if env_cap else 25.0)
        self._lock = threading.Lock()
        self._vendors: dict[str, VendorSpend] = {}

    @property
    def cap_usd(self) -> float:
        return self._cap

    def record(
        self,
        vendor: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        with self._lock:
            if vendor not in self._vendors:
                self._vendors[vendor] = VendorSpend()
            s = self._vendors[vendor]
            s.input_tokens += input_tokens
            s.output_tokens += output_tokens
            s.cost_usd += cost_usd

    def total_usd(self) -> float:
        with self._lock:
            return sum(s.cost_usd for s in self._vendors.values())

    def vendor_summary(self) -> dict[str, VendorSpend]:
        with self._lock:
            return dict(self._vendors)
