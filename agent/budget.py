"""Per-run cost. Local models have no token bill, but compute isn't free and a
runaway loop burns it all the same — so cost is wall time x an hourly rate,
plus token prices if a hosted model is swapped in."""
from __future__ import annotations

import os
from dataclasses import dataclass

USD_PER_HOUR = float(os.environ.get("AGENT_COMPUTE_USD_PER_HOUR", "0.35"))
USD_PER_M_IN = float(os.environ.get("AGENT_USD_PER_M_INPUT", "0"))
USD_PER_M_OUT = float(os.environ.get("AGENT_USD_PER_M_OUTPUT", "0"))
COST_CAP_USD = float(os.environ.get("AGENT_COST_CAP_USD", "0.01"))
TOKEN_CAP = int(os.environ.get("AGENT_TOKEN_CAP", "40000"))


@dataclass
class Budget:
    cap_usd: float = COST_CAP_USD
    cap_tokens: int = TOKEN_CAP
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, seconds: float, input_tokens: int, output_tokens: int) -> None:
        self.seconds += seconds
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def usd(self) -> float:
        return (self.seconds / 3600 * USD_PER_HOUR
                + self.input_tokens / 1e6 * USD_PER_M_IN
                + self.output_tokens / 1e6 * USD_PER_M_OUT)

    def exceeded(self) -> bool:
        return self.usd > self.cap_usd or self.input_tokens + self.output_tokens > self.cap_tokens
