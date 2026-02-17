"""Rolling-window KPI computation for speculative decoding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class RoundStats:
    accepted: int
    total: int
    tokens_produced: int  # accepted + bonus (if any)
    draft_latency_ms: float
    verify_latency_ms: float
    round_time_ms: float
    k: int = 1  # number of draft tokens in this round


class MetricsTracker:
    def __init__(self, window_size: int = 50):
        self._window: deque[RoundStats] = deque(maxlen=window_size)
        self._total_tokens: int = 0
        self._total_accepted: int = 0
        self._total_drafted: int = 0
        self._total_rounds: int = 0
        self._generation_start_ms: float | None = None

    def record_round(self, stats: RoundStats) -> None:
        self._window.append(stats)
        self._total_tokens += stats.tokens_produced
        self._total_accepted += stats.accepted
        self._total_drafted += stats.total
        self._total_rounds += 1

    def set_start_time(self, start_ms: float) -> None:
        self._generation_start_ms = start_ms

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_rounds(self) -> int:
        return self._total_rounds

    def acceptance_rate(self) -> float:
        """Windowed acceptance rate."""
        total_accepted = sum(r.accepted for r in self._window)
        total_drafted = sum(r.total for r in self._window)
        return total_accepted / total_drafted if total_drafted > 0 else 0.0

    def overall_acceptance_rate(self) -> float:
        return self._total_accepted / self._total_drafted if self._total_drafted > 0 else 0.0

    def effective_tps(self) -> float:
        """Tokens per second based on windowed rounds."""
        if not self._window:
            return 0.0
        total_time_s = sum(r.round_time_ms for r in self._window) / 1000
        total_tokens = sum(r.tokens_produced for r in self._window)
        return total_tokens / total_time_s if total_time_s > 0 else 0.0

    def baseline_tps(self) -> float:
        """Estimated baseline autoregressive TPS.

        In autoregressive mode, each token requires one full API call.
        A verify call checks K+1 tokens at once, so autoregressive cost
        per token is verify_latency / (k + 1).
        """
        if not self._window:
            return 0.0
        # Each verify call processes k+1 positions; in autoregressive mode
        # each token would take verify_latency_ms / (k+1)
        total_ar_time_ms = sum(
            r.verify_latency_ms / (r.k + 1) for r in self._window
        )
        total_ar_tokens = len(self._window)  # 1 token per "round" in AR mode
        return (total_ar_tokens / total_ar_time_ms) * 1000 if total_ar_time_ms > 0 else 0.0

    def speedup(self) -> float:
        baseline = self.baseline_tps()
        effective = self.effective_tps()
        return effective / baseline if baseline > 0 else 1.0

    def avg_draft_latency(self) -> float:
        if not self._window:
            return 0.0
        return sum(r.draft_latency_ms for r in self._window) / len(self._window)

    def avg_verify_latency(self) -> float:
        if not self._window:
            return 0.0
        return sum(r.verify_latency_ms for r in self._window) / len(self._window)
