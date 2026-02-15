"""Cerebras Cloud API wrapper for target model verification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .config import settings


@dataclass
class TargetTokenInfo:
    """Verification info for a single position from the target model."""
    token_str: str
    token_logprob: float
    top_logprobs: dict[str, float]  # token_str → logprob
    entropy: float


@dataclass
class VerificationResult:
    """Result of verifying K draft tokens + 1 bonus position."""
    positions: list[TargetTokenInfo]
    elapsed_ms: float


class TargetModel:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key,
        )

    async def verify_tokens(
        self,
        messages: list[dict],
        draft_tokens: list[str],
        k: int,
    ) -> VerificationResult:
        """Send context + draft tokens to Cerebras for batch verification.

        We ask the target model to generate K+1 tokens from the same context,
        with logprobs enabled, so we can compare its distribution against the
        draft model's choices.

        Args:
            messages: Chat messages (system + user + assistant context so far).
            draft_tokens: The K draft token strings to verify.
            k: Number of draft tokens (len(draft_tokens)).

        Returns:
            VerificationResult with per-position logprob info.
        """
        import time

        t0 = time.perf_counter()

        # Build the prompt: context up to where drafting started,
        # then let the target freely generate K+1 tokens
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            top_logprobs=20,
            max_completion_tokens=k + 1,
            temperature=0.01,  # Near-greedy; Cerebras requires >0 with top_logprobs
        )

        elapsed = (time.perf_counter() - t0) * 1000

        positions: list[TargetTokenInfo] = []
        content_logprobs = response.choices[0].logprobs.content

        for lp_entry in content_logprobs:
            # Build top logprobs map
            top_lp_map: dict[str, float] = {}
            for tlp in lp_entry.top_logprobs:
                top_lp_map[tlp.token] = tlp.logprob

            # Approximate entropy from top-20 logprobs
            entropy = _approx_entropy_from_top_logprobs(
                [tlp.logprob for tlp in lp_entry.top_logprobs]
            )

            positions.append(TargetTokenInfo(
                token_str=lp_entry.token,
                token_logprob=lp_entry.logprob,
                top_logprobs=top_lp_map,
                entropy=entropy,
            ))

        return VerificationResult(positions=positions, elapsed_ms=elapsed)


def _approx_entropy_from_top_logprobs(logprobs: list[float]) -> float:
    """Approximate Shannon entropy from top-N logprobs.

    Since we only have the top-20, we normalize them to form a
    proper distribution and compute entropy over that subset.
    """
    if not logprobs:
        return 0.0

    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for p in probs:
        p_norm = p / total
        if p_norm > 0:
            entropy -= p_norm * math.log(p_norm)
    return entropy
