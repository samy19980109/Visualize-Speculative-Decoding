"""Cerebras Cloud API wrapper for target model verification."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger(__name__)


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
        prompt_text: str,
        k: int,
    ) -> VerificationResult:
        """Verify draft tokens by generating K+1 tokens from the target model.

        Uses the /v1/completions endpoint (not chat) to send the raw prompt
        text including the chat template. This ensures the target model
        continues from exactly the right position, avoiding issues with
        assistant message prefilling not being supported.

        Args:
            prompt_text: Full raw text (chat template + generated text so far).
            k: Number of draft tokens to verify.

        Returns:
            VerificationResult with per-position logprob info.
        """
        t0 = time.perf_counter()

        logger.info(f"  TARGET prompt ends with: {repr(prompt_text[-80:])}")

        response = await self.client.completions.create(
            model=self.model,
            prompt=prompt_text,
            logprobs=20,
            max_tokens=k + 1,
            temperature=0.01,  # Near-greedy; Cerebras requires >0 with logprobs
        )

        elapsed = (time.perf_counter() - t0) * 1000

        positions: list[TargetTokenInfo] = []
        logprobs_data = response.choices[0].logprobs

        if logprobs_data and logprobs_data.tokens:
            for i in range(len(logprobs_data.tokens)):
                token_str = logprobs_data.tokens[i]
                token_logprob = (
                    logprobs_data.token_logprobs[i]
                    if logprobs_data.token_logprobs[i] is not None
                    else 0.0
                )
                top_lp_map = (
                    dict(logprobs_data.top_logprobs[i])
                    if logprobs_data.top_logprobs and logprobs_data.top_logprobs[i]
                    else {}
                )

                entropy = _approx_entropy_from_top_logprobs(list(top_lp_map.values()))

                positions.append(TargetTokenInfo(
                    token_str=token_str,
                    token_logprob=token_logprob,
                    top_logprobs=top_lp_map,
                    entropy=entropy,
                ))

        logger.info(
            f"  TARGET returned {len(positions)} positions: "
            f"{[p.token_str for p in positions[:5]]}..."
        )

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
