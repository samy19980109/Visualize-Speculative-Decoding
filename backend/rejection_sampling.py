"""Core speculative decoding rejection sampling logic."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schemas import TokenStatus


@dataclass
class ComparisonResult:
    """Result of comparing a draft token against the target at one position."""

    position: int
    status: TokenStatus
    draft_token: str
    final_token: str  # The token that survives (draft if accepted, target if resampled)
    final_token_id: int | None
    draft_logprob: float
    target_logprob: float | None
    acceptance_prob: float | None


@dataclass
class RoundResult:
    """Complete result of one speculation round."""

    comparisons: list[ComparisonResult]
    accepted_count: int
    bonus_token: str | None  # K+1th token if all K accepted
    bonus_token_id: int | None


def run_rejection_sampling(
    draft_tokens: list[dict],  # [{token_str, token_id, logprob}, ...]
    target_positions: list[dict],  # [{token_str, top_logprobs: {str: float}}, ...]
) -> RoundResult:
    """Run modified rejection sampling on one round of draft vs target.

    Algorithm (from Leviathan et al. 2023):
        For each position i = 0..K-1:
            1. If draft_token == target_token: ACCEPT (exact match)
            2. If draft_token in target's top_logprobs:
               acceptance_prob = min(1, exp(target_lp - draft_lp))
               If random() < acceptance_prob: ACCEPT
               Else: REJECT → use target's token (resample)
            3. If draft_token not in top-20: REJECT (negligible target prob)
        On first rejection, stop processing further positions.
        If all K accepted: bonus token = target's K+1th token.
    """
    comparisons: list[ComparisonResult] = []
    accepted_count = 0
    bonus_token = None
    bonus_token_id = None

    for i, draft in enumerate(draft_tokens):
        if i >= len(target_positions):
            break

        target = target_positions[i]
        draft_str = draft["token_str"]
        draft_lp = draft["logprob"]
        target_str = target["token_str"]
        target_top_lps = target["top_logprobs"]

        # Case 1: Exact match
        if draft_str == target_str:
            comparisons.append(
                ComparisonResult(
                    position=i,
                    status=TokenStatus.ACCEPTED,
                    draft_token=draft_str,
                    final_token=draft_str,
                    final_token_id=draft["token_id"],
                    draft_logprob=draft_lp,
                    target_logprob=target_top_lps.get(draft_str),
                    acceptance_prob=1.0,
                )
            )
            accepted_count += 1
            continue

        # Case 2: Draft token in target's top logprobs
        if draft_str in target_top_lps:
            target_lp = target_top_lps[draft_str]
            acceptance_prob = min(1.0, math.exp(target_lp - draft_lp))

            if random.random() < acceptance_prob:
                comparisons.append(
                    ComparisonResult(
                        position=i,
                        status=TokenStatus.ACCEPTED,
                        draft_token=draft_str,
                        final_token=draft_str,
                        final_token_id=draft["token_id"],
                        draft_logprob=draft_lp,
                        target_logprob=target_lp,
                        acceptance_prob=acceptance_prob,
                    )
                )
                accepted_count += 1
                continue
            else:
                # Reject and resample: use target's token
                comparisons.append(
                    ComparisonResult(
                        position=i,
                        status=TokenStatus.REJECTED,
                        draft_token=draft_str,
                        final_token=target_str,
                        final_token_id=draft["token_id"],
                        draft_logprob=draft_lp,
                        target_logprob=target_lp,
                        acceptance_prob=acceptance_prob,
                    )
                )
                # Add resampled token
                comparisons.append(
                    ComparisonResult(
                        position=i,
                        status=TokenStatus.RESAMPLED,
                        draft_token=draft_str,
                        final_token=target_str,
                        final_token_id=None,  # We don't have target token IDs from API
                        draft_logprob=draft_lp,
                        target_logprob=target_top_lps.get(target_str),
                        acceptance_prob=0.0,
                    )
                )
                break  # Stop on first rejection

        # Case 3: Not in top-20 → reject
        else:
            comparisons.append(
                ComparisonResult(
                    position=i,
                    status=TokenStatus.REJECTED,
                    draft_token=draft_str,
                    final_token=target_str,
                    final_token_id=draft["token_id"],
                    draft_logprob=draft_lp,
                    target_logprob=None,
                    acceptance_prob=0.0,
                )
            )
            comparisons.append(
                ComparisonResult(
                    position=i,
                    status=TokenStatus.RESAMPLED,
                    draft_token=draft_str,
                    final_token=target_str,
                    final_token_id=None,
                    draft_logprob=draft_lp,
                    target_logprob=target_top_lps.get(target_str),
                    acceptance_prob=0.0,
                )
            )
            break

    # If all K accepted and we have a K+1th position, that's the bonus token
    if accepted_count == len(draft_tokens) and len(target_positions) > len(
        draft_tokens
    ):
        bonus_pos = target_positions[len(draft_tokens)]
        bonus_token = bonus_pos["token_str"]
        bonus_token_id = None

    return RoundResult(
        comparisons=comparisons,
        accepted_count=accepted_count,
        bonus_token=bonus_token,
        bonus_token_id=bonus_token_id,
    )
