"""Orchestrator: draft → verify → compare → emit events loop."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from .draft_model import DraftModel, DraftToken
from .metrics import MetricsTracker, RoundStats
from .rejection_sampling import run_rejection_sampling
from .schemas import (
    DraftTokenEvent,
    ErrorEvent,
    GenerationDoneEvent,
    MetricsEvent,
    TokenStatus,
    TopToken,
    VerifyResultEvent,
)
from .target_model import TargetModel

logger = logging.getLogger(__name__)


class Speculator:
    def __init__(self, draft: DraftModel, target: TargetModel):
        self.draft = draft
        self.target = target

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        k: int = 8,
    ) -> AsyncGenerator:
        """Run speculative decoding loop, yielding events for the frontend."""
        metrics = MetricsTracker()
        metrics.set_start_time(time.perf_counter() * 1000)

        # Build initial context
        logger.info("Applying chat template...")
        context_ids = self.draft.apply_chat_template(prompt)
        logger.info(f"Context has {len(context_ids)} tokens")

        generated_text_so_far = ""
        generated_token_ids: list[
            int
        ] = []  # Track token IDs directly to avoid tokenizer drift
        current_round = 0
        total_tokens_produced = 0

        # Messages for target model API
        messages = [{"role": "user", "content": prompt}]

        try:
            while total_tokens_produced < max_tokens:
                current_round += 1
                round_start = time.perf_counter()

                # --- Phase 1: Draft K tokens locally ---
                # Use accumulated token IDs directly to avoid tokenizer drift
                full_context_ids = context_ids + generated_token_ids

                logger.info(
                    f"Round {current_round}: Drafting {k} tokens (context: {len(full_context_ids)} ids, generated_ids: {len(generated_token_ids)})..."
                )
                logger.info(
                    f"  DRAFT CONTEXT generated_token_ids: {generated_token_ids}"
                )
                draft_start = time.perf_counter()
                draft_tokens: list[DraftToken] = await asyncio.to_thread(
                    self.draft.generate_draft_tokens,
                    full_context_ids,
                    k,
                    temperature,
                )
                draft_elapsed = (time.perf_counter() - draft_start) * 1000
                logger.info(
                    f"Round {current_round}: Drafted {len(draft_tokens)} tokens in {draft_elapsed:.0f}ms: "
                    f"{[dt.token_str for dt in draft_tokens]}"
                )
                # DEBUG: Show what the draft model is generating
                for i, dt in enumerate(draft_tokens):
                    logger.info(
                        f"  DRAFTED pos={i} token='{dt.token_str}' id={dt.token_id}"
                    )

                # Emit draft token events with staggered timing
                for i, dt in enumerate(draft_tokens):
                    yield DraftTokenEvent(
                        round=current_round,
                        position=i,
                        token=dt.token_str,
                        token_id=dt.token_id,
                        logprob=dt.logprob,
                        entropy=dt.entropy,
                        top_tokens=[
                            TopToken(token=t, logprob=lp) for t, lp in dt.top_tokens
                        ],
                        draft_time_ms=dt.elapsed_ms,
                    )
                    await asyncio.sleep(0.05)  # 50ms stagger for animation

                # --- Phase 2: Verify via Cerebras API ---
                logger.info(f"Round {current_round}: Verifying via Cerebras API...")
                verify_messages = list(messages)
                if generated_text_so_far:
                    verify_messages.append(
                        {
                            "role": "assistant",
                            "content": generated_text_so_far,
                        }
                    )
                logger.info(f"  VERIFY MESSAGES: {verify_messages}")

                draft_strs = [dt.token_str for dt in draft_tokens]
                verification = await self.target.verify_tokens(
                    messages=verify_messages,
                    draft_tokens=draft_strs,
                    k=k,
                )
                logger.info(
                    f"Round {current_round}: Verification returned {len(verification.positions)} "
                    f"positions in {verification.elapsed_ms:.0f}ms"
                )

                # --- Phase 3: Rejection sampling ---
                draft_dicts = [
                    {
                        "token_str": dt.token_str,
                        "token_id": dt.token_id,
                        "logprob": dt.logprob,
                    }
                    for dt in draft_tokens
                ]
                target_dicts = [
                    {
                        "token_str": pos.token_str,
                        "top_logprobs": pos.top_logprobs,
                    }
                    for pos in verification.positions
                ]

                round_result = run_rejection_sampling(draft_dicts, target_dicts)
                logger.info(
                    f"Round {current_round}: Accepted {round_result.accepted_count}/{len(draft_tokens)}, "
                    f"bonus={'yes' if round_result.bonus_token else 'no'}"
                )
                # DEBUG: Log all comparisons
                for comp in round_result.comparisons:
                    logger.info(
                        f"  COMPARISON pos={comp.position} status={comp.status} draft='{comp.draft_token}' final='{comp.final_token}' final_id={comp.final_token_id}"
                    )

                # --- Phase 4: Emit verify result events ---
                tokens_this_round: list[str] = []
                token_ids_this_round: list[int] = []
                added_positions: set[int] = (
                    set()
                )  # Track which positions we've already added

                for comp in round_result.comparisons:
                    # Skip if we've already added a token for this position
                    if comp.position in added_positions:
                        logger.info(
                            f"    SKIPPING position {comp.position} - already added"
                        )
                        continue
                    target_pos = (
                        verification.positions[comp.position]
                        if comp.position < len(verification.positions)
                        else None
                    )

                    yield VerifyResultEvent(
                        round=current_round,
                        position=comp.position,
                        token=comp.final_token,
                        token_id=comp.final_token_id or 0,
                        status=comp.status,
                        draft_logprob=comp.draft_logprob,
                        target_logprob=comp.target_logprob,
                        acceptance_prob=comp.acceptance_prob,
                        target_entropy=target_pos.entropy if target_pos else None,
                        target_top_tokens=[
                            TopToken(token=t, logprob=lp)
                            for t, lp in list(target_pos.top_logprobs.items())[:5]
                        ]
                        if target_pos
                        else [],
                        verify_time_ms=verification.elapsed_ms,
                    )
                    await asyncio.sleep(0.08)  # 80ms stagger

                    # Collect accepted/resampled tokens for context
                    # Note: RESAMPLED replaces REJECTED, so only count RESAMPLED (not REJECTED)
                    if comp.status in (TokenStatus.ACCEPTED, TokenStatus.RESAMPLED):
                        # Only add one token per position
                        if comp.position not in added_positions:
                            added_positions.add(comp.position)
                            tokens_this_round.append(comp.final_token)
                            # Track token ID (use draft's ID for accepted, tokenize resampled to get ID)
                            if (
                                comp.status == TokenStatus.ACCEPTED
                                and comp.final_token_id
                            ):
                                token_ids_this_round.append(comp.final_token_id)
                                logger.info(
                                    f"    ACCEPTED token '{comp.final_token}' ID={comp.final_token_id}"
                                )
                            elif comp.status == TokenStatus.RESAMPLED:
                                # For resampled tokens, tokenize to get ID
                                resampled_ids = self.draft.tokenize(comp.final_token)
                                if resampled_ids:
                                    token_ids_this_round.append(resampled_ids[0])
                                    logger.info(
                                        f"    RESAMPLED token '{comp.final_token}' ID={resampled_ids[0]}"
                                    )
                                else:
                                    token_ids_this_round.append(0)  # Fallback
                                    logger.info(
                                        f"    RESAMPLED token '{comp.final_token}' ID=0 (fallback)"
                                    )
                        else:
                            logger.info(
                                f"    SKIPPING position {comp.position} - already added"
                            )

                # Handle bonus token
                if round_result.bonus_token:
                    tokens_this_round.append(round_result.bonus_token)
                    # Tokenize bonus token to get ID
                    bonus_ids = self.draft.tokenize(round_result.bonus_token)
                    if bonus_ids:
                        token_ids_this_round.append(bonus_ids[0])
                    else:
                        token_ids_this_round.append(0)  # Fallback

                    yield VerifyResultEvent(
                        round=current_round,
                        position=len(draft_tokens),
                        token=round_result.bonus_token,
                        token_id=0,
                        status=TokenStatus.BONUS,
                        draft_logprob=0.0,
                        target_logprob=None,
                        acceptance_prob=1.0,
                        target_entropy=None,
                        target_top_tokens=[],
                        verify_time_ms=verification.elapsed_ms,
                    )

                # Update generated context (track both string and token IDs)
                # Join tokens with spaces when needed - tokens may or may not have leading whitespace
                for token in tokens_this_round:
                    if (
                        token.strip()
                        and generated_text_so_far
                        and not generated_text_so_far.endswith(" ")
                        and not token.startswith(" ")
                    ):
                        generated_text_so_far += " " + token
                    else:
                        generated_text_so_far += token
                generated_token_ids.extend(token_ids_this_round)
                total_tokens_produced += len(tokens_this_round)

                # DEBUG: Log what was added to context
                logger.info(f"  ADDED tokens_this_round: {tokens_this_round}")
                logger.info(f"  ADDED token_ids_this_round: {token_ids_this_round}")
                logger.info(f"  CONTEXT generated_token_ids: {generated_token_ids}")
                logger.info(
                    f"  CONTEXT generated_text_so_far: {repr(generated_text_so_far[:100])}"
                )

                round_time = (time.perf_counter() - round_start) * 1000
                logger.info(
                    f"Round {current_round}: Produced {len(tokens_this_round)} tokens "
                    f"in {round_time:.0f}ms total"
                )

                # --- Phase 5: Emit metrics ---
                round_stats = RoundStats(
                    accepted=round_result.accepted_count,
                    total=len(draft_tokens),
                    tokens_produced=len(tokens_this_round),
                    draft_latency_ms=draft_elapsed,
                    verify_latency_ms=verification.elapsed_ms,
                    round_time_ms=round_time,
                )
                metrics.record_round(round_stats)

                yield MetricsEvent(
                    round=current_round,
                    acceptance_rate=metrics.acceptance_rate(),
                    round_accepted=round_result.accepted_count,
                    round_total=len(draft_tokens),
                    effective_tps=metrics.effective_tps(),
                    baseline_tps=metrics.baseline_tps(),
                    speedup=metrics.speedup(),
                    draft_latency_ms=metrics.avg_draft_latency(),
                    verify_latency_ms=metrics.avg_verify_latency(),
                    total_tokens_generated=metrics.total_tokens,
                )

                # Check for EOS or stop condition
                if any(
                    stop in generated_text_so_far
                    for stop in ["<|eot_id|>", "<|end_of_text|>", "</s>"]
                ):
                    logger.info(f"EOS detected after {current_round} rounds")
                    break

            # --- Done ---
            logger.info(
                f"Generation complete: {metrics.total_tokens} tokens in "
                f"{metrics.total_rounds} rounds"
            )
            yield GenerationDoneEvent(
                total_tokens=metrics.total_tokens,
                total_rounds=metrics.total_rounds,
                final_acceptance_rate=metrics.overall_acceptance_rate(),
                average_speedup=metrics.speedup(),
                generated_text=generated_text_so_far,
            )

        except Exception as e:
            logger.exception(f"Speculator error in round {current_round}: {e}")
            yield ErrorEvent(
                message=str(e),
                round=current_round,
            )
