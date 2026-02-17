"""Orchestrator: draft → verify → compare → emit events loop."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

from .config import get_settings
from .draft_model import DraftToken
from .interfaces import DraftModelProtocol, TargetModelProtocol
from .metrics import MetricsTracker, RoundStats
from .rejection_sampling import (
    DraftInput,
    RoundResult,
    TargetInput,
    run_rejection_sampling,
)
from .schemas import (
    DraftTokenEvent,
    ErrorEvent,
    GenerationDoneEvent,
    MetricsEvent,
    TokenStatus,
    TopToken,
    VerifyResultEvent,
)
from .target_model import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationState:
    """Mutable state for a single generation run."""

    context_ids: list[int]
    generated_text_so_far: str = ""
    generated_token_ids: list[int] = field(default_factory=list)
    current_round: int = 0
    total_tokens_produced: int = 0


class Speculator:
    def __init__(self, draft: DraftModelProtocol, target: TargetModelProtocol):
        self.draft = draft
        self.target = target

    async def _run_draft_phase(
        self,
        state: GenerationState,
        k: int,
        temperature: float,
    ) -> tuple[list[DraftToken], float]:
        """Draft K tokens locally via the draft model.

        Returns (draft_tokens, draft_elapsed_ms).
        """
        full_context_ids = state.context_ids + state.generated_token_ids

        logger.info(
            f"Round {state.current_round}: Drafting {k} tokens "
            f"(context: {len(full_context_ids)} ids, "
            f"generated_ids: {len(state.generated_token_ids)})..."
        )
        logger.debug(f"  DRAFT CONTEXT generated_token_ids: {state.generated_token_ids}")

        draft_start = time.perf_counter()
        draft_tokens: list[DraftToken] = await asyncio.to_thread(
            self.draft.generate_draft_tokens,
            full_context_ids,
            k,
            temperature,
        )
        draft_elapsed = (time.perf_counter() - draft_start) * 1000

        logger.info(
            f"Round {state.current_round}: Drafted {len(draft_tokens)} tokens "
            f"in {draft_elapsed:.0f}ms: {[dt.token_str for dt in draft_tokens]}"
        )
        for i, dt in enumerate(draft_tokens):
            logger.debug(f"  DRAFTED pos={i} token='{dt.token_str}' id={dt.token_id}")

        return draft_tokens, draft_elapsed

    async def _emit_draft_events(
        self,
        draft_tokens: list[DraftToken],
        current_round: int,
    ) -> AsyncGenerator:
        """Yield DraftTokenEvent for each draft token with staggered timing."""
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

    async def _run_verify_phase(
        self,
        state: GenerationState,
        prompt: str,
        k: int,
    ) -> VerificationResult:
        """Verify draft tokens via the Cerebras API."""
        logger.info(f"Round {state.current_round}: Verifying via Cerebras API...")
        verification = await self.target.verify_tokens(
            prompt=prompt,
            generated_text=state.generated_text_so_far,
            k=k,
        )
        logger.info(
            f"Round {state.current_round}: Verification returned "
            f"{len(verification.positions)} positions in {verification.elapsed_ms:.0f}ms"
        )
        return verification

    def _run_rejection_sampling(
        self,
        draft_tokens: list[DraftToken],
        verification: VerificationResult,
        current_round: int,
    ) -> RoundResult:
        """Run rejection sampling with typed inputs."""
        draft_inputs = [
            DraftInput(
                token_str=dt.token_str,
                token_id=dt.token_id,
                logprob=dt.logprob,
            )
            for dt in draft_tokens
        ]
        target_inputs = [
            TargetInput(
                token_str=pos.token_str,
                top_logprobs=pos.top_logprobs,
            )
            for pos in verification.positions
        ]

        round_result = run_rejection_sampling(draft_inputs, target_inputs)
        logger.info(
            f"Round {current_round}: Accepted {round_result.accepted_count}/"
            f"{len(draft_tokens)}, bonus={'yes' if round_result.bonus_token else 'no'}"
        )
        for comp in round_result.comparisons:
            logger.debug(
                f"  COMPARISON pos={comp.position} status={comp.status} "
                f"draft='{comp.draft_token}' final='{comp.final_token}' "
                f"final_id={comp.final_token_id}"
            )
        return round_result

    async def _emit_verify_events(
        self,
        round_result: RoundResult,
        draft_tokens: list[DraftToken],
        verification: VerificationResult,
        current_round: int,
    ) -> AsyncGenerator[VerifyResultEvent, None]:
        """Yield VerifyResultEvents and collect accepted tokens/IDs.

        Returns via the generator; caller collects tokens_this_round and
        token_ids_this_round from the side effects on the lists passed in.
        """
        tokens_this_round: list[str] = []
        token_ids_this_round: list[int] = []
        added_positions: set[int] = set()

        for comp in round_result.comparisons:
            # Only process the first token per position (REJECTED followed by RESAMPLED)
            if comp.position in added_positions:
                logger.debug(f"    SKIPPING position {comp.position} - already added")
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

            # Collect accepted/resampled tokens for context update
            if comp.status in (TokenStatus.ACCEPTED, TokenStatus.RESAMPLED):
                added_positions.add(comp.position)
                tokens_this_round.append(comp.final_token)
                if comp.status == TokenStatus.ACCEPTED and comp.final_token_id:
                    token_ids_this_round.append(comp.final_token_id)
                    logger.debug(
                        f"    ACCEPTED token '{comp.final_token}' ID={comp.final_token_id}"
                    )
                elif comp.status == TokenStatus.RESAMPLED:
                    resampled_ids = self.draft.tokenize(comp.final_token)
                    if resampled_ids:
                        token_ids_this_round.extend(resampled_ids)
                        logger.debug(
                            f"    RESAMPLED token '{comp.final_token}' IDs={resampled_ids}"
                        )
                    else:
                        logger.warning(
                            f"    RESAMPLED token '{comp.final_token}' produced no IDs"
                        )

        # Handle bonus token
        if round_result.bonus_token:
            tokens_this_round.append(round_result.bonus_token)
            bonus_ids = self.draft.tokenize(round_result.bonus_token)
            if bonus_ids:
                token_ids_this_round.extend(bonus_ids)
            else:
                logger.warning(
                    f"    BONUS token '{round_result.bonus_token}' produced no IDs"
                )

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

        # Store results on the generator function for the caller to access
        # We use a different approach: yield a special sentinel or use side effects
        # Instead, we'll make this a regular method that returns both events and data
        self._last_tokens_this_round = tokens_this_round
        self._last_token_ids_this_round = token_ids_this_round

    def _update_context(
        self,
        state: GenerationState,
        tokens_this_round: list[str],
        token_ids_this_round: list[int],
    ) -> None:
        """Update generation state with newly produced tokens."""
        state.generated_token_ids.extend(token_ids_this_round)
        state.total_tokens_produced += len(tokens_this_round)
        state.generated_text_so_far = self.draft.decode(state.generated_token_ids)

        logger.debug(f"  ADDED tokens_this_round: {tokens_this_round}")
        logger.debug(f"  ADDED token_ids_this_round: {token_ids_this_round}")
        logger.debug(f"  CONTEXT generated_token_ids: {state.generated_token_ids}")
        logger.debug(
            f"  CONTEXT generated_text_so_far: {repr(state.generated_text_so_far[:100])}"
        )

    def _make_metrics_event(
        self,
        metrics: MetricsTracker,
        round_result: RoundResult,
        draft_tokens: list[DraftToken],
        current_round: int,
    ) -> MetricsEvent:
        """Build a MetricsEvent from current tracker state."""
        return MetricsEvent(
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

    def _is_eos(self, text: str) -> bool:
        """Check if any EOS token appears in the generated text."""
        settings = get_settings()
        return any(stop in text for stop in settings.eos_tokens)

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

        logger.info("Applying chat template...")
        context_ids = self.draft.apply_chat_template(prompt)
        logger.info(f"Context has {len(context_ids)} tokens")

        state = GenerationState(context_ids=context_ids)

        try:
            while state.total_tokens_produced < max_tokens:
                state.current_round += 1
                round_start = time.perf_counter()

                # --- Draft K tokens locally ---
                draft_tokens, draft_elapsed = await self._run_draft_phase(
                    state, k, temperature
                )

                # Emit draft token events
                async for event in self._emit_draft_events(
                    draft_tokens, state.current_round
                ):
                    yield event

                # --- Verify via Cerebras API ---
                verification = await self._run_verify_phase(state, prompt, k)

                # --- Rejection sampling ---
                round_result = self._run_rejection_sampling(
                    draft_tokens, verification, state.current_round
                )

                # --- Emit verify result events ---
                async for event in self._emit_verify_events(
                    round_result, draft_tokens, verification, state.current_round
                ):
                    yield event

                tokens_this_round = self._last_tokens_this_round
                token_ids_this_round = self._last_token_ids_this_round

                # --- Update context ---
                self._update_context(state, tokens_this_round, token_ids_this_round)

                round_time = (time.perf_counter() - round_start) * 1000
                logger.info(
                    f"Round {state.current_round}: Produced {len(tokens_this_round)} "
                    f"tokens in {round_time:.0f}ms total"
                )

                # --- Emit metrics ---
                round_stats = RoundStats(
                    accepted=round_result.accepted_count,
                    total=len(draft_tokens),
                    tokens_produced=len(tokens_this_round),
                    draft_latency_ms=draft_elapsed,
                    verify_latency_ms=verification.elapsed_ms,
                    round_time_ms=round_time,
                    k=len(draft_tokens),
                )
                metrics.record_round(round_stats)
                yield self._make_metrics_event(
                    metrics, round_result, draft_tokens, state.current_round
                )

                # Check for EOS
                if self._is_eos(state.generated_text_so_far):
                    logger.info(f"EOS detected after {state.current_round} rounds")
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
                generated_text=state.generated_text_so_far,
            )

        except Exception as e:
            logger.exception(f"Speculator error in round {state.current_round}: {e}")
            yield ErrorEvent(
                message=str(e),
                round=state.current_round,
            )
