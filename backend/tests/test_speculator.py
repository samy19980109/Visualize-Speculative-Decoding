"""Unit tests for speculative decoding loop using stub models (no GPU/MLX required)."""

from __future__ import annotations

import os

import pytest

# Set required env vars before importing backend modules that call get_settings()
os.environ.setdefault("CEREBRAS_API_KEY", "test-key")
os.environ.setdefault("CEREBRAS_TARGET_MODEL", "test-model")

from backend.config import get_settings  # noqa: E402

# Clear the lru_cache so test env vars take effect
get_settings.cache_clear()

from backend.draft_model import DraftToken  # noqa: E402
from backend.interfaces import DraftModelProtocol, TargetModelProtocol
from backend.rejection_sampling import (
    DraftInput,
    TargetInput,
    run_rejection_sampling,
)
from backend.schemas import TokenStatus
from backend.speculator import Speculator
from backend.target_model import TargetTokenInfo, VerificationResult


# --- Stub Models ---


class StubDraftModel:
    """Stub that returns pre-configured draft tokens."""

    def __init__(self, tokens: list[DraftToken] | None = None):
        self._tokens = tokens or []
        self._decode_map: dict[int, str] = {}

    def load(self) -> None:
        pass

    def generate_draft_tokens(
        self, context_ids: list[int], k: int, temperature: float
    ) -> list[DraftToken]:
        return self._tokens[:k]

    def get_prompt_text(self, prompt: str) -> str:
        return f"<user>{prompt}<assistant>"

    def tokenize(self, text: str) -> list[int]:
        # Simple: return hash-based ID
        return [abs(hash(text)) % 10000]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._decode_map.get(tid, f"[{tid}]") for tid in token_ids)

    def apply_chat_template(self, prompt: str) -> list[int]:
        return [1, 2, 3]  # Dummy context IDs


class StubTargetModel:
    """Stub that returns pre-configured verification results."""

    def __init__(self, result: VerificationResult | None = None):
        self._result = result or VerificationResult(positions=[], elapsed_ms=10.0)

    async def verify_tokens(
        self, prompt: str, generated_text: str, k: int
    ) -> VerificationResult:
        return self._result


# --- Protocol Conformance ---


def test_stub_draft_implements_protocol():
    assert isinstance(StubDraftModel(), DraftModelProtocol)


def test_stub_target_implements_protocol():
    assert isinstance(StubTargetModel(), TargetModelProtocol)


# --- Rejection Sampling ---


def test_rejection_sampling_all_accepted():
    drafts = [
        DraftInput(token_str="hello", token_id=1, logprob=-0.5),
        DraftInput(token_str="world", token_id=2, logprob=-0.3),
    ]
    targets = [
        TargetInput(token_str="hello", top_logprobs={"hello": -0.5}),
        TargetInput(token_str="world", top_logprobs={"world": -0.3}),
        TargetInput(token_str="!", top_logprobs={"!": -0.1}),  # bonus
    ]
    result = run_rejection_sampling(drafts, targets)
    assert result.accepted_count == 2
    assert result.bonus_token == "!"
    assert all(c.status == TokenStatus.ACCEPTED for c in result.comparisons)


def test_rejection_sampling_rejected_has_no_token_id():
    drafts = [
        DraftInput(token_str="foo", token_id=99, logprob=-0.1),
    ]
    targets = [
        TargetInput(token_str="bar", top_logprobs={"bar": -0.2}),
    ]
    result = run_rejection_sampling(drafts, targets)
    rejected = [c for c in result.comparisons if c.status == TokenStatus.REJECTED]
    assert len(rejected) == 1
    assert rejected[0].final_token_id is None  # Phase 1 bug fix


def test_rejection_sampling_exact_match():
    drafts = [DraftInput(token_str="the", token_id=5, logprob=-1.0)]
    targets = [TargetInput(token_str="the", top_logprobs={"the": -1.0})]
    result = run_rejection_sampling(drafts, targets)
    assert result.accepted_count == 1
    assert result.comparisons[0].final_token_id == 5


# --- Speculator Integration ---


@pytest.mark.asyncio
async def test_speculator_generates_events():
    """Full loop with stubs — should produce draft, verify, metrics, and done events."""
    draft_tokens = [
        DraftToken(
            token_id=10,
            token_str="Hello",
            logprob=-0.5,
            entropy=1.0,
            top_tokens=[("Hello", -0.5)],
            elapsed_ms=5.0,
        ),
    ]

    target_result = VerificationResult(
        positions=[
            TargetTokenInfo(
                token_str="Hello",
                token_logprob=-0.5,
                top_logprobs={"Hello": -0.5},
                entropy=1.0,
            ),
            TargetTokenInfo(
                token_str="!",
                token_logprob=-0.3,
                top_logprobs={"!": -0.3},
                entropy=0.5,
            ),
        ],
        elapsed_ms=20.0,
    )

    stub_draft = StubDraftModel(tokens=draft_tokens)
    stub_draft._decode_map = {10: "Hello", abs(hash("!")) % 10000: "!"}
    stub_target = StubTargetModel(result=target_result)

    speculator = Speculator(draft=stub_draft, target=stub_target)

    events = []
    async for event in speculator.generate(
        prompt="Say hi", max_tokens=2, temperature=0.7, k=1
    ):
        events.append(event)

    event_types = [e.type for e in events]
    assert "draft_token" in event_types
    assert "verify_result" in event_types
    assert "metrics" in event_types
    assert "done" in event_types


# --- Metrics ---


def test_metrics_baseline_tps_uses_k():
    from backend.metrics import MetricsTracker, RoundStats

    tracker = MetricsTracker()
    tracker.record_round(
        RoundStats(
            accepted=4,
            total=4,
            tokens_produced=5,
            draft_latency_ms=10,
            verify_latency_ms=50,
            round_time_ms=60,
            k=4,
        )
    )
    # baseline_tps = 1000 / (verify_ms / (k+1)) = 1000 / (50/5) = 1000/10 = 100
    assert abs(tracker.baseline_tps() - 100.0) < 0.01
