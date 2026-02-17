from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class TokenStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RESAMPLED = "resampled"
    BONUS = "bonus"


class EventType(str, Enum):
    DRAFT_TOKEN = "draft_token"
    VERIFY_RESULT = "verify_result"
    METRICS = "metrics"
    DONE = "done"
    ERROR = "error"


# --- Incoming ---

class StartGenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    k: Optional[int] = None


# --- Token info shared between events ---

class TopToken(BaseModel):
    token: str
    logprob: float


# --- Outgoing events ---

class DraftTokenEvent(BaseModel):
    type: str = EventType.DRAFT_TOKEN
    round: int
    position: int
    token: str
    token_id: int
    logprob: float
    entropy: float
    top_tokens: list[TopToken] = Field(default_factory=list)
    draft_time_ms: float


class VerifyResultEvent(BaseModel):
    type: str = EventType.VERIFY_RESULT
    round: int
    position: int
    token: str
    token_id: int
    status: TokenStatus
    draft_logprob: float
    target_logprob: Optional[float] = None
    acceptance_prob: Optional[float] = None
    target_entropy: Optional[float] = None
    target_top_tokens: list[TopToken] = Field(default_factory=list)
    verify_time_ms: float  # Total verification latency (shared across round)


class MetricsEvent(BaseModel):
    type: str = EventType.METRICS
    round: int
    acceptance_rate: float
    round_accepted: int
    round_total: int
    effective_tps: float
    baseline_tps: float
    speedup: float
    draft_latency_ms: float
    verify_latency_ms: float
    total_tokens_generated: int


class GenerationDoneEvent(BaseModel):
    type: str = EventType.DONE
    total_tokens: int
    total_rounds: int
    final_acceptance_rate: float
    average_speedup: float
    generated_text: str


class ErrorEvent(BaseModel):
    type: str = EventType.ERROR
    message: str
    round: Optional[int] = None
