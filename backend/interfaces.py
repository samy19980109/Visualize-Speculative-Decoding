"""Protocol definitions for draft and target models (DI-friendly)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .draft_model import DraftToken
from .target_model import VerificationResult


@runtime_checkable
class DraftModelProtocol(Protocol):
    def load(self) -> None: ...
    def generate_draft_tokens(
        self, context_ids: list[int], k: int, temperature: float
    ) -> list[DraftToken]: ...
    def get_prompt_text(self, prompt: str) -> str: ...
    def tokenize(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def apply_chat_template(self, prompt: str) -> list[int]: ...


@runtime_checkable
class TargetModelProtocol(Protocol):
    async def verify_tokens(
        self, prompt: str, generated_text: str, k: int
    ) -> VerificationResult: ...
