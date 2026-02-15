"""MLX-LM draft model wrapper for speculative decoding."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


@dataclass
class DraftToken:
    token_id: int
    token_str: str
    logprob: float
    entropy: float
    top_tokens: list[tuple[str, float]]  # (token_str, logprob)
    elapsed_ms: float


class DraftModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._cache = None

    def load(self) -> None:
        """Load model and tokenizer from HuggingFace hub / local cache."""
        self.model, self.tokenizer = load(self.model_name)

    def _reset_cache(self) -> None:
        self._cache = make_prompt_cache(self.model)

    def generate_draft_tokens(
        self,
        context_ids: list[int],
        k: int,
        temperature: float = 0.7,
    ) -> list[DraftToken]:
        """Generate K draft tokens with logprobs from the current context.

        Args:
            context_ids: Full token sequence so far (prompt + generated).
            k: Number of tokens to draft.
            temperature: Sampling temperature.

        Returns:
            List of DraftToken with logprobs and entropy.
        """
        self._reset_cache()

        prompt = mx.array(context_ids)
        sampler = make_sampler(temp=temperature) if temperature > 0 else None

        drafts: list[DraftToken] = []
        gen = generate_step(
            prompt=prompt,
            model=self.model,
            max_tokens=k,
            sampler=sampler,
            prompt_cache=self._cache,
        )

        for token_val, logprobs_arr in gen:
            t0 = time.perf_counter()
            # token_val may be int or mx.array depending on mlx-lm version
            if isinstance(token_val, mx.array):
                mx.eval(token_val, logprobs_arr)
                token_id = token_val.item()
            else:
                mx.eval(logprobs_arr)
                token_id = int(token_val)
            elapsed = (time.perf_counter() - t0) * 1000

            token_str = self.tokenizer.decode([token_id])

            # Normalize to proper log-probabilities (log-softmax).
            # generate_step may return logits in some mlx-lm versions;
            # this is a no-op when already normalized.
            logprobs_arr = logprobs_arr - mx.logsumexp(logprobs_arr)

            logprob = logprobs_arr[token_id].item()

            # Shannon entropy: H = -sum(p * log(p))
            probs = mx.exp(logprobs_arr)
            entropy = -mx.sum(probs * logprobs_arr).item()

            # Top-10 tokens
            top_k_indices = mx.argpartition(-logprobs_arr, kth=10)[:10]
            mx.eval(top_k_indices)
            top_tokens = []
            for idx in top_k_indices.tolist():
                tok_str = self.tokenizer.decode([idx])
                tok_lp = logprobs_arr[idx].item()
                top_tokens.append((tok_str, tok_lp))
            top_tokens.sort(key=lambda x: x[1], reverse=True)

            drafts.append(DraftToken(
                token_id=token_id,
                token_str=token_str,
                logprob=logprob,
                entropy=entropy,
                top_tokens=top_tokens[:10],
                elapsed_ms=elapsed,
            ))

        return drafts

    def get_prompt_text(self, prompt: str) -> str:
        """Get the raw chat template text (not tokenized) for use with completions API."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs (no special tokens like BOS)."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def apply_chat_template(self, prompt: str) -> list[int]:
        """Apply chat template and return token IDs."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return self.tokenizer.encode(text)
