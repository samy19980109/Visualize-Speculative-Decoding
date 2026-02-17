# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpeculatoViz: real-time visualization of speculative decoding at the edge-cloud boundary. A small local draft model (MLX-LM on Apple Silicon) generates candidate tokens, which a large cloud target model (Cerebras) verifies in batches. The frontend visualizes the draft-verify-accept/reject cycle with a D3 token tree, color-coded text output, and live performance metrics.

## Commands

### Backend
```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run (auto-reload)
uvicorn backend.main:app --reload  # http://localhost:8000

# Lint
ruff check backend/
ruff format backend/
```

### Frontend
```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (proxies WS to :8000)
npm run build    # production build
npm run lint     # ESLint
```

### Environment
```bash
cp .env.example .env
# Required: CEREBRAS_API_KEY, CEREBRAS_TARGET_MODEL (e.g. gpt-oss-120b)
# Optional: DRAFT_MODEL, SPECULATION_K, TEMPERATURE, MAX_TOKENS
```

### Tests
```bash
# Run backend tests (no GPU/MLX required — uses stub models)
pytest backend/tests/ -v
```

## Architecture

### Backend (`backend/`) — Python, FastAPI, WebSocket

- **`main.py`** — App entry, WebSocket `/ws/tokens`, health check `/api/health`. Draft model loaded as singleton at startup.
- **`speculator.py`** — Orchestrator: decomposed into single-responsibility methods (`_run_draft_phase`, `_emit_draft_events`, `_run_verify_phase`, `_run_rejection_sampling`, `_emit_verify_events`, `_update_state`, `_check_eos`). Uses a `GenerationState` dataclass for mutable per-generation state instead of instance variables.
- **`draft_model.py`** — Wraps MLX-LM `generate_step` for local draft token generation with logprobs. Uses `mx.eval` for lazy evaluation. Provides `get_prompt_text()` to extract raw chat template text for the completions endpoint. Generation setup extracted into `_setup_generation()` helper.
- **`target_model.py`** — Calls Cerebras `/v1/completions` endpoint with the target model's native prompt format (Harmony for GPT-OSS, raw text for others). Returns per-position logprobs and entropy for K+1 tokens via typed `VerificationResult`/`TargetTokenInfo` dataclasses. Uses `logprobs=20` parameter.
- **`rejection_sampling.py`** — Implements modified rejection sampling (Leviathan et al. 2023). Uses typed `DraftInput`/`TargetInput` dataclasses instead of untyped dicts. Handles resampling and bonus token generation.
- **`interfaces.py`** — Protocol definitions (`DraftModelProtocol`, `TargetModelProtocol`) for dependency injection. Enables testing with stub models without GPU/MLX.
- **`metrics.py`** — Rolling-window (50 rounds) KPIs: acceptance rate, TPS, speedup, latency breakdown.
- **`schemas.py`** — Pydantic event models: `DraftTokenEvent`, `VerifyResultEvent`, `MetricsEvent`, `GenerationDoneEvent`, `ErrorEvent`. Token statuses: pending, accepted, rejected, resampled, bonus.
- **`config.py`** — `pydantic-settings` loading from `.env`. Uses `@lru_cache` via `get_settings()` for singleton access. Includes configurable `cors_origins` and `eos_tokens`.
- **`tests/test_speculator.py`** — Unit tests using `StubDraftModel`/`StubTargetModel` (no GPU/MLX required). Tests the full draft→verify→sampling loop, bonus token generation, and EOS detection.

### Frontend (`frontend/src/`) — React 19, TypeScript, Vite

- **`hooks/useWebSocket.ts`** — WebSocket connection with auto-reconnect. Snake→camelCase conversion extracted to `lib/camelCase.ts`.
- **`hooks/useSpecDecState.ts`** — `useReducer`-based state management. Uses surgical spine-copy (`cloneRootWithUpdate`) instead of `structuredClone` for tree updates. Dead state fields removed (`currentRound`, `error`, `finalStats`). Tracks `acceptedTokens` incrementally and `finalGeneratedText` from done events. Exposes memoized `visibleTokens` via `useMemo`.
- **`components/TokenTree.tsx`** — D3 hierarchical tree layout. Color coding: green=accepted, red=rejected, orange=resampled, blue=bonus. Node size=entropy.
- **`components/TextOutput.tsx`** — Color-coded streaming text display.
- **`components/KPIDashboard.tsx`**, `AcceptanceGauge.tsx`, `LatencyChart.tsx`, `TPSChart.tsx`, `SpeedupIndicator.tsx` — Recharts-based metrics panels. Shared styles extracted to `lib/styles.ts`.
- **`lib/camelCase.ts`** — Recursive `snakeToCamel` utility for WebSocket event conversion.
- **`lib/styles.ts`** — Shared CSS-in-JS constants (`sectionHeaderStyle`, `cardStyle`, `tooltipStyle`, `chartEmptyStyle`) used across chart components.
- **`lib/treeUtils.ts`** — Extracted tree traversal helpers (`findNode`, `findDeepest`) used by the state reducer.

### Data Flow
```
User prompt → WebSocket → Speculator
  ├─ Draft: MLX generates K tokens locally (~10-50ms)
  ├─ Verify: Cerebras /v1/completions verifies batch (~30-100ms)
  ├─ Compare: Rejection sampling accepts/rejects each position
  └─ Emit: Stream events to frontend via WebSocket (50-80ms stagger)
```

## Key Design Decisions

- **Protocol-based DI**: `DraftModelProtocol` and `TargetModelProtocol` in `interfaces.py` decouple the speculator from concrete model implementations, enabling unit testing with lightweight stubs.
- **GenerationState dataclass**: Per-generation mutable state (`context_ids`, `generated_text_so_far`, `generated_token_ids`, `current_round`, `total_tokens_produced`) is isolated in a dataclass rather than stored as instance variables on the Speculator.
- **Typed rejection sampling inputs**: `DraftInput`/`TargetInput` dataclasses replace untyped dicts for type-safe rejection sampling.
- **Surgical tree updates**: Frontend uses spine-copy (`cloneRootWithUpdate`) instead of `structuredClone` on the entire tree, only cloning nodes on the path from root to the updated node.
- **Model-native prompt formatting**: The target model formats verification prompts in its own native template (Harmony for GPT-OSS, raw text for others) via the Completions API. This avoids prompt format mismatches when draft and target use different model families/tokenizers.
- **Token ID tracking**: Text is reconstructed from accumulated token IDs (`self._tokenizer.decode(self.token_ids)`) rather than string concatenation, avoiding tokenizer drift.
- **Log-softmax normalization**: Draft model logprobs are normalized via `mx.softmax` → `mx.log` since raw logits from `generate_step` aren't log-probabilities.
- **Singleton draft model**: MLX model loaded once at FastAPI startup via `@asynccontextmanager` lifespan, shared across requests.
- **Configurable EOS tokens**: End-of-sequence tokens are configurable via `Settings.eos_tokens` rather than hardcoded, supporting multiple model families.

## Requirements

- Apple Silicon Mac (M1+) — required for MLX
- Python 3.11+
- Node.js 18+
- Cerebras API key
