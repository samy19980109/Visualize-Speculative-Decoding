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
# Required: CEREBRAS_API_KEY, CEREBRAS_TARGET_MODEL (e.g. llama-3.3-70b)
# Optional: DRAFT_MODEL, SPECULATION_K, TEMPERATURE, MAX_TOKENS
```

## Architecture

### Backend (`backend/`) — Python, FastAPI, WebSocket

- **`main.py`** — App entry, WebSocket `/ws/tokens`, health check `/api/health`. Draft model loaded as singleton at startup.
- **`speculator.py`** — Orchestrator: runs the draft→verify→rejection-sampling loop, emits events via WebSocket. Maintains `generated_text_so_far` as concatenated token text and `token_ids` to avoid tokenizer drift.
- **`draft_model.py`** — Wraps MLX-LM `generate_step` for local draft token generation with logprobs. Uses `mx.eval` for lazy evaluation. Provides `get_prompt_text()` to extract raw chat template text for the completions endpoint.
- **`target_model.py`** — Calls Cerebras `/v1/completions` endpoint (not chat) with raw prompt text. Returns per-position logprobs and entropy for K+1 tokens. Uses `logprobs=20` parameter.
- **`rejection_sampling.py`** — Implements modified rejection sampling (Leviathan et al. 2023). Accepts/rejects draft tokens by comparing draft vs target log-probability distributions. Handles resampling and bonus token generation.
- **`metrics.py`** — Rolling-window (50 rounds) KPIs: acceptance rate, TPS, speedup, latency breakdown.
- **`schemas.py`** — Pydantic event models: `DraftTokenEvent`, `VerifyResultEvent`, `MetricsEvent`, `GenerationDoneEvent`, `ErrorEvent`. Token statuses: pending, accepted, rejected, resampled, bonus.
- **`config.py`** — `pydantic-settings` loading from `.env`.

### Frontend (`frontend/src/`) — React 19, TypeScript, Vite

- **`hooks/useWebSocket.ts`** — WebSocket connection with auto-reconnect. Converts snake_case↔camelCase automatically.
- **`hooks/useSpecDecState.ts`** — `useReducer`-based state management. Builds token tree for D3, tracks metrics history, manages generated text.
- **`components/TokenTree.tsx`** — D3 force-directed graph. Color coding: green=accepted, red=rejected, orange=resampled, blue=bonus. Node size=entropy.
- **`components/TextOutput.tsx`** — Color-coded streaming text display.
- **`components/KPIDashboard.tsx`**, `AcceptanceGauge.tsx`, `LatencyChart.tsx`, `TPSChart.tsx`, `SpeedupIndicator.tsx` — Recharts-based metrics panels.

### Data Flow
```
User prompt → WebSocket → Speculator
  ├─ Draft: MLX generates K tokens locally (~10-50ms)
  ├─ Verify: Cerebras /v1/completions verifies batch (~30-100ms)
  ├─ Compare: Rejection sampling accepts/rejects each position
  └─ Emit: Stream events to frontend via WebSocket (50-80ms stagger)
```

## Key Design Decisions

- **Completions API, not Chat API**: Cerebras doesn't support assistant message prefilling. The target model uses `/v1/completions` with raw prompt text concatenation for reliable continuation.
- **Token ID tracking**: Text is reconstructed from accumulated token IDs (`self._tokenizer.decode(self.token_ids)`) rather than string concatenation, avoiding tokenizer drift.
- **Log-softmax normalization**: Draft model logprobs are normalized via `mx.softmax` → `mx.log` since raw logits from `generate_step` aren't log-probabilities.
- **Singleton draft model**: MLX model loaded once at FastAPI startup via `@asynccontextmanager` lifespan, shared across requests.

## Requirements

- Apple Silicon Mac (M1+) — required for MLX
- Python 3.11+
- Node.js 18+
- Cerebras API key
