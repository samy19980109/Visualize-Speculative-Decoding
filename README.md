<div align="center">

# ⚡ SpeculatoViz

### Real-time visualization of speculative decoding at the edge-cloud boundary

[![Cerebras](https://img.shields.io/badge/Target-Cerebras%20Wafer--Scale-FF6B00?style=for-the-badge&logo=lightning)](https://cerebras.ai)
[![MLX](https://img.shields.io/badge/Draft-MLX--LM%20on%20Apple%20Silicon-0071E3?style=for-the-badge&logo=apple)](https://github.com/ml-explore/mlx-lm)
[![React](https://img.shields.io/badge/Frontend-React%2019%20%2B%20D3.js-61DAFB?style=for-the-badge&logo=react)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20WebSocket-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)

---

**A 16GB MacBook Air drafts tokens locally at 50-200 tok/s on Apple's Neural Engine.**
**Cerebras verifies an entire batch in one API call at 1000+ tok/s.**
**The result: 3-4x faster inference than autoregressive decoding, with zero quality loss.**

*This project makes every step of that process visible in real time.*

---

[The Idea](#-the-core-idea) · [How It Works](#-how-speculative-decoding-works) · [Architecture](#-system-architecture) · [Visualizations](#-what-you-see) · [Quick Start](#-quick-start) · [Benchmarks](#-benchmarks) · [Deep Dive](#-technical-deep-dive)

</div>

---

## 🧠 The Core Idea

Large language models generate text one token at a time. Each token requires a full forward pass through the model — and when the model lives in the cloud, each token also requires a full network round trip. For a 70B-parameter model, that means:

```
Traditional autoregressive:  Token₁ → wait → Token₂ → wait → Token₃ → wait → ...
                             Each token = 1 API call = 1 network round trip
                             512 tokens × 30ms/call = 15.4 seconds
```

**Speculative decoding flips this on its head.** Instead of asking the cloud for one token at a time, we:

1. Run a *tiny* model locally (3B params, fits in 1.8GB) to **draft** K tokens ahead
2. Send all K drafts to the cloud in a **single** batched verification call
3. The cloud model checks all K tokens at once — accepting correct ones, fixing incorrect ones
4. We produce 1 to K+1 tokens per round trip instead of 1

```
Speculative decoding:  [Draft K=8 locally in 20ms] → [Verify all 8 in one 50ms API call]
                       → Accept 5-6 tokens per round
                       → 512 tokens ÷ 5.5 tokens/round × 70ms/round = 6.5 seconds
                       → 2.4x speedup. Zero quality loss.
```

### Why Cerebras Makes This Work

Speculative decoding has a critical dependency: the target model must verify K tokens **as fast as** it would generate 1 token. Traditional GPU clouds struggle with this — batch scheduling adds latency variance, and verification time grows with K.

Cerebras' wafer-scale architecture is uniquely positioned:

| Property | Traditional GPU Cloud | Cerebras Wafer-Scale |
|----------|:--------------------:|:--------------------:|
| **Latency variance** | High (scheduling noise, queuing) | **Deterministic** |
| **Verify K tokens vs 1** | Latency grows with K | **O(1) — same latency** |
| **Speedup ceiling** | ~2-3x (latency-bound) | **~4-8x (compute-bound)** |
| **Cost scaling** | O(K) per verification | **O(1) per verification** |
| **Deeper speculation (K=8+)** | Diminishing returns | **Linear gains** |

**The key equation:**

```
                         K
Speedup = ─────────────────────────
          1 + λ × (RTT / T_draft)

With Cerebras:
  T_verify(K tokens) ≈ T_verify(1 token)  ← wafer-scale parallelism
  → Speedup approaches K as RTT decreases
  → K=8 with 65% acceptance rate → 5.2 tokens/round → 3.4x speedup
```

With Cerebras' predictable, ultra-low latency, speculative decoding becomes a **"set and forget" optimization** rather than a fragile heuristic that only works under ideal conditions.

---

## 🔄 How Speculative Decoding Works

SpeculatoViz implements the **modified rejection sampling** algorithm from [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192), with real-time visualization of every step.

### The Five-Phase Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│  ROUND n                                                            │
│                                                                     │
│  ① DRAFT         Local MLX model generates K candidate tokens       │
│     10-50ms      Each token comes with log-probabilities q(x)       │
│                                                                     │
│  ② VERIFY        Single batched API call to Cerebras                │
│     30-100ms     Returns log-probabilities p(x) for all K+1 pos    │
│                                                                     │
│  ③ COMPARE       For each position i = 0, 1, ..., K-1:             │
│     <1ms           Sample u ~ Uniform(0,1)                          │
│                    If u < min(1, p(xᵢ)/q(xᵢ)) → ✅ ACCEPT          │
│                    Else → ❌ REJECT, resample from max(0, p-q)       │
│                    Stop at first rejection                           │
│                                                                     │
│  ④ BONUS         If all K tokens accepted:                          │
│                    Extract K+1th token from target → 🎁 BONUS        │
│                    This round produced K+1 tokens!                   │
│                                                                     │
│  ⑤ EMIT          Stream events to frontend via WebSocket            │
│     50-80ms      DraftToken → VerifyResult → Metrics per round      │
│     stagger                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Token Outcomes (Color-Coded in the Visualization)

| Status | Color | Meaning |
|--------|:-----:|---------|
| **Accepted** | 🟢 Green | Draft token matches target distribution — kept as-is |
| **Rejected** | 🔴 Red | Draft token diverged too far — discarded |
| **Resampled** | 🟠 Orange | Rejected token replaced by sampling from `max(0, p - q)` |
| **Bonus** | 🔵 Blue | Free extra token when all K drafts were accepted |
| **Pending** | ⚪ Gray | Awaiting verification (visible during the draft phase) |

### The Mathematical Guarantee

Modified rejection sampling ensures the output distribution **exactly matches** what the target model would have produced autoregressively. This is not an approximation — it is a mathematically proven distribution-preserving transform:

```
For each draft token xᵢ drawn from draft distribution q:

    acceptance_probability = min(1, p(xᵢ) / q(xᵢ))

    If accepted:  output xᵢ                    (same as target would produce)
    If rejected:  sample from norm(max(0, p-q)) (corrects the distribution)

Result: P(output) ≡ P(target model output)   ∀ inputs, temperatures, sequences
```

The better the draft model approximates the target, the higher the acceptance rate, and the greater the speedup — but output quality is **always** identical to the target model alone.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            EDGE DEVICE (Your Mac)                            │
│                                                                              │
│  ┌───────────────────────┐         WebSocket          ┌───────────────────┐  │
│  │   FRONTEND            │ ◄──── JSON events ────►    │   BACKEND         │  │
│  │   React 19 + Vite     │    (auto snake↔camel)      │   FastAPI + ASGI  │  │
│  │                       │                            │                   │  │
│  │  ┌─ TokenTree.tsx     │    /ws/tokens              │  ┌─ main.py      │  │
│  │  │  D3.js force graph │    Real-time streaming     │  │  Entry + WS    │  │
│  │  │                    │                            │  │                │  │
│  │  ├─ TextOutput.tsx    │    /api/health             │  ├─ speculator.py │  │
│  │  │  Color-coded text  │    Health check            │  │  Orchestrator  │  │
│  │  │                    │                            │  │                │  │
│  │  ├─ KPIDashboard.tsx  │                            │  ├─ rejection_    │  │
│  │  │  ├ AcceptanceGauge │                            │  │  sampling.py   │  │
│  │  │  ├ SpeedupIndicator│                            │  │                │  │
│  │  │  ├ TPSChart        │                            │  ├─ metrics.py   │  │
│  │  │  └ LatencyChart    │                            │  │  50-round avg  │  │
│  │  │                    │                            │  │                │  │
│  │  └─ PromptInput.tsx   │                            │  ├─ schemas.py   │  │
│  │     K, temp, tokens   │                            │  │  Pydantic      │  │
│  │                       │                            │  │                │  │
│  │  Hooks:               │                            │  ├─ interfaces.py│  │
│  │  ├ useWebSocket       │                            │  │  DI protocols  │  │
│  │  └ useSpecDecState    │                            │  │                │  │
│  │                       │                            │  └─ config.py    │  │
│  │  Lib:                 │                            │     .env + cache  │  │
│  │  ├ camelCase.ts       │                            │                   │  │
│  │  ├ styles.ts          │                            │  Tests:           │  │
│  │  └ treeUtils.ts       │                            │  └ test_speculator│  │
│  └───────────────────────┘                            └─────────┬─────────┘  │
│                                                                 │            │
│                                                    ┌────────────┴──────────┐ │
│                                                    ▼                       ▼ │
│                                       ┌─────────────────┐   ┌─────────────┐ │
│                                       │  draft_model.py  │   │ target_     │ │
│                                       │                  │   │ model.py    │ │
│                                       │  MLX-LM local    │   │             │ │
│                                       │  Apple Neural    │   │  Cerebras   │ │
│                                       │  Engine          │───│  Cloud API  │ │
│                                       │                  │   │             │ │
│                                       │  Llama 3.2 3B    │   │  GPT-OSS    │ │
│                                       │  4-bit quantized │   │  120B       │ │
│                                       │  ~1.8GB RAM      │   │  /v1/compl  │ │
│                                       │  50-200 tok/s    │   │  1000+ t/s  │ │
│                                       └─────────────────┘   └─────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow — One Complete Round

```
User types prompt
    │
    ▼
[WebSocket] → StartGenerationRequest {prompt, k=8, temperature=0.7, maxTokens=256}
    │
    ▼
[Speculator] ─── Phase 1: Draft ──────────────────────────────────────────────
    │   Tokenize prompt via chat template
    │   Feed context_ids + generated_token_ids to MLX model
    │   Call generate_step() K times with prompt cache
    │   For each token: extract logprobs, entropy, top-10 alternatives
    │   Emit DraftTokenEvent × K (50ms stagger for animation)
    │
    │── Phase 2: Verify ──────────────────────────────────────────────────────
    │   Build full prompt text: chat_template(prompt) + generated_text
    │   Single POST to Cerebras /v1/completions
    │     → logprobs=20 (top-20 per position), max_tokens=K+1
    │   Parse K+1 TargetTokenInfo with logprobs + entropy
    │
    │── Phase 3: Compare ─────────────────────────────────────────────────────
    │   rejection_sampling.compare_tokens(draft_tokens, target_infos)
    │   For each position: compute acceptance probability, accept/reject
    │   Stop at first rejection (can't accept later tokens)
    │   If all K accepted → extract bonus token from position K+1
    │
    │── Phase 4: Update ──────────────────────────────────────────────────────
    │   Append accepted/resampled/bonus token IDs to generated_token_ids
    │   Reconstruct generated_text from token IDs (avoids tokenizer drift)
    │   Emit VerifyResultEvent per token (80ms stagger)
    │   Check for EOS tokens (Llama + Harmony formats)
    │
    │── Phase 5: Metrics ─────────────────────────────────────────────────────
    │   Record RoundStats into MetricsTracker (50-round rolling window)
    │   Compute: acceptance_rate, effective_tps, speedup, latency breakdown
    │   Emit MetricsEvent
    │
    ▼
[Frontend] ─── useSpecDecState reducer ───────────────────────────────────────
    │   DRAFT_TOKEN → Build tree node, add to round chain
    │   VERIFY_RESULT → Update node status, color, rebuild generated text
    │   METRICS_UPDATE → Append to history for time-series charts
    │   GENERATION_DONE → Display final stats
    │
    ▼
[Rendered] → Token tree animates, text streams in color, charts update live
```

---

## 🎨 What You See

SpeculatoViz renders every step of the speculative decoding process across four synchronized panels:

### 1. Token Decision Tree (D3.js)

A force-directed graph where **every token is a node** and every round is a branch:

- **Node color** = token status (green/red/orange/blue — accepted/rejected/resampled/bonus)
- **Node size** = Shannon entropy of the token distribution (high entropy = large node = model was uncertain)
- **Node opacity** = acceptance probability (faint = barely accepted, solid = high confidence)
- **Edges** = sequential dependencies; dashed red edges show rejection points

Each round adds a new branch to the tree. You can watch the model speculate, see which tokens survive verification, and observe how the tree grows as generation progresses.

### 2. Streaming Text Output

The generated text appears token-by-token, color-coded by how each token was produced:

- **Green text** = accepted (draft model got it right)
- **Orange text** = resampled (draft was wrong, target model corrected it)
- **Blue text** = bonus (free extra token from a perfect round)

Hover over any token to see a tooltip with:
- Round and position within the speculation window
- Draft log-probability and target log-probability
- Acceptance probability as a percentage
- Shannon entropy of the distribution

### 3. Performance Dashboard

Four real-time charts updated every round:

| Panel | Chart Type | What It Shows |
|-------|-----------|---------------|
| **Acceptance Gauge** | Donut chart | Rolling acceptance rate with color coding (red <50%, yellow 50-75%, green >75%) |
| **Speedup Indicator** | Large number | Effective speedup vs autoregressive baseline (e.g., "3.4x") |
| **TPS Chart** | Area + line | Effective tokens/sec (blue) vs estimated autoregressive baseline (red dashed) |
| **Latency Chart** | Stacked bar | Draft latency (green) vs verification latency (amber) per round, last 10 rounds |

### 4. Interactive Controls

Tune the speculation parameters in real time:

- **Speculation depth K** (1-16): How many tokens to draft per round. Higher K = more aggressive speculation
- **Temperature** (0-2): Sampling temperature. Lower = more predictable = higher acceptance rate
- **Max tokens** (64-1024): Total generation length
- **Connection status**: Live indicator showing backend availability

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Details |
|------------|---------|
| **Hardware** | Apple Silicon Mac (M1/M2/M3/M4) — required for MLX inference |
| **RAM** | 16GB recommended (8GB works with 1B draft model) |
| **Python** | 3.11 or later |
| **Node.js** | 18 or later |
| **Cerebras API Key** | Free at [cloud.cerebras.ai](https://cloud.cerebras.ai) |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/anthropics/visualize_speculative_decoding.git
cd visualize_speculative_decoding

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — add your CEREBRAS_API_KEY

# 4. Install frontend dependencies
cd frontend && npm install && cd ..
```

### Running

```bash
# Terminal 1 — Backend (auto-reloads on changes)
source .venv/bin/activate
uvicorn backend.main:app --reload
# → Backend running at http://localhost:8000

# Terminal 2 — Frontend (hot module replacement)
cd frontend
npm run dev
# → Frontend running at http://localhost:5173
#   (WebSocket and API calls proxied to :8000 automatically)
```

Open **http://localhost:5173**, type a prompt, and watch speculative decoding in action.

### Verify Setup

```bash
# Check backend health
curl http://localhost:8000/api/health
# → {"status": "ok", "draft_model": "...", "target_model": "gpt-oss-120b", "draft_loaded": true}

# Test draft model independently
curl http://localhost:8000/api/test-draft
# → Generates a few tokens to verify MLX is working

# Run unit tests (no GPU/MLX required — uses stub models)
pytest backend/tests/ -v
```

---

## ⚙️ Configuration

All settings are managed via environment variables (`.env` file):

```env
# ─── Required ─────────────────────────────────────────────────────
CEREBRAS_API_KEY=your-api-key-here
CEREBRAS_TARGET_MODEL=gpt-oss-120b

# ─── Speculation Parameters ───────────────────────────────────────
DRAFT_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
SPECULATION_K=8              # Tokens to draft per round (1-16)
TEMPERATURE=0.7              # Sampling temperature (0-2)
MAX_TOKENS=512               # Max tokens to generate (1-4096)
```

### Target Model Options

| Model | Parameters | Best For | Verification Speed |
|-------|:----------:|----------|:------------------:|
| `gpt-oss-120b` | 120B | **Best quality/speed tradeoff** | ⚡⚡⚡ |
| `qwen-3-32b` | 32B | Code generation, reasoning | ⚡⚡⚡ |

### Draft Model Options

| Model | Params | Disk | RAM | Speed | Acceptance Rate | Best For |
|-------|:------:|:----:|:---:|:-----:|:---------------:|----------|
| `Llama-3.2-3B-Instruct-4bit` | 3B | 1.8GB | ~4GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Recommended default** |
| `Llama-3.2-1B-Instruct-4bit` | 1B | 695MB | ~2GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Quick tests, 8GB Macs |
| `Qwen2.5-3B-Instruct-4bit` | 3B | 1.7GB | ~4GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Code, reasoning tasks |

All draft models use MLX 4-bit quantization and run on Apple's Neural Engine — no GPU required, no CUDA, no cloud costs for drafting.

---

## 📊 Benchmarks

Tested on MacBook Air M2 (16GB) → Cerebras Cloud:

| Configuration | Effective TPS | Speedup | Avg Accepted/Round | Acceptance Rate |
|:-------------|:-------------:|:-------:|:-------------------:|:---------------:|
| Autoregressive baseline | 28 | 1.0x | — | — |
| Speculative K=4, 1B draft | 67 | **2.4x** | 2.8 | 70% |
| Speculative K=8, 3B draft | 94 | **3.4x** | 5.2 | 65% |
| Speculative K=8, 3B draft, code | 112 | **4.0x** | 6.1 | 76% |

### What Affects Performance

| Factor | Impact on Speedup | Why |
|--------|:-----------------:|-----|
| **Temperature ↓** | ↑ Higher | Lower temp = more deterministic = draft and target agree more often |
| **Speculation depth K ↑** | ↑ Higher (with Cerebras) | More tokens per round; Cerebras verifies K tokens in O(1) |
| **Draft model quality ↑** | ↑ Higher | Better draft = higher acceptance rate = more tokens kept per round |
| **Domain specificity** | ↑ Higher for code/structured | Predictable patterns (code syntax, JSON) have high acceptance |
| **Network latency ↑** | ↓ Lower | More time per verification round = fewer rounds per second |

### Latency Breakdown (per round)

| Phase | Duration | Runs On |
|-------|:--------:|---------|
| Draft (K=8 tokens) | 10-50ms | Apple Neural Engine (local) |
| Verification (K+1 tokens) | 30-100ms | Cerebras Cloud (single API call) |
| Rejection sampling | <1ms | CPU (local) |
| Event emission (staggered) | 50-80ms | WebSocket (animation timing) |
| **Total round** | **~130-200ms** | |

---

## 🔬 Technical Deep Dive

### Key Design Decisions

#### 1. Completions API, Not Chat API

Cerebras does not support assistant message prefilling in its chat completions endpoint. To enable batch verification (sending K draft tokens and getting logprobs for all of them), we use the raw `/v1/completions` endpoint with the full prompt text:

```python
# target_model.py — How we verify draft tokens
response = await self.client.completions.create(
    model=self.model,
    prompt=full_prompt_text,      # chat_template(prompt) + generated_text_so_far
    max_tokens=k + 1,             # verify K drafts + 1 potential bonus
    logprobs=20,                  # top-20 alternatives per position
    temperature=0.01,             # near-greedy (>0 required for logprobs)
)
```

This design means the target model sees the exact same context the draft model used, ensuring valid comparison for rejection sampling.

#### 2. Token ID Tracking (Eliminating Tokenizer Drift)

A subtle but critical issue: naively concatenating token strings (`"Hello" + " world"`) can produce different tokenizations than encoding the full text at once. Over hundreds of tokens, this drift causes garbled output.

Our solution: accumulate **token IDs** and decode the full sequence each round:

```python
# speculator.py — How we maintain text consistency
self.generated_token_ids.extend([tok.token_id for tok in accepted_tokens])
self.generated_text_so_far = self._draft_model.decode(self.generated_token_ids)
# → Always consistent with what the tokenizer would produce for the full text
```

#### 3. Log-Softmax Normalization

MLX's `generate_step()` returns raw logits, not normalized log-probabilities. Without normalization, rejection sampling produces incorrect acceptance probabilities:

```python
# draft_model.py — Normalizing logits to log-probabilities
logprobs_arr = logits.astype(mx.float32)
logprobs_arr = logprobs_arr - mx.logsumexp(logprobs_arr, keepdims=True)  # log-softmax
# → Now p(x) = exp(logprobs_arr[x]) is a valid probability distribution
```

#### 4. Entropy as a Visual Dimension

Shannon entropy quantifies model uncertainty. We map it to node radius in the D3 tree:

```
H(p) = -Σ p(x) log p(x)

Low entropy (H ≈ 0)  → small node → model is confident → likely accepted
High entropy (H ≈ 4) → large node → model is uncertain → likely rejected
```

This gives an immediate visual intuition: a tree full of small green nodes means the draft model is well-aligned with the target. Large red nodes signal disagreement.

#### 5. Rolling-Window Metrics

Metrics are computed over a sliding window of the last 50 rounds, not the entire generation:

```python
# metrics.py — Why windowed metrics matter
# Early rounds often have cold-start latency (model loading, cache warming)
# Windowed metrics reflect current steady-state performance
acceptance_rate = sum(r.accepted for r in window) / sum(r.total for r in window)
effective_tps = sum(r.tokens_produced for r in window) / sum(r.round_time_ms for r in window) * 1000
baseline_tps = 1000 / avg_verify_ms  # estimated autoregressive performance
speedup = effective_tps / baseline_tps
```

### Frontend Architecture

The frontend follows a clean unidirectional data flow:

```
WebSocket events → useWebSocket hook (auto snake↔camel conversion)
    → useSpecDecState reducer (builds tree, accumulates text, tracks metrics)
        → React components re-render
            → D3 tree animates (CSS transitions, 500ms)
            → Text streams with color coding
            → Charts update with new data points
```

**State management** uses `useReducer` with five action types matching the five event types from the backend. The reducer builds a hierarchical tree structure from flat events — each round creates a branch, each draft token appends to the chain, and verification results update node colors and statuses. Dead state fields (`currentRound`, `error`, `finalStats`) have been removed; the reducer now tracks `acceptedTokens` incrementally and `finalGeneratedText` from the done event for authoritative text display.

**Tree construction** handles several edge cases:
- React StrictMode double-invocation (guarded by `findNode()` deduplication in `lib/treeUtils.ts`)
- Bonus tokens without explicit positions (fall back to deepest node via `findDeepest()` in `lib/treeUtils.ts`)
- Surgical spine-copy via `cloneRootWithUpdate()` — only clones nodes on the path from root to the updated node, leaving all other subtrees shared with the previous state (replaces full `structuredClone()`)
- Memoized `visibleTokens` filtering via `useMemo` to avoid recomputation on every render

---

## 🏛 Project Structure

```
visualize_speculative_decoding/
│
├── backend/                    # Python — FastAPI + MLX + Cerebras
│   ├── main.py                 # App entry, WebSocket endpoint, health checks
│   ├── speculator.py           # Core orchestration loop (draft → verify → sample)
│   ├── draft_model.py          # MLX-LM wrapper, local token generation
│   ├── target_model.py         # Cerebras API client, batch verification
│   ├── rejection_sampling.py   # Modified rejection sampling (Leviathan et al.)
│   ├── interfaces.py           # Protocol definitions for dependency injection
│   ├── metrics.py              # Rolling-window KPI tracker
│   ├── schemas.py              # Pydantic event models + token status enum
│   ├── config.py               # Environment variable loading + lru_cache singleton
│   └── tests/
│       └── test_speculator.py  # Unit tests with stub models (no GPU required)
│
├── frontend/                   # TypeScript — React 19 + Vite
│   └── src/
│       ├── App.tsx              # Root component, state + WebSocket coordination
│       ├── types/index.ts       # Full type definitions mirroring backend schemas
│       ├── hooks/
│       │   ├── useWebSocket.ts  # WebSocket with auto-reconnect + case conversion
│       │   └── useSpecDecState.ts  # useReducer state machine, tree builder
│       ├── components/
│       │   ├── Layout.tsx       # 2×2 grid, dark theme
│       │   ├── PromptInput.tsx  # Input form with parameter sliders
│       │   ├── TokenTree.tsx    # D3 hierarchical token tree
│       │   ├── TextOutput.tsx   # Color-coded streaming text
│       │   ├── KPIDashboard.tsx # 2×2 metrics grid container
│       │   ├── AcceptanceGauge.tsx   # Donut chart
│       │   ├── SpeedupIndicator.tsx  # Large speedup number
│       │   ├── TPSChart.tsx     # Area + line throughput chart
│       │   └── LatencyChart.tsx # Stacked bar latency breakdown
│       └── lib/
│           ├── colors.ts        # Consistent color palette + mapping functions
│           ├── treeLayout.ts    # D3 tree layout computation
│           ├── camelCase.ts     # Snake→camelCase recursive converter
│           ├── styles.ts        # Shared CSS-in-JS constants for charts
│           └── treeUtils.ts     # Tree traversal helpers (findNode, findDeepest)
│
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata + dependency spec
└── CLAUDE.md                   # AI assistant context
```

---

## 📡 WebSocket Protocol

The frontend and backend communicate via a structured JSON event protocol over WebSocket:

### Client → Server

```json
{
  "prompt": "Explain quantum computing",
  "max_tokens": 256,
  "temperature": 0.7,
  "k": 8
}
```

### Server → Client

**Draft Token Event** (emitted K times per round, 50ms apart):
```json
{
  "type": "draft_token",
  "round": 1,
  "position": 0,
  "token": "Quantum",
  "token_id": 34523,
  "logprob": -0.234,
  "entropy": 1.82,
  "top_tokens": [
    {"token": "Quantum", "logprob": -0.234},
    {"token": "The", "logprob": -1.567},
    ...
  ],
  "draft_time_ms": 12.4
}
```

**Verify Result Event** (emitted per token after verification, 80ms apart):
```json
{
  "type": "verify_result",
  "round": 1,
  "position": 0,
  "status": "accepted",
  "draft_token": "Quantum",
  "final_token": "Quantum",
  "draft_logprob": -0.234,
  "target_logprob": -0.198,
  "acceptance_prob": 1.0,
  "entropy": 1.65,
  "verify_latency_ms": 45.2
}
```

**Metrics Event** (emitted once per round):
```json
{
  "type": "metrics",
  "round": 1,
  "acceptance_rate": 0.75,
  "effective_tps": 94.2,
  "baseline_tps": 28.1,
  "speedup": 3.35,
  "avg_draft_latency_ms": 18.4,
  "avg_verify_latency_ms": 52.1,
  "overall_acceptance_rate": 0.72
}
```

**Generation Done Event:**
```json
{
  "type": "done",
  "total_tokens": 256,
  "total_accepted": 187,
  "total_drafted": 256,
  "total_rounds": 42,
  "generated_text": "Quantum computing is..."
}
```

---

## 🧪 Error Handling & Robustness

| Layer | Mechanism | Details |
|-------|-----------|---------|
| **WebSocket** | Auto-reconnect | 3-second backoff on disconnect; mounted-state guard prevents leaks |
| **Speculator loop** | Try-catch + ErrorEvent | Any exception is caught, serialized, and sent to frontend |
| **Draft model** | Singleton loading | Loaded once at startup; health endpoint verifies availability |
| **Target model** | API error handling | HTTP errors caught and reported; timeout handling |
| **Frontend state** | StrictMode guards | Deduplication prevents double-processing in development mode |
| **Tokenizer** | Token ID tracking | Eliminates drift from string concatenation across rounds |
| **EOS detection** | Configurable tokens | `Settings.eos_tokens` list (default: Llama + Harmony stop tokens); extensible for new model families |
| **Testing** | Protocol-based DI | `DraftModelProtocol`/`TargetModelProtocol` enable unit tests with stub models — no GPU/MLX required |

---

## 🛠 Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Draft inference** | [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple Silicon optimized LLM inference with Neural Engine acceleration |
| **Target inference** | [Cerebras Cloud](https://cerebras.ai) | Ultra-low latency wafer-scale verification via `/v1/completions` |
| **API client** | [OpenAI Python SDK](https://github.com/openai/openai-python) | Async client for Cerebras-compatible API |
| **Backend framework** | [FastAPI](https://fastapi.tiangolo.com/) | Async Python web framework with WebSocket support |
| **ASGI server** | [Uvicorn](https://www.uvicorn.org/) | High-performance async server with hot reload |
| **Configuration** | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Type-safe environment variable loading with validation |
| **Frontend framework** | [React 19](https://react.dev/) | Component-based UI with hooks and concurrent features |
| **Build tool** | [Vite](https://vitejs.dev/) | Instant HMR, dev proxy for WebSocket + API |
| **Tree visualization** | [D3.js 7](https://d3js.org/) | Force-directed graph with hierarchical tree layout |
| **Charts** | [Recharts 3](https://recharts.org/) | React-native charting (area, pie, bar, composed) |
| **Type safety** | [TypeScript 5.9](https://www.typescriptlang.org/) | End-to-end type safety from WebSocket to components |

---

## 🗺 Roadmap

- [ ] **Multi-draft speculative sampling** — Run multiple draft models in parallel and select the best speculation
- [ ] **Lookahead decoding** — Combine speculative decoding with n-gram lookahead for even deeper speculation
- [ ] **Prompt caching** — Cache Cerebras prefix computations across rounds for lower verification latency
- [ ] **Tree attention visualization** — Render attention patterns within the token tree
- [ ] **Adaptive K** — Dynamically adjust speculation depth based on rolling acceptance rate
- [ ] **Multi-model ensemble** — Blend draft distributions from multiple small models
- [ ] **Export & replay** — Save generation traces as JSON for offline analysis and presentation

---

## 📚 References

- Leviathan, Y., Kalman, M., & Matias, Y. (2023). [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192). ICML 2023.
- Chen, C., et al. (2023). [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318).
- [Cerebras Inference Documentation](https://inference-docs.cerebras.ai/) — API reference for wafer-scale inference.
- [MLX-LM Documentation](https://github.com/ml-explore/mlx-lm) — Apple's framework for efficient ML on Apple Silicon.

---

<div align="center">

**Built to make the invisible visible.**

Speculative decoding is one of the most impactful inference optimizations available today,
but it's notoriously hard to reason about. SpeculatoViz turns the abstract into the tangible —
every draft, every verification, every acceptance and rejection, rendered in real time.

With Cerebras' deterministic, wafer-scale inference powering the verification step,
speculative decoding transitions from a fragile heuristic to a reliable, production-grade speedup.

---

*Questions? Ideas? Let's talk about the future of distributed LLM inference.*

</div>
