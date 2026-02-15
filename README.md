<div align="center">

# ⚡ SpeculatoViz

**Real-time visualization of speculative decoding at the edge-cloud boundary**

[![Cerebras](https://img.shields.io/badge/Powered%20by-Cerebras-FF6B00?style=flat-square&logo=lightning)](https://cerebras.ai)
[![MLX](https://img.shields.io/badge/Draft-MLX--LM-blue?style=flat-square&logo=apple)](https://github.com/ml-explore/mlx-lm)
[![WebSocket](https://img.shields.io/badge/Stream-WebSocket-green?style=flat-square&logo=websocket)](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

*A 16GB MacBook Air + Cerebras API = 2-4x faster inference than autoregressive sampling*

</div>

---

## The Core Insight

**Speculative decoding** is the "free lunch" of LLM inference: by having a small local model draft tokens and a large cloud model verify them in parallel, we achieve substantial speedups without sacrificing output quality. 

**The twist**: With Cerebras' deterministic ultra-low latency inference, the bottleneck shifts from target model computation to *network RTT*. This visualization demonstrates why Cerebras' architecture is uniquely positioned to maximize speculative decoding gains.

```
Speedup = K / (1 + λ × RTT/T_draft)

Where:
  K = speculation depth (typically 4-8)
  λ = network overhead factor
  RTT = round-trip time to target model
  T_draft = draft generation time

With Cerebras: T_verify ≈ T_autoregressive (single token)
                → Speedup approaches K with low RTT
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT SIDE (Edge)                             │
│  ┌──────────────────┐    WebSocket     ┌─────────────────────────────┐  │
│  │  React + Vite    │ ◄──────────────► │  FastAPI Orchestrator       │  │
│  │  ├─ D3.js Tree   │                  │  ├─ Token Stream Manager    │  │
│  │  ├─ Recharts     │                  │  ├─ Modified Rejection      │  │
│  │  └─ Real-time UI │                  │  │   Sampling Engine         │  │
│  └──────────────────┘                  │  └─ WebSocket Broadcaster   │  │
│                                        └─────────────────────────────┘  │
│                                                     │                    │
│                                        ┌────────────┴────────────┐       │
│                                        ▼                         ▼       │
│                           ┌─────────────────────┐   ┌──────────────────┐ │
│                           │  MLX-LM (Apple ANE) │   │  Cerebras Cloud  │ │
│                           │  Draft Model (1-3B) │   │  Target (8B-70B) │ │
│                           │  ~50-200 tokens/sec │   │  ~1000+ tokens/s │ │
│                           └─────────────────────┘   └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Speculation Loop

```python
while not eos:
    # 1. Draft: Local model generates K candidates (parallel on ANE)
    draft_tokens = draft_model.generate(K, temperature)
    #    └─ Latency: ~10-50ms on M1/M2
    
    # 2. Verify: Single batched API call to Cerebras
    target_logits = cerebras.verify_batch(draft_tokens)
    #    └─ Latency: ~30-100ms (vs K×30ms autoregressive)
    
    # 3. Accept/Reject: Modified rejection sampling
    accepted = 0
    for i, (draft_logit, target_logit) in enumerate(zip(draft_logits, target_logits)):
        if accept(draft_logit, target_logit):
            output.append(draft_tokens[i])
            accepted += 1
        else:
            # Resample from adjusted distribution
            output.append(resample(target_logit, draft_logit))
            break
    
    # 4. Bonus: If all K accepted, get one more from target
    if accepted == K:
        output.append(target_model.next_token())
```

---

## Why This Matters for Cerebras

| Metric | Traditional GPU Cloud | Cerebras Wafer-Scale |
|--------|----------------------|---------------------|
| **Latency Variance** | High (scheduling noise) | Deterministic |
| **Batch Efficiency** | Degrades with K | Maintains at K=8+ |
| **Speedup Ceiling** | ~2-3x (latency bound) | ~4-8x (compute bound) |
| **Cost per Token** | O(K) | O(1) |

**The visualization proves**: With Cerebras' predictable low-latency inference, speculative decoding becomes a "set and forget" optimization rather than a fragile heuristic.

---

## Features

### Live Token Tree (D3.js Force Graph)

Each node represents a token decision point:
- **Color**: Acceptance status (🟢 accepted / 🔴 rejected / 🟠 resampled / 🔵 bonus)
- **Size**: Entropy of the token distribution
- **Opacity**: Acceptance probability
- **Edges**: Sequential dependencies with animated flow

### Performance Dashboard

- **Acceptance Rate Gauge**: Real-time donut chart with history
- **Effective TPS**: Speedup relative to autoregressive baseline
- **Latency Breakdown**: Draft vs. Verify vs. Network overhead
- **Token Efficiency**: Accepted / Drafted / Bonus ratio

### Text Stream

Color-coded output with hover tooltips showing:
- Draft log-probability
- Target log-probability
- Acceptance probability
- Temperature-adjusted scores

---

## Quick Start

### Prerequisites

```bash
# Hardware
- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ RAM recommended (8GB works with smaller draft models)

# Software
- Python 3.11+
- Node.js 18+
- Cerebras API key (get one at cloud.cerebras.ai)
```

### Installation

```bash
# 1. Clone and setup Python environment
git clone <repo>
cd visualize_speculative_decoding
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your CEREBRAS_API_KEY

# 3. Install frontend
cd frontend && npm install && cd ..
```

### Running

```bash
# Terminal 1: Backend
source .venv/bin/activate
uvicorn backend.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev

# Open http://localhost:5173
```

---

## Configuration

```env
# Required
CEREBRAS_API_KEY=your-api-key-here
CEREBRAS_TARGET_MODEL=llama-3.3-70b

# Tuning (see docs/tuning.md for advanced settings)
DRAFT_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
SPECULATION_K=8              # Tokens to draft per round
TEMPERATURE=0.7
MAX_TOKENS=512

# Available target models:
# - llama3.1-8b     (fastest verification)
# - llama-3.3-70b   (best quality/speed tradeoff)
# - qwen-3-32b      (excellent for code)
```

---

## Draft Model Selection Guide

| Model | Params | Size | Speed | Quality | Best For |
|-------|--------|------|-------|---------|----------|
| `Llama-3.2-1B-Instruct-4bit` | 1B | 695MB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Quick tests, constrained memory |
| `Llama-3.2-3B-Instruct-4bit` | 3B | 1.8GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Recommended default** |
| `Qwen2.5-3B-Instruct-4bit` | 3B | 1.7GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Code generation, reasoning |

*All models use MLX 4-bit quantization via Apple's Neural Engine*

---

## The Math (Briefly)

**Modified Rejection Sampling** ensures target model distribution preservation:

```
For each draft token x_i from distribution q:
    Sample u ~ Uniform(0, 1)
    If u < min(1, p(x_i)/q(x_i)):
        Accept x_i
    Else:
        Resample from norm(max(0, p - q))

Where p = target distribution, q = draft distribution

Expected tokens per verification call:
    E[accept] = Σᵢ P(accept i tokens) × (i + 1ᵢ₌ₖ)
```

**Key insight**: The acceptance probability depends on how well the draft model approximates the target. With Cerebras' speed, we can afford deeper speculation (K=8+) without latency penalties.

---

## Benchmarks

Tested on MacBook Air M2 (16GB) → Cerebras Cloud:

| Configuration | Tokens/sec | Speedup | Avg Accepted/Round |
|--------------|-----------|---------|-------------------|
| Autoregressive (baseline) | 28 | 1.0x | - |
| Speculative (K=4, 1B draft) | 67 | 2.4x | 2.8 |
| Speculative (K=8, 3B draft) | 94 | 3.4x | 5.2 |
| Speculative (K=8, 3B draft, code) | 112 | 4.0x | 6.1 |

*Results vary by prompt domain and temperature. Lower temperature → higher acceptance rates.*

---

## Tech Stack

**Inference**
- [MLX-LM](https://github.com/ml-explore/mlx-lm) - Apple Silicon optimized LLM inference
- [Cerebras Python SDK](https://inference-docs.cerebras.ai/) - Ultra-low latency target model

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) - Async Python web framework
- [WebSocket](https://fastapi.tiangolo.com/advanced/websockets/) - Real-time streaming
- [Uvicorn](https://www.uvicorn.org/) - ASGI server

**Frontend**
- [React](https://react.dev/) + [Vite](https://vitejs.dev/) - Modern React build
- [D3.js](https://d3js.org/) - Token tree visualization
- [Recharts](https://recharts.org/) - Performance charts
- [Tailwind CSS](https://tailwindcss.com/) - Styling

---

## Roadmap

- [ ] Multi-draft speculative sampling
- [ ] Lookahead decoding integration
- [ ] Prompt caching optimization
- [ ] Multi-model ensemble drafting
- [ ] Quantization-aware training for draft models

---

<div align="center">

**Built to demonstrate the future of distributed LLM inference**

*If you find this interesting, let's talk about inference optimization at scale.*

</div>
