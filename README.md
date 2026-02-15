# SpeculatoViz

Real-time visualization of speculative decoding, where a small local draft model (MLX-LM on Apple Silicon) proposes tokens and a large target model (Cerebras Cloud API) verifies them in batch.

**Key insight**: A 16GB MacBook Air running a tiny 3B model can accelerate a 70B+ cloud model by 2-4x through speculative decoding, with the bottleneck being network RTT rather than local compute.

## Architecture

```
React Frontend (Vite)          FastAPI Backend
┌────────────────────┐         ┌──────────────────────────┐
│ Prompt Input       │  WS     │ Speculator (orchestrator) │
│ Token Tree (D3)    │◄───────►│ ├─ Draft Model (MLX-LM)  │
│ Text Output        │         │ ├─ Rejection Sampling     │
│ KPI Dashboard      │         │ └─ Target Model (Cerebras)│
└────────────────────┘         └──────────────────────────┘
```

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Node.js 18+
- [Cerebras API key](https://cloud.cerebras.ai/)

## Setup

### 1. Backend

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Cerebras API key and target model
```

### 2. Frontend

```bash
cd frontend
npm install
```

## Configuration

Edit `.env` with your settings:

```env
# Required
CEREBRAS_API_KEY=your-key-here
CEREBRAS_TARGET_MODEL=llama-3.3-70b

# Optional
DRAFT_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
SPECULATION_K=8
TEMPERATURE=0.7
MAX_TOKENS=512
```

Available target models on Cerebras: `llama3.1-8b`, `llama-3.3-70b`, `qwen-3-32b`

## Running

### Start backend (from project root)

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

The first run will download the draft model (~1.8GB) from HuggingFace.

### Start frontend (separate terminal)

```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## How It Works

Each generation round:
1. **Draft**: The local MLX model generates K=8 candidate tokens (~milliseconds)
2. **Verify**: All K tokens are sent to Cerebras for batch verification (single API call)
3. **Compare**: Modified rejection sampling accepts/rejects each draft token by comparing log-probability distributions
4. **Visualize**: Events stream to the frontend via WebSocket, updating the token tree and metrics in real-time

### Visualization Components

- **Token Tree**: D3.js tree showing draft/verify decisions. Node color = status (green=accepted, red=rejected, orange=resampled, blue=bonus). Node size = entropy. Opacity = acceptance probability.
- **Text Output**: Color-coded generated text with token-level tooltips.
- **Acceptance Gauge**: Donut chart showing current acceptance rate.
- **TPS Chart**: Effective tokens/sec vs estimated autoregressive baseline.
- **Latency Chart**: Draft (local) vs verify (API) latency comparison.
- **Speedup Indicator**: Real-time speedup multiplier.

## Draft Model Options

| Model | Params | Size | Use case |
|-------|--------|------|----------|
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | 1B | 695MB | Maximum compatibility |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 3B | 1.81GB | Recommended default |
| `mlx-community/Qwen2.5-3B-Instruct-4bit` | 3B | 1.74GB | Higher quality drafts |

Set via `DRAFT_MODEL` in `.env`.
