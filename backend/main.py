"""FastAPI application with WebSocket endpoint for speculative decoding visualization."""

from __future__ import annotations

import asyncio
import logging
import traceback

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .draft_model import DraftModel
from .schemas import StartGenerationRequest
from .speculator import Speculator
from .target_model import TargetModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SpeculatoViz", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global draft model singleton (loaded once at startup)
_draft_model: DraftModel | None = None


@app.on_event("startup")
async def startup():
    global _draft_model
    logger.info(f"Loading draft model: {settings.draft_model}")
    _draft_model = DraftModel(settings.draft_model)
    await asyncio.to_thread(_draft_model.load)
    logger.info("Draft model loaded successfully")
    logger.info(f"Target model: {settings.cerebras_target_model}")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "draft_model": settings.draft_model,
        "target_model": settings.cerebras_target_model,
        "draft_loaded": _draft_model is not None,
    }


@app.get("/api/test-draft")
async def test_draft():
    """Diagnostic endpoint: test that the draft model can generate tokens."""
    if _draft_model is None:
        return {"error": "Draft model not loaded"}
    try:
        context_ids = _draft_model.apply_chat_template("Say hello.")
        logger.info(f"Test draft: context has {len(context_ids)} tokens")
        tokens = await asyncio.to_thread(
            _draft_model.generate_draft_tokens, context_ids, 3, 0.7
        )
        return {
            "status": "ok",
            "tokens": [
                {"token": t.token_str, "logprob": t.logprob, "entropy": t.entropy}
                for t in tokens
            ],
        }
    except Exception as e:
        logger.exception("test-draft failed")
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.websocket("/ws/tokens")
async def websocket_tokens(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Wait for a generation request
            data = await websocket.receive_json()
            request = StartGenerationRequest(**data)

            logger.info(
                f"Generation request: prompt={request.prompt[:50]}... "
                f"k={request.k} temp={request.temperature} max={request.max_tokens}"
            )

            target = TargetModel(
                model=settings.cerebras_target_model,
                api_key=settings.cerebras_api_key,
            )
            speculator = Speculator(draft=_draft_model, target=target)

            try:
                async for event in speculator.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    k=request.k,
                ):
                    event_data = event.model_dump()
                    logger.info(f"Sending event: type={event_data.get('type')} round={event_data.get('round')}")
                    await websocket.send_json(event_data)
            except Exception as e:
                logger.exception(f"Generation error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def run():
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    run()
