"""FastAPI application with WebSocket endpoint for speculative decoding visualization."""

from __future__ import annotations

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .draft_model import DraftModel
from .schemas import ErrorEvent, StartGenerationRequest
from .speculator import Speculator
from .target_model import TargetModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singletons populated during lifespan
_draft_model: DraftModel | None = None
_target_model: TargetModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _draft_model, _target_model
    settings = get_settings()

    logger.info(f"Loading draft model: {settings.draft_model}")
    _draft_model = DraftModel(settings.draft_model)
    await asyncio.to_thread(_draft_model.load)
    logger.info("Draft model loaded successfully")

    logger.info(f"Target model: {settings.cerebras_target_model}")
    _target_model = TargetModel(
        model=settings.cerebras_target_model,
        api_key=settings.cerebras_api_key,
    )

    yield

    _draft_model = None
    _target_model = None


app = FastAPI(title="SpeculatoViz", version="0.1.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    settings = get_settings()
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


def _resolve_request(request: StartGenerationRequest) -> dict:
    """Merge request params with config defaults for any unset fields."""
    settings = get_settings()
    return {
        "prompt": request.prompt,
        "max_tokens": request.max_tokens if request.max_tokens is not None else settings.max_tokens,
        "temperature": request.temperature if request.temperature is not None else settings.temperature,
        "k": request.k if request.k is not None else settings.speculation_k,
    }


@app.websocket("/ws/tokens")
async def websocket_tokens(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Wait for a generation request
            data = await websocket.receive_json()
            request = StartGenerationRequest(**data)
            params = _resolve_request(request)

            logger.info(
                f"Generation request: prompt={params['prompt'][:50]}... "
                f"k={params['k']} temp={params['temperature']} max={params['max_tokens']}"
            )

            speculator = Speculator(draft=_draft_model, target=_target_model)

            client_disconnected = False
            try:
                async for event in speculator.generate(
                    prompt=params["prompt"],
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    k=params["k"],
                ):
                    event_data = event.model_dump()
                    logger.info(
                        f"Sending event: type={event_data.get('type')} round={event_data.get('round')}"
                    )
                    try:
                        await websocket.send_json(event_data)
                    except WebSocketDisconnect:
                        logger.info(
                            "Client disconnected during generation, stopping..."
                        )
                        client_disconnected = True
                        break
            except WebSocketDisconnect:
                logger.info("Client disconnected during generation")
                client_disconnected = True
            except Exception as e:
                logger.exception(f"Generation error: {e}")
                try:
                    await websocket.send_json(
                        ErrorEvent(message=str(e)).model_dump()
                    )
                except WebSocketDisconnect:
                    logger.info("Client disconnected while sending error")
                    client_disconnected = True

            # Break out of outer loop if client disconnected
            if client_disconnected:
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json(ErrorEvent(message=str(e)).model_dump())
        except Exception:
            pass


def run():
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    run()
