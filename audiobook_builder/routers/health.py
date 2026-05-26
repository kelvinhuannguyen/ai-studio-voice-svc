"""Public health endpoint — no auth required.

V2 worker-py polls this every ~30s with a cache TTL to decide whether to route
TTS / transcribe requests here (vs. falling back to FPT.AI / ElevenLabs).
"""
from fastapi import APIRouter

from state import registry

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": registry.device,
        "cuda_available": registry.cuda_available,
        "models": registry.loaded_status(),
        # Legacy field expected by Phase G+/H worker-py healthchecks.
        "model_loaded": registry.loaded_status().get("omnivoice", False),
    }
