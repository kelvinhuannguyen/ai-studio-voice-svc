"""`POST /api/clone-multilingual` — CosyVoice2 zero-shot clone (EN/ZH/JA/KO).

V2 worker-py's `vp_clone` posts here. The reference WAV is the SAME file used
by OmniVoice (shared `Voice_ref/` folder), so registering once via
`/api/voice-ref` enables both VN OmniVoice and multilingual CosyVoice clones.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from state import registry

log = logging.getLogger(__name__)
router = APIRouter()


class CloneRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    voice_name: str = Field(..., description="Registered voice ref name (case-insensitive)")
    language: Literal["en", "zh", "ja", "ko"] = "en"
    output_format: Literal["mp3"] = "mp3"


@router.post("/api/clone-multilingual")
async def api_clone_multilingual(req: CloneRequest) -> Response:
    """Returns raw MP3 bytes (Content-Type: audio/mpeg). V2 streams these
    straight to R2 — no JSON wrapping needed for the happy path.
    """
    try:
        engine = await registry.get_cosyvoice()
        mp3_bytes = await asyncio.to_thread(
            engine.clone_line, req.text, req.voice_name, req.language
        )
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        log.exception("clone-multilingual failed")
        raise HTTPException(500, detail=f"clone-multilingual failed: {e}")
    return Response(content=mp3_bytes, media_type="audio/mpeg")
