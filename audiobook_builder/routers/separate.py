"""`POST /api/separate-tracks` — Demucs htdemucs_ft vocals/music split.

Phase J Quick Dub: separate vocals so the cloned dub can replace original
speech while preserving background music. Returned as base64-encoded WAV;
caller (V2 worker-py) uploads to R2 and stores R2 URLs.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from state import registry

log = logging.getLogger(__name__)
router = APIRouter()


class SeparateRequest(BaseModel):
    audio_url: str = Field(..., description="Publicly fetchable audio URL")
    model: Literal["htdemucs_ft", "htdemucs", "mdx_extra"] = "htdemucs_ft"


class SeparateResponse(BaseModel):
    vocals_b64: str
    music_b64: str
    format: str = "wav"
    model: str


@router.post("/api/separate-tracks", response_model=SeparateResponse)
async def api_separate_tracks(req: SeparateRequest) -> SeparateResponse:
    try:
        engine = await registry.get_demucs()
        result = await engine.separate(req.audio_url, model=req.model)
        return SeparateResponse(**result)
    except Exception as e:
        log.exception("separate-tracks failed")
        raise HTTPException(500, detail=f"separate-tracks failed: {e}")
