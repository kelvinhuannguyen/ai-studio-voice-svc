"""`POST /api/transcribe` — Whisper Large v3 → SRT.

Used by V2 export pipeline: each scene's audio_url → SRT → burned into MP4
final via ffmpeg subtitles filter.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from state import registry

log = logging.getLogger(__name__)
router = APIRouter()


class TranscribeRequest(BaseModel):
    audio_url: str = Field(..., description="Publicly fetchable audio (R2/CDN)")
    language: str | None = Field(
        None,
        description="ISO 639-1 (vi, en, zh, ...). None or 'auto' → auto-detect.",
    )
    model: Literal["large-v3", "large-v2", "medium", "small"] = "large-v3"
    output_format: Literal["srt", "json"] = "srt"


class TranscribeResponse(BaseModel):
    language: str
    language_probability: float | None = None
    duration: float | None = None
    segments: list[dict]
    srt: str


@router.post("/api/transcribe", response_model=TranscribeResponse)
async def api_transcribe(req: TranscribeRequest) -> TranscribeResponse:
    try:
        engine = await registry.get_whisper()
        result = await asyncio.to_thread(
            engine.transcribe, req.audio_url, req.language
        )
        return TranscribeResponse(**result)
    except Exception as e:
        log.exception("transcribe failed")
        raise HTTPException(500, detail=f"transcribe failed: {e}")
