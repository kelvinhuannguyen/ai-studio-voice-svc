"""`POST /api/ingest-youtube` — yt-dlp download with host whitelist.

Phase J Quick Dub: paste YouTube/Vimeo URL → ingest → return base64 of
both the merged MP4 and an extracted MP3. Caller (V2 worker-py) uploads
to R2 and triggers Whisper transcribe + clone + Demucs separate next.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engines.yt_ingest import ingest_youtube

log = logging.getLogger(__name__)
router = APIRouter()


class IngestRequest(BaseModel):
    url: str = Field(..., description="YouTube or Vimeo URL")
    max_duration_s: int = Field(600, ge=10, le=1800)


class IngestResponse(BaseModel):
    title: str
    duration: float
    video_b64: str
    video_format: str
    audio_b64: str
    audio_format: str


@router.post("/api/ingest-youtube", response_model=IngestResponse)
async def api_ingest_youtube(req: IngestRequest) -> IngestResponse:
    try:
        result = await ingest_youtube(req.url, max_duration_s=req.max_duration_s)
        return IngestResponse(**result)
    except ValueError as e:
        # Host not allowed / duration too long
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        log.exception("ingest-youtube failed")
        raise HTTPException(500, detail=f"ingest-youtube failed: {e}")
