"""Voice reference registration — `POST /api/voice-ref`.

The service pulls a WAV/MP3/M4A from `url`, normalizes it to 24 kHz mono WAV,
writes it to `audiobook_builder/Voice_ref/{NAME}_{mode}.wav`, then triggers
`audio_gen.voice_cache` to re-scan the folder so the new voice is immediately
available for OmniVoice cloning. CosyVoice reads the same folder.
"""
from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pydub import AudioSegment

from state import audio_gen

log = logging.getLogger(__name__)
router = APIRouter()

# `Voice_ref/` lives next to `audio_generator.py` (audio_gen scans it on init).
VOICE_REF_DIR = (
    Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "Voice_ref"
).resolve()


class VoiceRefRequest(BaseModel):
    url: str = Field(..., description="Publicly fetchable audio URL (R2 / CDN)")
    voice_name: str = Field(..., min_length=1, max_length=64)
    mode: Literal["voice", "synthetic"] = "voice"


class VoiceRefResponse(BaseModel):
    success: bool = True
    voice_name: str
    registered_path: str
    duration_seconds: float | None = None


def _reload_voice_cache() -> None:
    """Re-scan Voice_ref/ and refresh `audio_gen.voice_cache` so the newly
    registered file is picked up without restarting the server."""
    try:
        audio_gen.voice_cache.clear()
        if not VOICE_REF_DIR.exists():
            return
        for file in os.listdir(VOICE_REF_DIR):
            full = str(VOICE_REF_DIR / file)
            if file.endswith("_synthetic.wav"):
                audio_gen.voice_cache[file.split("_synthetic")[0].lower()] = full
            elif file.endswith("_voice.wav"):
                audio_gen.voice_cache[file.split("_voice")[0].lower()] = full
    except Exception as e:
        log.warning(f"[voice_ref] reload_voice_cache failed: {e}")


@router.post("/api/voice-ref", response_model=VoiceRefResponse)
async def register_voice_ref(req: VoiceRefRequest) -> VoiceRefResponse:
    VOICE_REF_DIR.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(req.url)
            resp.raise_for_status()
        audio = AudioSegment.from_file(BytesIO(resp.content))
        audio = audio.set_frame_rate(24000).set_channels(1)
        out_path = VOICE_REF_DIR / f"{req.voice_name.upper()}_{req.mode}.wav"
        audio.export(out_path, format="wav")
        duration = len(audio) / 1000.0
    except Exception as e:
        log.exception("voice-ref ingest failed")
        raise HTTPException(500, detail=f"voice-ref failed: {e}")

    _reload_voice_cache()
    log.info(
        f"[voice-ref] registered {req.voice_name} ({req.mode}) "
        f"→ {out_path.name} ({duration:.1f}s)"
    )
    return VoiceRefResponse(
        voice_name=req.voice_name.upper(),
        registered_path=str(out_path),
        duration_seconds=duration,
    )
