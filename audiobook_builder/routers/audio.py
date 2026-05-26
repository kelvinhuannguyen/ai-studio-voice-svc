import asyncio
import os
import uuid
from io import BytesIO

import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from pydub import AudioSegment
from state import audio_gen, TEMP_DIR, OUTPUT_DIR

router = APIRouter()


class RenderLineRequest(BaseModel):
    id: int
    text: str
    speaker: str


class AssembleAudioRequest(BaseModel):
    filenames: list[str]


class TestVoiceRequest(BaseModel):
    text: str
    speaker: str


class SyntheticVoiceRequest(BaseModel):
    speaker: str
    instruct: str
    sample_text: str = (
        "Xin chào, hệ thống đã ghi nhận thành công chất giọng chuẩn của tôi. "
        "Với thiết lập này, tôi có thể truyền đạt mọi cung bậc cảm xúc một cách tự nhiên nhất, "
        "từ những lời thì thầm bí ẩn cho đến những đoạn cao trào dữ dội. "
        "Hãy lưu giữ bản ghi âm mẫu này thật kỹ để đảm bảo độ nhất quán tuyệt đối "
        "cho toàn bộ câu chuyện dài kỳ của chúng ta nhé."
    )


@router.get("/api/audio")
async def api_get_audio(path: str):
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    raise HTTPException(404, "Not found")


@router.post("/api/render-line")
async def api_render_line(req: RenderLineRequest):
    os.makedirs(TEMP_DIR, exist_ok=True)
    wav_path = os.path.join(TEMP_DIR, f"line_{req.id}.wav")
    success = audio_gen.generate(req.text, wav_path, req.speaker)
    if not success:
        raise HTTPException(500, "Render failed")
    try:
        info = sf.info(wav_path)
        duration = info.frames / info.samplerate
    except Exception:
        duration = 2.0
    return {"audio_path": wav_path, "file": wav_path, "duration": duration}


@router.post("/api/assemble-audio")
async def api_assemble_audio(req: AssembleAudioRequest):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "assembled.mp3")
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=800)
    for f in req.filenames:
        if os.path.exists(f):
            combined += AudioSegment.from_wav(f) + silence
    combined.export(output_path, format="mp3", bitrate="192k")
    return FileResponse(output_path, media_type="audio/mpeg")


@router.post("/api/test-voice")
async def api_test_voice(req: TestVoiceRequest):
    os.makedirs(TEMP_DIR, exist_ok=True)
    wav_path = os.path.join(TEMP_DIR, f"test_{req.speaker}.wav")
    if audio_gen.generate(req.text, wav_path, req.speaker):
        return FileResponse(wav_path, media_type="audio/wav")
    raise HTTPException(500, "Generate failed")


@router.post("/api/create-synthetic-voice")
async def api_create_synthetic_voice(req: SyntheticVoiceRequest):
    os.makedirs(TEMP_DIR, exist_ok=True)
    wav_path = os.path.join(TEMP_DIR, f"voice_{req.speaker}.wav")
    audio_gen.generate(req.sample_text, wav_path, req.speaker)
    return {"status": "success"}


# ═════════════════════════════════════════════════════════════════════════════
# Phase I — MP3-direct + batch endpoints used by V2 worker-py
# ═════════════════════════════════════════════════════════════════════════════


class RenderLineMp3Request(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    speaker: str | None = "NARRATOR"


def _generate_wav_sync(text: str, speaker: str) -> str:
    """Wrap audio_gen.generate (which writes a WAV file synchronously) so we
    can call it from an async endpoint via `asyncio.to_thread`."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    wav_path = os.path.join(TEMP_DIR, f"render_{uuid.uuid4().hex}.wav")
    ok = audio_gen.generate(text, wav_path, speaker or "NARRATOR")
    if not ok or not os.path.exists(wav_path):
        raise RuntimeError("OmniVoice generate failed")
    return wav_path


def _wav_to_mp3_bytes(wav_path: str) -> bytes:
    audio = AudioSegment.from_wav(wav_path)
    buf = BytesIO()
    audio.export(buf, format="mp3", bitrate="192k")
    return buf.getvalue()


@router.post("/api/render-line-mp3")
async def api_render_line_mp3(req: RenderLineMp3Request) -> Response:
    """Render one TTS line via OmniVoice and stream MP3 192k bytes back.

    V2 worker-py's `kj_render_line()` uses this — avoids the WAV→assemble
    round-trip from the legacy `/api/render-line` endpoint.
    """
    try:
        wav_path = await asyncio.to_thread(
            _generate_wav_sync, req.text, req.speaker or "NARRATOR"
        )
        mp3 = await asyncio.to_thread(_wav_to_mp3_bytes, wav_path)
    except Exception as e:
        raise HTTPException(500, detail=f"render-line-mp3 failed: {e}")
    finally:
        # Clean up the temp WAV — caller doesn't need it.
        try:
            if "wav_path" in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
        except OSError:
            pass
    return Response(content=mp3, media_type="audio/mpeg")


class RenderBatchItem(BaseModel):
    text: str
    speaker: str | None = None
    emotion: str | None = None
    voice_id: str | None = None


class RenderBatchRequest(BaseModel):
    lines: list[RenderBatchItem] = Field(..., min_length=1, max_length=200)
    output: str = "mp3-per-line"  # only "mp3-per-line" supported in v1


@router.post("/api/render-batch")
async def api_render_batch(req: RenderBatchRequest) -> dict:
    """Render multiple lines sequentially (OmniVoice is not reentrant on a
    single GPU). Returns a list of MP3 file URLs that the caller can fetch.

    Each MP3 is written to OUTPUT_DIR and served via the existing
    `GET /api/audio?path=...` endpoint.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    urls: list[str] = []
    for idx, line in enumerate(req.lines):
        speaker = line.voice_id or line.speaker or "NARRATOR"
        try:
            wav_path = await asyncio.to_thread(_generate_wav_sync, line.text, speaker)
            mp3_bytes = await asyncio.to_thread(_wav_to_mp3_bytes, wav_path)
        except Exception as e:
            raise HTTPException(
                500, detail=f"render-batch line {idx} ({speaker}) failed: {e}"
            )
        out_name = f"batch_{uuid.uuid4().hex}.mp3"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "wb") as f:
            f.write(mp3_bytes)
        try:
            os.remove(wav_path)
        except OSError:
            pass
        # The /api/audio endpoint accepts an absolute path; V2 follows it
        # immediately so a local path string works for in-cluster calls.
        urls.append(f"/api/audio?path={out_path}")

    return {"urls": urls, "count": len(urls)}
