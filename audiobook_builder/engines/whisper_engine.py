"""faster-whisper wrapper.

V2 worker-py's `vp_transcribe` posts here for auto-subtitle generation. The
service returns both the structured segments AND a precomputed SRT string —
V2 prefers SRT (passed straight to ffmpeg's `subtitles` filter for burn-in).
"""
from __future__ import annotations

import io
import logging
from typing import Iterable

import httpx

log = logging.getLogger(__name__)


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: Iterable[dict]) -> str:
    out: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = _format_srt_timestamp(float(seg["start"]))
        end = _format_srt_timestamp(float(seg["end"]))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        out.append(f"{idx}\n{start} --> {end}\n{text}")
    return "\n\n".join(out)


class WhisperEngine:
    """Lazy-init faster-whisper Large v3.

    Memory: ~3GB VRAM on CUDA float16, ~5GB CPU int8.
    Throughput on T4: ~1x real-time per request.
    """

    def __init__(self, *, device: str = "cuda", model_size: str = "large-v3") -> None:
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        log.info(f"[WhisperEngine] loading {model_size} on {device} ({compute_type})")
        self.device = device
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log.info("[WhisperEngine] ready")

    def transcribe(self, audio_url: str, language: str | None = None) -> dict:
        """Download `audio_url` → run Whisper → return `{segments, language, srt}`.

        Caller passes `language=None` to let Whisper auto-detect.
        """
        log.info(f"[WhisperEngine] transcribe url={audio_url[:80]}... lang={language!r}")
        with httpx.Client(timeout=300.0, follow_redirects=True) as client:
            r = client.get(audio_url)
            r.raise_for_status()
            audio_bytes = r.content

        buf = io.BytesIO(audio_bytes)
        # `language=None` → auto-detect; `vad_filter=True` → skip silence (faster
        # + cleaner SRT timing).
        segments_iter, info = self.model.transcribe(
            buf,
            language=language if (language and language != "auto") else None,
            vad_filter=True,
            beam_size=5,
        )
        seg_list = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments_iter
        ]
        srt = segments_to_srt(seg_list)
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": seg_list,
            "srt": srt,
        }
