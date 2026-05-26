"""Demucs htdemucs_ft wrapper — vocals/music source separation.

Used by V2 Phase J Quick Dub: separate vocals from background music so the
cloned dub can replace the original speech while preserving ambient audio.

We shell out to the Demucs CLI (most robust path; the Python API has changed
shape across releases). Output files are written to a temp dir, then we pass
them back to the caller as base64 — the caller is responsible for uploading
to R2 or wherever it wants the audio to live.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class DemucsEngine:
    """Lazy-init wrapper. The first call downloads Demucs model weights
    (~2GB) via `demucs` itself.
    """

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        # Sanity check Demucs is importable / on PATH.
        if shutil.which("demucs") is None:
            try:
                import demucs  # noqa: F401
            except Exception as e:
                raise RuntimeError(
                    "demucs not installed — `pip install demucs` "
                    f"(import error: {e})"
                )
        log.info(f"[DemucsEngine] ready (device={device})")

    async def separate(self, audio_url: str, *, model: str = "htdemucs_ft") -> dict:
        """Download → separate → return `{vocals_b64, music_b64, sample_rate}`.

        Format is WAV bytes base64-encoded. Caller decodes + uploads to R2.
        """
        log.info(f"[DemucsEngine] separate url={audio_url[:80]}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / f"input_{uuid.uuid4().hex}.wav"
            async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
                r = await client.get(audio_url)
                r.raise_for_status()
                in_path.write_bytes(r.content)

            out_dir = Path(tmpdir) / "out"
            out_dir.mkdir(exist_ok=True)
            cmd = [
                "demucs",
                "-n", model,
                "-o", str(out_dir),
                "--two-stems", "vocals",
                "-d", self.device,
                str(in_path),
            ]
            log.info(f"[DemucsEngine] $ {' '.join(cmd)}")
            await asyncio.to_thread(
                subprocess.run,
                cmd,
                check=True,
                capture_output=True,
            )

            # Demucs writes to out_dir/<model>/<input-stem>/{vocals,no_vocals}.wav
            stem_dir = next((out_dir / model).iterdir())
            vocals = stem_dir / "vocals.wav"
            music = stem_dir / "no_vocals.wav"
            if not vocals.exists() or not music.exists():
                raise RuntimeError(f"Demucs output missing in {stem_dir}")

            vocals_b64 = base64.b64encode(vocals.read_bytes()).decode("ascii")
            music_b64 = base64.b64encode(music.read_bytes()).decode("ascii")

        return {
            "vocals_b64": vocals_b64,
            "music_b64": music_b64,
            "format": "wav",
            "model": model,
        }
