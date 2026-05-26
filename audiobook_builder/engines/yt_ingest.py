"""yt-dlp wrapper — download a video + extracted audio for Phase J Quick Dub.

Host whitelist enforced at this layer too (the router also checks). Limits:
  · max_duration_s: caps download length so a malicious URL can't blow disk
  · output format pinned: mp4 video + m4a/mp3 audio for predictable downstream
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
from urllib.parse import urlparse

log = logging.getLogger(__name__)

ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "vimeo.com",
    "www.vimeo.com",
}


def _check_host(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"host {host!r} not allowed (whitelist: {sorted(ALLOWED_HOSTS)})")


async def ingest_youtube(url: str, *, max_duration_s: int = 600) -> dict:
    """Download via yt-dlp. Returns `{video_b64, audio_b64, title, duration, ext}`
    where the b64 strings are the file contents. Caller uploads to R2.

    `max_duration_s` defaults to 10 minutes — caller can override but we cap
    at 30 minutes hard.
    """
    _check_host(url)
    max_duration_s = min(int(max_duration_s), 1800)

    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not installed — `pip install yt-dlp`")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, f"yt_{uuid.uuid4().hex}.%(ext)s")
        # Probe metadata first so we can reject early if too long.
        meta = await asyncio.to_thread(
            subprocess.run,
            ["yt-dlp", "--dump-json", "--no-warnings", url],
            check=True,
            capture_output=True,
            text=True,
        )
        import json

        info = json.loads(meta.stdout.strip().split("\n")[-1])
        duration = float(info.get("duration") or 0)
        title = str(info.get("title") or "untitled")
        if duration > max_duration_s:
            raise ValueError(
                f"video duration {duration:.0f}s exceeds limit {max_duration_s}s"
            )

        # Download both: best video+audio mp4 AND audio-only m4a.
        await asyncio.to_thread(
            subprocess.run,
            [
                "yt-dlp",
                "-f", "bv*+ba/b",
                "--merge-output-format", "mp4",
                "-o", out_template,
                "--no-warnings",
                url,
            ],
            check=True,
            capture_output=True,
        )
        audio_template = out_template.replace(".%(ext)s", ".audio.%(ext)s")
        await asyncio.to_thread(
            subprocess.run,
            [
                "yt-dlp",
                "-f", "ba/b",
                "-x", "--audio-format", "mp3",
                "-o", audio_template,
                "--no-warnings",
                url,
            ],
            check=True,
            capture_output=True,
        )

        # Find the resulting files (yt-dlp templates the ext, so glob).
        files = list(Path(tmpdir).iterdir())
        video_path = next(
            (p for p in files if p.suffix == ".mp4" and ".audio." not in p.name), None
        )
        audio_path = next((p for p in files if p.suffix == ".mp3"), None)
        if not video_path or not audio_path:
            raise RuntimeError(f"yt-dlp output missing in {tmpdir}: {[p.name for p in files]}")

        video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")

    return {
        "title": title,
        "duration": duration,
        "video_b64": video_b64,
        "video_format": "mp4",
        "audio_b64": audio_b64,
        "audio_format": "mp3",
    }
