"""Shared server singletons.

  · `audio_gen`  — OmniVoice (eager-loaded on import, existing behaviour).
  · `registry`   — ModelRegistry for Whisper / CosyVoice / Demucs (lazy-loaded,
    with LRU-style eviction to stay under the GPU VRAM budget on T4 16GB).

The legacy `audio_gen` global stays for backward compat with `routers/audio.py`
and other modules that import it directly. New Phase I endpoints use `registry`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from typing import Any

from audio_generator import AudioGenerator

log = logging.getLogger(__name__)

TEMP_DIR = "temp_audio"
OUTPUT_DIR = "output"
PROFILE_FILE = os.path.join(os.path.dirname(__file__), "project_profile.json")

# Soft VRAM ceiling (GB). When `torch.cuda.memory_allocated()` exceeds this we
# evict the least-recently-used model from the registry. T4 has 16GB total —
# leaving ~4GB headroom for activations / buffers.
VRAM_CEILING_GB = float(os.getenv("VOICE_SVC_VRAM_CEILING_GB", "12.0"))

print("Booting server — loading AI model...")
audio_gen = AudioGenerator()
print("Boot complete!")

flowkit_state = {
    "flowKey": None,
    "callbackSecret": os.getenv("FLOWKIT_CALLBACK_SECRET", "change-me"),
    "active_ws": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model registry — Phase I (Whisper, CosyVoice, Demucs)
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    """Lazy-loads inference engines on first use. Evicts LRU when VRAM crosses
    the ceiling. Use `await registry.get_<name>()` from any async endpoint.

    `omnivoice` is tracked here too even though it eager-loads at import time —
    so /health can report it consistently with the others.
    """

    def __init__(self) -> None:
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()
        # OmniVoice was loaded eagerly above — record it so /health reflects state.
        self._models["omnivoice"] = audio_gen
        self.device, self.cuda_available = self._detect_device()

    @staticmethod
    def _detect_device() -> tuple[str, bool]:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", True
        except Exception:
            pass
        return "cpu", False

    def _vram_used_gb(self) -> float:
        if not self.cuda_available:
            return 0.0
        try:
            import torch

            return torch.cuda.memory_allocated() / 1024**3
        except Exception:
            return 0.0

    async def _evict_if_needed(self, want_keep: str) -> None:
        """Drop oldest model(s) (other than `want_keep`, and never OmniVoice
        since existing routers rely on it staying loaded) until VRAM is under
        the ceiling."""
        while self._vram_used_gb() > VRAM_CEILING_GB and len(self._models) > 1:
            victim = next(
                (k for k in self._models if k not in (want_keep, "omnivoice")), None
            )
            if not victim:
                break
            log.info(
                f"[Registry] evicting {victim} (VRAM "
                f"{self._vram_used_gb():.1f}GB > {VRAM_CEILING_GB}GB)"
            )
            del self._models[victim]
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    async def _touch(self, name: str) -> None:
        if name in self._models:
            self._models.move_to_end(name)

    async def get_omnivoice(self):
        await self._touch("omnivoice")
        return self._models["omnivoice"]

    async def get_whisper(self):
        async with self._lock:
            await self._evict_if_needed(want_keep="whisper")
            if "whisper" not in self._models:
                log.info("[Registry] loading Whisper Large v3 ...")
                from engines.whisper_engine import WhisperEngine

                self._models["whisper"] = WhisperEngine(device=self.device)
            await self._touch("whisper")
            return self._models["whisper"]

    async def get_cosyvoice(self):
        async with self._lock:
            await self._evict_if_needed(want_keep="cosyvoice")
            if "cosyvoice" not in self._models:
                log.info("[Registry] loading CosyVoice2-0.5B ...")
                from engines.cosyvoice_engine import CosyVoiceEngine

                self._models["cosyvoice"] = CosyVoiceEngine(device=self.device)
            await self._touch("cosyvoice")
            return self._models["cosyvoice"]

    async def get_demucs(self):
        async with self._lock:
            await self._evict_if_needed(want_keep="demucs")
            if "demucs" not in self._models:
                log.info("[Registry] loading Demucs htdemucs_ft ...")
                from engines.demucs_engine import DemucsEngine

                self._models["demucs"] = DemucsEngine(device=self.device)
            await self._touch("demucs")
            return self._models["demucs"]

    def loaded_status(self) -> dict[str, bool]:
        known = ("omnivoice", "whisper", "cosyvoice", "demucs")
        return {k: (k in self._models) for k in known}


registry = ModelRegistry()
