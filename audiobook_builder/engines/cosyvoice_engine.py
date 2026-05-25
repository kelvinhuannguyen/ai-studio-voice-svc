"""CosyVoice2-0.5B wrapper — multilingual zero-shot voice cloning.

V2 worker-py's `vp_clone` posts here for EN/ZH/JA/KO clone. The reference
WAV is the SAME file used by OmniVoice (shared `Voice_ref/` folder, named
`<NAME>_voice.wav`), so once a user registers a voice via `/api/voice-ref`
both engines can use it.

NOTE on imports: CosyVoice's Python API has shifted between versions. The
import path below targets the official `FunAudioLLM/CosyVoice` package. If
your installed wheel differs, adjust the import — the rest of the wrapper
stays the same.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import numpy as np
from pydub import AudioSegment

log = logging.getLogger(__name__)

VOICE_REF_DIR = (
    Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "Voice_ref"
).resolve()


def _find_voice_ref(voice_name: str) -> Path:
    """Look up a registered ref WAV by character name, case-insensitive."""
    upper = voice_name.upper()
    candidates = [
        VOICE_REF_DIR / f"{upper}_voice.wav",
        VOICE_REF_DIR / f"{upper}_synthetic.wav",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"voice ref not found for {voice_name!r} — register via /api/voice-ref first"
    )


def _to_mp3_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert int16/float numpy samples → MP3 192k bytes."""
    if samples.dtype != np.int16:
        # CosyVoice returns float32 in [-1, 1] — scale to int16.
        clipped = np.clip(samples, -1.0, 1.0)
        samples = (clipped * 32767.0).astype(np.int16)
    audio = AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate="192k")
    return buf.getvalue()


class CosyVoiceEngine:
    """Lazy-init CosyVoice2-0.5B. ~4GB VRAM on CUDA fp16."""

    def __init__(self, *, device: str = "cuda") -> None:
        # The CosyVoice CLI module changes path between minor versions; try the
        # known locations in order.
        cosy_cls = None
        last_err: Exception | None = None
        for module_path in (
            "cosyvoice.cli.cosyvoice",   # FunAudioLLM upstream
            "cosyvoice.cosyvoice",       # alt packaging
        ):
            try:
                mod = __import__(module_path, fromlist=["CosyVoice2"])
                cosy_cls = getattr(mod, "CosyVoice2", None) or getattr(mod, "CosyVoice", None)
                if cosy_cls is not None:
                    break
            except Exception as e:
                last_err = e

        if cosy_cls is None:
            raise RuntimeError(
                f"CosyVoice not importable; pip install cosyvoice or check version. "
                f"Last error: {last_err}"
            )

        log.info(f"[CosyVoiceEngine] loading CosyVoice2-0.5B on {device}")
        self.device = device
        self.model = cosy_cls(
            "iic/CosyVoice2-0.5B",
            load_jit=False,
            load_trt=False,
            fp16=(device == "cuda"),
        )
        log.info("[CosyVoiceEngine] ready")

    def clone_line(
        self,
        text: str,
        voice_name: str,
        language: str = "en",
    ) -> bytes:
        """Zero-shot clone `text` using the registered ref WAV for `voice_name`.
        Returns MP3 192k bytes.
        """
        ref_path = _find_voice_ref(voice_name)
        log.info(
            f"[CosyVoiceEngine] clone speaker={voice_name} lang={language} "
            f"ref={ref_path.name} text-len={len(text)}"
        )
        # CosyVoice's inference_zero_shot returns an iterable of dicts each with
        # `tts_speech` (torch Tensor) — sample rate is on the model.
        chunks = list(
            self.model.inference_zero_shot(
                text,
                "",                # prompt text — empty for pure zero-shot
                str(ref_path),
                stream=False,
            )
        )
        if not chunks:
            raise RuntimeError("CosyVoice produced no audio")

        # Concatenate chunks if model streamed multiple.
        import torch  # local import to avoid hard dep when module loads

        speech = torch.cat([c["tts_speech"] for c in chunks], dim=-1)
        sample_rate = getattr(self.model, "sample_rate", 22050)
        samples_np = speech.cpu().numpy().squeeze().astype(np.float32)
        return _to_mp3_bytes(samples_np, sample_rate)
