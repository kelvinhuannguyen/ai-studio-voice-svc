# syntax=docker/dockerfile:1.6
# ──────────────────────────────────────────────────────────────────────────────
# ai-studio-voice-svc — unified voice service (OmniVoice + CosyVoice + Whisper
# + Demucs + yt-dlp) behind one FastAPI on port 8000.
# Built for NVIDIA GPU on FPT Smart Cloud (T4 16GB recommended).
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git ffmpeg ca-certificates curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install Python deps. Torch first (large, pin to CUDA-matched wheel).
COPY audiobook_builder/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
        torch==2.3.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir -r /app/requirements.txt \
 # CosyVoice has no PyPI wheel — install from upstream (skip if image build
 # without network access for CosyVoice; service still works for everything else).
 && pip install --no-cache-dir \
        git+https://github.com/FunAudioLLM/CosyVoice.git \
        || echo "[Dockerfile] CosyVoice install skipped — clone endpoints will return 503 until installed"

COPY . /app/
RUN mkdir -p /app/audiobook_builder/Voice_ref \
             /app/audiobook_builder/temp_audio \
             /app/audiobook_builder/output \
             /root/.cache

WORKDIR /app/audiobook_builder

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
