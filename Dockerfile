# syntax=docker/dockerfile:1.6
# ──────────────────────────────────────────────────────────────────────────────
# ai-studio-voice-svc — unified voice service (OmniVoice + CosyVoice + Whisper
# + Demucs + yt-dlp) behind one FastAPI on port 8000.
# Built for NVIDIA GPU on FPT Smart Cloud (H100 / T4).
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

# Ubuntu 22.04's default python3 is 3.10. Don't try to swap to 3.11 — apt's
# python3-pip is tied to 3.10, and mixing the two leads to `uvicorn: not found`
# at runtime because pip installs to 3.10's site-packages while the symlinked
# python binary runs 3.11.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git ffmpeg ca-certificates curl libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python deps. Torch first (large, pin to CUDA-matched wheel).
COPY audiobook_builder/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir \
        torch==2.4.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
 && pip3 install --no-cache-dir -r /app/requirements.txt \
 # CosyVoice has no PyPI wheel — install from upstream. Allowed to fail at
 # image build; CosyVoice endpoints return 503 until manually fixed.
 && (pip3 install --no-cache-dir git+https://github.com/FunAudioLLM/CosyVoice.git \
     || echo "[Dockerfile] CosyVoice install skipped — clone endpoints will 503")

COPY . /app/
RUN mkdir -p /app/audiobook_builder/Voice_ref \
             /app/audiobook_builder/temp_audio \
             /app/audiobook_builder/output \
             /root/.cache

WORKDIR /app/audiobook_builder

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Invoke uvicorn via `python3 -m uvicorn` instead of the `uvicorn` shebang
# script — this avoids PATH issues if /usr/local/bin isn't in the entrypoint's
# PATH (some NVIDIA CUDA base images strip it).
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
