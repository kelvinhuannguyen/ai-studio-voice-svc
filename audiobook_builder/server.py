# --- Fix for Windows ConnectionResetError in asyncio ---
import sys
if sys.platform == "win32":
    import asyncio
    from functools import wraps
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig = _ProactorBasePipeTransport._call_connection_lost

        @wraps(_orig)
        def _silence(self, *args, **kwargs):
            try:
                return _orig(self, *args, **kwargs)
            except (ConnectionResetError, RuntimeError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _silence
    except ImportError:
        pass
# -------------------------------------------------------

import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth import verify_token
from routers import (
    audio,
    video,
    script,
    assets,
    export,
    project,
    flowkit,
    # Phase I — unified service additions
    health,
    voice_ref,
    transcribe,
    tts_clone,
    separate,
    ingest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Khởi chạy hệ thống Background Job Polling cho FlowKit...")
    asyncio.create_task(flowkit.poll_jobs_loop())
    yield


app = FastAPI(title="ai-studio-voice-svc (Audiobook Factory + Voice-Pro unified)", lifespan=lifespan)

# CORS — tighten for prod via env var ALLOWED_ORIGIN (comma-separated)
_allowed = os.getenv("ALLOWED_ORIGIN", "*")
_origins = [o.strip() for o in _allowed.split(",")] if _allowed != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Public (no auth) ────────────────────────────────────────────────────────
app.include_router(health.router)

# ─── Authenticated endpoints ────────────────────────────────────────────────
_auth_deps = [Depends(verify_token)]
app.include_router(audio.router,      dependencies=_auth_deps)
app.include_router(video.router,      dependencies=_auth_deps)
app.include_router(script.router,     dependencies=_auth_deps)
app.include_router(assets.router,     dependencies=_auth_deps)
app.include_router(export.router,     dependencies=_auth_deps)
app.include_router(project.router,    dependencies=_auth_deps)
app.include_router(flowkit.router,    dependencies=_auth_deps)
# Phase I — voice-svc bundle (Whisper / CosyVoice / Demucs / yt-dlp / voice-ref)
app.include_router(voice_ref.router,  dependencies=_auth_deps)
app.include_router(transcribe.router, dependencies=_auth_deps)
app.include_router(tts_clone.router,  dependencies=_auth_deps)
app.include_router(separate.router,   dependencies=_auth_deps)
app.include_router(ingest.router,     dependencies=_auth_deps)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
