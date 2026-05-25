"""Shared-secret auth dependency for the voice service.

V2 worker-py calls this service across the network. We use a single
shared-secret token sent as `X-VOICE-SVC-Token`. Legacy headers from the
Phase G+/H split (`X-KJ-Token`, `X-VP-Token`) are still accepted so older
worker-py builds continue working during migration.

In dev — set the env var to an empty string (or leave it unset) to bypass
auth entirely. In prod — set a 32+ char random value and keep it in
worker-py's `VOICE_SVC_TOKEN`.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

VOICE_SVC_AUTH_TOKEN = os.getenv("VOICE_SVC_AUTH_TOKEN") or ""


async def verify_token(
    x_voice_svc_token: str | None = Header(None, alias="X-VOICE-SVC-Token"),
    # Backward-compat headers — accept any one of these.
    x_kj_token: str | None = Header(None, alias="X-KJ-Token"),
    x_vp_token: str | None = Header(None, alias="X-VP-Token"),
) -> None:
    if not VOICE_SVC_AUTH_TOKEN:
        return  # dev mode bypass
    presented = x_voice_svc_token or x_kj_token or x_vp_token
    if presented != VOICE_SVC_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")
