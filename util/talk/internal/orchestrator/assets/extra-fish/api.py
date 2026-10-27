"""Private batch endpoint. Talk owns admission, model offload and restoration."""
import asyncio
import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal

from speech import chunks, synthesize

backend = None
busy = False
quiescing = False
last_used = time.monotonic()
BACKEND = "http://127.0.0.1:8794"


@contextlib.asynccontextmanager
async def lifespan(app):
    global backend
    backend = subprocess.Popen(["sgl-omni", "serve", "--config", "/opt/fish/config.json", "--host", "127.0.0.1", "--port", "8794", "--allowed-local-media-path", "/opt/fish"], start_new_session=True)
    try:
        yield
    finally:
        if backend.poll() is None:
            os.killpg(backend.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(backend.wait, timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(backend.pid, signal.SIGKILL)
                await asyncio.to_thread(backend.wait)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(BACKEND + "/health")
        if response.status_code != 200:
            raise ValueError("not ready")
    except Exception as error:
        raise HTTPException(503, "Fish model warming up") from error
    return {"status": "ok", "model": "watchpen/drama-3-int4-koen160", "sampler": "ca9e878-codebook30-crossing-token", "speech_policy": "native-tags-v1"}


@app.get("/v1/runtime/memory")
def memory():
    return {"status": "ok", "ready": True, "busy": busy, "queued": 0, "quiescing": quiescing, "idle_for_seconds": time.monotonic() - last_used}


@app.post("/v1/runtime/{operation}")
def lifecycle(operation: str):
    global quiescing
    if operation not in ("quiesce", "resume"):
        raise HTTPException(404)
    if operation == "quiesce" and busy:
        raise HTTPException(409, "Speech generation is active")
    quiescing = operation == "quiesce"
    return {"status": "ok"}


from voice_catalog import VoiceID, reference, get_voice


class Voice(BaseModel):
    preset: VoiceID = "sohee"
    presets: list[VoiceID] = Field(default_factory=list, max_length=16)
    sentence_pause: float = Field(default=.18, ge=.05, le=1)
    session_id: str = Field(default="", max_length=128)
    reference_revision: int | None = Field(default=None, ge=1)
    preparing_reference: bool = False
    pinned_reference: bool = False
    continuous: bool = False


class Batch(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=16)
    voice: Voice = Field(default_factory=Voice)


@app.post("/v1/audio/batch")
async def batch(payload: Batch, request: Request):
    global busy, last_used
    if busy or quiescing:
        raise HTTPException(409, "Fish is busy or quiescing")
    if sum(len(text) for text in payload.texts) > 1800 or any(not text.strip() or len(text) > 500 for text in payload.texts):
        raise HTTPException(422, "Each narration must contain 1–500 characters")
    try:
        for text in payload.texts:
            chunks(text, continuous=payload.voice.continuous)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    selected = payload.voice.presets or [payload.voice.preset] * len(payload.texts)
    if len(selected) != len(payload.texts):
        raise HTTPException(422, "Voice and text counts must match")
    try:
        metadata = [get_voice(voice_id, payload.voice.session_id, payload.voice.reference_revision,
                              allow_pending=payload.voice.preparing_reference, allow_history=payload.voice.pinned_reference) for voice_id in selected]
        references = [reference(voice_id, payload.voice.session_id, payload.voice.reference_revision,
                                allow_pending=payload.voice.preparing_reference, allow_history=payload.voice.pinned_reference) for voice_id in selected]
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error
    def generate():
        items = []
        for text, voice_id, (path, transcript), voice_data in zip(payload.texts, selected, references, metadata):
            item = synthesize(text, BACKEND, str(path), pause=payload.voice.sentence_pause, reference_text=transcript, continuous=payload.voice.continuous)
            item.update(voice_preset=voice_id, reference_sha256=voice_data['reference_sha256'], reference_revision=voice_data['revision'])
            items.append(item)
        return items
    busy = True
    task = asyncio.create_task(asyncio.to_thread(generate))
    try:
        while not task.done():
            if await request.is_disconnected():
                # Closing the HTTP connection alone does not guarantee CUDA work
                # stops. Kill the private model group; Talk then removes it.
                os.killpg(backend.pid, signal.SIGTERM)
                raise HTTPException(499, "Client disconnected")
            await asyncio.sleep(.25)
        reported = selected[0] if len(set(selected)) == 1 else None
        return {"items": await task, "model": "drama-3-int4-koen160", "temperature": 1.0, "top_p": .9, "codebook_top_k": 30, "voice_preset": reported, "reference_sha256": metadata[0]["reference_sha256"] if reported else None, "reference_revision": metadata[0]["revision"] if reported else None, "reference_source": metadata[0].get("reference_source", "음성 참조") if reported else "multiple references"}
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        busy = False
        last_used = time.monotonic()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8726)
