"""Private music worker; Talk owns GPU admission and image restoration."""
import asyncio
import base64
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

app = FastAPI()
busy = False
quiescing = False
last_used = time.monotonic()
backend = "ace"


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["ace-turbo", "ace-xl"] = "ace-turbo"
    prompt: str = Field(min_length=5, max_length=2000)
    duration_seconds: int = Field(default=30, ge=10, le=120)
    seed: int = Field(default=42, ge=0, le=2147483647)
    bpm: int | None = Field(default=None, ge=30, le=300)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": backend, "busy": busy}


@app.get("/v1/runtime/memory")
async def memory():
    return {"status": "ok", "ready": True, "busy": busy, "queued": 0,
            "quiescing": quiescing, "idle_for_seconds": time.monotonic() - last_used}


@app.post("/v1/runtime/{operation}")
async def lifecycle(operation: str):
    global quiescing
    if operation not in ("quiesce", "resume"):
        raise HTTPException(404)
    if operation == "quiesce" and busy:
        raise HTTPException(409, "Music generation is active")
    quiescing = operation == "quiesce"
    return {"status": "ok"}


@app.post("/v1/music/generate")
async def generate(payload: Generation, request: Request):
    global busy, last_used
    if busy or quiescing:
        raise HTTPException(409, "Music worker is busy or quiescing")
    busy = True
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="music-") as directory:
            folder = Path(directory)
            (folder / "request.json").write_text(payload.model_dump_json())
            with (folder / "generate.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    "python", "/app/generate.py", str(folder / "request.json"), str(folder / "music.wav"),
                    stdout=log, stderr=log, start_new_session=True)
                started = time.monotonic()
                while process.returncode is None:
                    if await request.is_disconnected():
                        raise HTTPException(499, "Client disconnected")
                    if time.monotonic() - started > 1800:
                        raise HTTPException(504, "Music generation exceeded 30 minutes")
                    try:
                        await asyncio.wait_for(process.wait(), timeout=.5)
                    except TimeoutError:
                        pass
                if process.returncode != 0:
                    error = (folder / "generate.log").read_text(errors="replace")[-6000:]
                    raise HTTPException(502, "Music generation failed: " + error)
            metadata = json.loads((folder / "music.json").read_text())
            return {"audio": base64.b64encode((folder / "music.wav").read_bytes()).decode(),
                    "mime_type": "audio/wav", "metadata": metadata}
    finally:
        if process is not None and process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        busy = False
        last_used = time.monotonic()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MUSIC_PORT", "8727")))
