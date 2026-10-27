import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
import json
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import store
import assets
import authoring
import clip_store
import voice_catalog
import voice_tools
import voice_jobs
import voice_library
import voice_api
from spec import Plan, Artwork

stopping = threading.Event()
TALK = os.environ.get("SHORTS_TALK_ENDPOINT", "http://127.0.0.1:8585").rstrip("/")


def deliver_completed() -> None:
    for job in voice_jobs.pending_deliveries():
        try:
            response = httpx.post(TALK + '/api/voices/publish', json={'session_id':job['session_id'],'job_id':job['id']}, timeout=60)
            if response.status_code == 404:
                voice_jobs.cancel_session(job['session_id'])
                voice_jobs.acknowledge(job['id'], job['version'])
                continue
            response.raise_for_status()
            voice_jobs.acknowledge(job['id'], job['version'])
        except Exception as error:
            print(f"Voice delivery pending: {job['id']}: {error}", flush=True)
    with store.connect() as connection:
        rows = connection.execute("SELECT id,session_id FROM jobs WHERE status='completed' ORDER BY updated DESC LIMIT 50").fetchall()
    for job in rows:
        folder = store.ROOT / job["id"]
        if (folder / "published.json").exists():
            continue
        try:
            response = httpx.post(TALK + "/api/shorts/publish", json={"session_id": job["session_id"], "job_id": job["id"]}, timeout=30)
            response.raise_for_status()
            (folder / "published.json").write_text(json.dumps(response.json(), ensure_ascii=False))
        except Exception as error:
            print(f"Talk delivery pending: {job['id']}: {error}", flush=True)
    with store.connect() as connection:
        rows = connection.execute("SELECT id,session_id FROM clip_jobs WHERE status='completed' ORDER BY updated DESC LIMIT 50").fetchall()
    for job in rows:
        folder = clip_store.folder(job['id'])
        if (folder/'published.json').exists():
            continue
        try:
            response=httpx.post(TALK+'/api/clip-production/publish',json={'session_id':job['session_id'],'job_id':job['id']},timeout=30)
            response.raise_for_status()
            clip_store.atomic_json(folder/'published.json',response.json())
        except Exception as error:
            print(f"Clip delivery pending: {job['id']}: {error}",flush=True)


def process_jobs() -> None:
    next_delivery = 0
    while not stopping.wait(0.5):
        if time.monotonic() >= next_delivery:
            deliver_completed()
            voice_library.repair_manifests()
            voice_library.cleanup()
            next_delivery = time.monotonic() + 30
        # Fairness across durable queues, while retaining one model-owning job.
        with store.connect() as connection:
            pending = []
            for table, backend_type in [('jobs', store), ('clip_jobs', clip_store), ('voice_jobs', voice_jobs)]:
                row = connection.execute(f"SELECT created FROM {table} WHERE status='queued' AND canceled=0 ORDER BY created LIMIT 1").fetchone()
                if row: pending.append((row[0], backend_type))
        if not pending: continue
        backend = min(pending, key=lambda item:item[0])[1]
        job_id = backend.claim()
        if job_id is None:
            continue
        folder = voice_jobs.folder(job_id) if backend is voice_jobs else clip_store.folder(job_id) if backend is clip_store else store.ROOT / job_id
        folder.mkdir(exist_ok=True)
        with (folder / "worker.log").open("ab") as log:
            runner = 'voice_runner.py' if backend is voice_jobs else 'clip_render.py' if backend is clip_store else 'render.py'
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name(runner)), job_id], stdout=log, stderr=log, start_new_session=True)
            while process.poll() is None:
                if stopping.wait(0.5) or backend.get(job_id)["canceled"]:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    if backend.get(job_id)["canceled"]:
                        backend.update(job_id, "canceled", "중지됨", backend.get(job_id)["progress"])
                    else:
                        backend.update(job_id, "queued", "다시 시작하면 이어서 진행", backend.get(job_id)["progress"])
                    break
            current = backend.get(job_id)
            if current["status"] == "running":
                backend.update(job_id, "failed", "실패", current["progress"], (folder / "error.txt").read_text() if (folder / "error.txt").exists() else "제작 작업이 중단됐습니다. 다시 시도할 수 있습니다.")
            elif current["status"] == "completed" or (backend is voice_jobs and current["status"] in voice_jobs.TERMINAL):
                deliver_completed()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    store.initialize()
    clip_store.initialize()
    voice_jobs.initialize()
    voice_library.repair_manifests()
    stopping.clear()
    thread = threading.Thread(target=process_jobs, daemon=True)
    thread.start()
    yield
    stopping.set()
    thread.join(timeout=20)


app = FastAPI(lifespan=lifespan)
app.include_router(voice_api.router)


@app.middleware('http')
async def bounded_voice_upload(request, call_next):
    if request.url.path == '/v1/voice-jobs/upload':
        try:
            length = int(request.headers.get('content-length', '0'))
        except ValueError:
            length = 0
        if length < 1 or length > voice_api.MAX_SOURCE_BYTES + 128 * 1024:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'음성 업로드 크기를 확인할 수 없거나 제한을 초과했습니다'}, status_code=413)
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.6.0", "renderers": ["hyperframes", "manim", "lottie", "svg"], "durable_jobs": True, "model_authored_templates": True, "reviewed_image_to_video": True, "native_tag_delivery": True}


@app.get("/v1/runtime/memory")
def memory():
    with store.connect() as connection:
        count = connection.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
        count += connection.execute("SELECT count(*) FROM clip_jobs WHERE status IN ('queued','running')").fetchone()[0]
        count += connection.execute("SELECT count(*) FROM voice_jobs WHERE status IN ('queued','running')").fetchone()[0]
    return {"status": "ok", "busy": count > 0, "queued": count, "memory_gib": 6}


@app.post('/v1/clips')
def create_clip(request:clip_store.Request):
    try:
        return clip_store.create(request)
    except (ValueError,OSError) as error:
        raise HTTPException(422,str(error)) from error


@app.get('/v1/clips')
def clips(session_id:str):
    return clip_store.list_jobs(session_id)


@app.get('/v1/clips/{job_id}')
def clip(job_id:str):
    try:
        return clip_store.get(job_id)
    except (ValueError,KeyError) as error:
        raise HTTPException(404,'Production not found') from error


class ClipAction(BaseModel):
    cancel_turn_key: str = Field(default='',max_length=64,pattern=r'^(?:[a-f0-9]{64})?$')


@app.post('/v1/clips/{job_id}/{operation}')
def clip_action(job_id:str,operation:str,request:ClipAction | None=Body(default=None)):
    try:
        return clip_store.action(job_id,operation,request.cancel_turn_key if request else '')
    except (ValueError,KeyError) as error:
        raise HTTPException(409,str(error)) from error


@app.get('/v1/clips/{job_id}/files/{name}')
def clip_file(job_id:str,name:str):
    job=clip(job_id)
    if name not in ('final.mp4','final-image.png','preview-sheet.jpg','plan.json','report.json'):
        raise HTTPException(404)
    if name in ('final.mp4','final-image.png','preview-sheet.jpg') and job['status']!='completed':
        raise HTTPException(409,'Production has not passed review')
    target=clip_store.folder(job_id)/name
    if not target.is_file():raise HTTPException(404)
    return FileResponse(target,filename=name)


@app.post("/v1/projects")
def create(project: store.Project):
    if not project.scenes and not project.brief.strip():
        raise HTTPException(422, "주제 또는 대본을 입력하세요")
    try:
        if project.plan is not None:
            project.plan = Plan.model_validate(project.plan).model_dump()
        return store.create(project)
    except (ValueError, KeyError) as error:
        raise HTTPException(422, str(error)) from error


@app.get("/v1/projects")
def projects(session_id: str):
    return store.list_jobs(session_id)


@app.get("/v1/projects/{job_id}")
def project(job_id: str):
    try:
        return store.get(job_id)
    except (ValueError, KeyError) as error:
        raise HTTPException(404, "Project not found") from error


@app.post("/v1/projects/{job_id}/{operation}")
def action(job_id: str, operation: str):
    if operation not in ("cancel", "retry", "render"):
        raise HTTPException(404)
    try:
        return store.action(job_id, operation)
    except (ValueError, KeyError) as error:
        raise HTTPException(409, str(error)) from error


@app.get("/v1/projects/{job_id}/files/{name}")
def file(job_id: str, name: str):
    project(job_id)
    if name not in ("final.mp4", "storyboard.json", "timing.json", "quality.json", "sources.json", "poster.png", "preview.png", "preview-sheet.jpg", "audit.json", "artwork.json", "composition-spec.json"):
        raise HTTPException(404)
    target = store.ROOT / job_id / name
    if not target.is_file():
        raise HTTPException(404)
    return FileResponse(target, filename=name)


class AssetRequest(BaseModel):
    session_id: str
    action: str
    query: str = ""
    source_url: str = ""
    asset_id: str = ""
    kind: str = "lottie"
    name: str = "Animation"
    content: dict | str | None = None


@app.post("/v1/assets")
def asset_tool(request: AssetRequest):
    try:
        assets.session_root(request.session_id)
        if request.action == "search":
            return assets.search(request.query)
        if request.action == "import":
            return assets.import_lottie(request.session_id, request.source_url)
        if request.action == "create":
            return assets.create_asset(request.session_id, request.name, request.kind, request.content)
        if request.action == "list":
            return {"assets": assets.list_assets(request.session_id)}
        if request.action == "inspect":
            record, root = assets.get_asset(request.session_id, request.asset_id)
            return {"asset": record, "content": (root/record["filename"]).read_text()}
        raise ValueError("Unknown asset action")
    except (ValueError, KeyError, httpx.HTTPError) as error:
        raise HTTPException(422, str(error)) from error


class TemplateRequest(BaseModel):
    session_id: str
    action: str
    name: str = "Local model template"
    template_id: str = ""
    job_id: str = ""
    artwork: dict | None = None


@app.post("/v1/templates")
def template_tool(request: TemplateRequest):
    try:
        if request.action == "list":
            return {"templates": authoring.list_templates(request.session_id)}
        if request.action == "inspect":
            return authoring.get_template(request.session_id, request.template_id)
        if request.action in ("create", "save"):
            artwork = request.artwork
            if request.job_id:
                job = store.get(request.job_id)
                if job["session_id"] != request.session_id:
                    raise ValueError("다른 대화의 템플릿은 사용할 수 없습니다")
                artwork = json.loads((store.ROOT/request.job_id/"artwork.json").read_text())
            if artwork is None:
                raise ValueError("실제 CSS·GSAP·HTML 코드 또는 완성된 작업 ID가 필요합니다")
            return authoring.save_template(request.session_id, request.name, artwork, request.job_id)
        raise ValueError("Unknown template action")
    except (ValueError, KeyError, OSError) as error:
        raise HTTPException(422, str(error)) from error


class ProjectRequest(BaseModel):
    session_id: str
    action: str
    job_id: str = ""
    brief: str = ""
    title: str = "Shorts"
    template_id: str = ""
    design_mode: Literal["redesign", "preserve"] = "redesign"
    asset_ids: list[str] = Field(default_factory=list)
    duration_seconds: int | None = Field(default=None, ge=15, le=120)
    preview_only: bool = True
    plan: dict | None = None


@app.post("/v1/project-tools")
def project_tool(request: ProjectRequest):
    try:
        if request.action == "capabilities":
            return {"authoring": "Local model writes CSS, HTML/SVG overlays and GSAP code; no fixed visual theme", "contract": authoring.CONTRACT,
                    "assets": ["AI images/video via Talk", "conversation images", "LottieFiles verified free imports", "original Lottie JSON", "original SVG", "reuse prior project scenes"],
                    "templates": ["create", "inspect", "save from project", "list", "apply by template_id"],
                    "workflow": ["plan", "asset selection", "speech and duration fit", "model-authored template", "browser audit", "local VLM review", "repair", "preview", "render"],
                    "voice": {"presets": [v["id"] for v in voice_catalog.catalog()], "catalog": voice_catalog.public_catalog(), "tool": "shorts_voices", "speed": [.8, 1.2], "sentence_pause": [.08, .8]},
                    "music": ["none", "generated: ACE-Step Turbo / XL Base", "explicit legacy synth_keys / synth_pad"],
                    "limits": {"seconds": 120, "scenes": 16, "commercial_assets": "Verify source-page license; preserve source and attribution"},
                    "plan_schema": Plan.model_json_schema(), "artwork_schema": Artwork.model_json_schema()}
        if request.action == "draft":
            plan = Plan.model_validate(request.plan).model_dump() if request.plan else None
            return store.create(store.Project(session_id=request.session_id, title=request.title, brief=request.brief,
                                             duration_seconds=request.duration_seconds or 60, preview_only=True, template_id=request.template_id, design_mode=request.design_mode, asset_ids=request.asset_ids, plan=plan))
        if request.action == "list":
            return {"jobs": [{key:value for key,value in job.items() if key!="project"} for job in store.list_jobs(request.session_id)[:20]]}
        job = store.get(request.job_id)
        if job["session_id"] != request.session_id:
            raise ValueError("이 대화의 작업이 아닙니다")
        folder = store.ROOT/request.job_id
        if request.action == "inspect":
            result = {"job": job}
            for name in ("storyboard", "artwork", "quality", "sources", "audit"):
                path = folder/(name+".json")
                if path.exists():
                    data = json.loads(path.read_text())
                    if name == "audit":
                        data = {key:value for key,value in data.items() if key != "samples"}
                    result[name] = data
            return result
        if request.action == "render":
            return store.action(request.job_id, "render")
        if request.action in ("revise", "preview"):
            data = dict(job["project"])
            data.update(source_job_id=request.job_id, voice_only=False, scenes=[], plan=request.plan, preview_only=request.preview_only or request.action == "preview", design_mode=request.design_mode)
            if request.design_mode == "redesign":
                data["template_id"] = request.template_id
            if request.brief:
                data["brief"] = (data.get("brief", "") + "\n\nLATEST REVISION REQUEST (takes precedence):\n" + request.brief)[-12000:]
            if request.duration_seconds is not None:
                data["duration_seconds"] = request.duration_seconds
            if request.template_id:
                data["template_id"] = request.template_id
            if request.asset_ids:
                data["asset_ids"] = request.asset_ids
            return store.create(store.Project.model_validate(data))
        raise ValueError("Unknown project action")
    except (ValueError, KeyError, OSError) as error:
        raise HTTPException(422, str(error)) from error


class VoiceRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    action: Literal['list', 'preview', 'select', 'apply']
    voice_id: str = ''
    job_id: str = ''


@app.post('/v1/voices')
def voices_tool(request: VoiceRequest):
    try:
        return voice_tools.execute(request.session_id, request.action, request.voice_id, request.job_id)
    except (ValueError, KeyError, OSError) as error:
        raise HTTPException(422, str(error)) from error


@app.get('/v1/voices/{voice_id}/preview.wav')
def voice_preview(voice_id: str, session_id: str = ''):
    try:
        return FileResponse(voice_catalog.preview(voice_id, session_id), media_type='audio/wav', filename=voice_id+'-Fish-preview.wav')
    except (ValueError, OSError) as error:
        raise HTTPException(404, str(error)) from error
