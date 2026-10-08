"""Talk image API backed only by the qualified native Qwen Image 2.1 graph."""
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import io
import os
from pathlib import Path
import secrets
import time
import uuid
import json

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
import httpx
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

MODEL = "qwen-image-2.1-uc-nvfp4"
KEEP_MODELS_LOADED = os.getenv("SPARKTALK_KEEP_MODELS_LOADED", "0") == "1"
DIT_RESIDENT = os.getenv("SPARKTALK_IMAGE_RESIDENCY", "legacy") == "dit"
JOB = Path(os.getenv("JOB_DIR", "/job"))
COMFY = "http://127.0.0.1:" + os.getenv("COMFY_PORT", "8188")
INPUT = Path("/tmp/qwen-image21/input")
MAX_PIXELS = 16_777_216
MAX_REFERENCES = 10
OPERATIONS = ["generate", "reference_generate", "identity_edit", "head_swap", "inpaint", "outpaint", "object_remove", "background_cleanup", "background_remove"]


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL
    prompt: str = Field(default="", max_length=16000)
    operation: str = "generate"
    size: str = "1024x1024"
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    n: int = Field(default=1, ge=1, le=1)
    response_format: str = "b64_json"
    output_format: str = "png"
    source_image: str | None = None
    head_image: str | None = None
    reference_images: list[str] = Field(default_factory=list, max_length=MAX_REFERENCES)
    anypaint_image: str | None = None
    anypaint_mask: str | None = None
    mask_box: list[int] | None = None
    reference_crop_box: list[int] | None = None
    preserve_source: bool = False
    outpaint_left: int = Field(default=0, ge=0, le=1536, multiple_of=16)
    outpaint_top: int = Field(default=0, ge=0, le=1536, multiple_of=16)
    outpaint_right: int = Field(default=0, ge=0, le=1536, multiple_of=16)
    outpaint_bottom: int = Field(default=0, ge=0, le=1536, multiple_of=16)


class Runtime:
    def __init__(self):
        self.gate = asyncio.Lock()
        self.state_lock = asyncio.Lock()
        self.jobs = 0
        self.active = False
        self.quiescing = False
        self.last_completed = time.monotonic()

    @asynccontextmanager
    async def job(self):
        async with self.state_lock:
            if self.quiescing:
                raise HTTPException(503, "Qwen Image 2.1 is returning idle memory; retry after runtime startup")
            self.jobs += 1
        acquired = False
        try:
            await self.gate.acquire()
            acquired = True
            self.active = True
            yield
        finally:
            if acquired:
                self.active = False
                self.gate.release()
            async with self.state_lock:
                self.jobs -= 1
                self.last_completed = time.monotonic()


runtime = Runtime()
app = FastAPI(title="SparkTalk Qwen Image 2.1")


def decode_bytes(data):
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("source exceeds 16 megapixels")
            result = ImageOps.exif_transpose(image)
            return result.convert("RGBA" if "A" in result.getbands() else "RGB")
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "invalid image or source exceeds 16 megapixels") from error


def decode(value):
    try:
        prefix, encoded = value.split(",", 1)
        if not prefix.startswith("data:image/") or not prefix.endswith(";base64"):
            raise ValueError("image data URL required")
        return decode_bytes(base64.b64decode(encoded, validate=True))
    except (ValueError, AttributeError, binascii.Error) as error:
        raise HTTPException(400, "invalid image data URL") from error


def box_mask(source, box):
    if len(box) != 4 or not (0 <= box[0] < box[2] <= source.width and 0 <= box[1] < box[3] <= source.height):
        raise HTTPException(400, "mask_box must be [left, top, right, bottom] within source pixels")
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
    return mask


def working_image(image):
    scale = min(1.0, 1024 / max(image.size))
    size = tuple(max(32, round(value * scale / 32) * 32) for value in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def output_size(value):
    try:
        width, height = map(int, value.split("x"))
    except (ValueError, AttributeError) as error:
        raise HTTPException(400, "size must be WIDTHxHEIGHT") from error
    if any(dimension < 256 or dimension > 1024 or dimension % 16 for dimension in (width, height)):
        raise HTTPException(400, "generation dimensions must be 256..1024 multiples of 16")
    return width, height


def prepare(request):
    if request.model != MODEL or request.operation not in OPERATIONS:
        raise HTTPException(400, "unsupported Qwen Image 2.1 model or operation")
    if request.response_format != "b64_json" or request.output_format != "png":
        raise HTTPException(400, "Qwen Image 2.1 returns b64_json PNG")
    operation = request.operation
    prompt = request.prompt.strip()
    references = [decode(value) for value in request.reference_images]
    source = None
    mask = None
    preserve_box = None
    if operation in ("inpaint", "object_remove", "outpaint"):
        source = decode(request.anypaint_image or request.source_image)
        if operation == "outpaint":
            pads = (request.outpaint_left, request.outpaint_top, request.outpaint_right, request.outpaint_bottom)
            if not any(pads) or request.anypaint_mask or request.mask_box is not None:
                raise HTTPException(400, "outpaint requires nonzero padding and no mask")
            left, top, right, bottom = pads
            size = (source.width + left + right, source.height + top + bottom)
            if size[0] * size[1] > MAX_PIXELS:
                raise HTTPException(400, "expanded canvas exceeds 16 megapixels")
            canvas = Image.new(source.mode, size, (0, 255, 0, 255) if source.mode == "RGBA" else (0, 255, 0))
            canvas.paste(source, (left, top))
            preserve_box = (left, top, left + source.width, top + source.height)
            source = canvas
            references = [working_image(canvas)]
            prompt = "Extend the photograph into the green border. Continue the scene, lighting and perspective naturally. Preserve the original central photograph. Remove every green placeholder. " + prompt
        else:
            if bool(request.anypaint_mask) == (request.mask_box is not None):
                raise HTTPException(400, "inpaint/object_remove require exactly one mask or mask_box")
            mask = decode(request.anypaint_mask).convert("L") if request.anypaint_mask else box_mask(source, request.mask_box)
            if mask.size != source.size or mask.getextrema() == (0, 0):
                raise HTTPException(400, "mask must match the source and contain editable pixels")
            scaled = working_image(source)
            references = [scaled, mask.resize(scaled.size, Image.Resampling.NEAREST).convert("RGB")]
            instruction = "Remove the marked object completely and reconstruct the original scene." if operation == "object_remove" else "Apply the requested edit inside the white mask."
            prompt = "Image 1 is the source. Image 2 is an edit mask: white marks the editable region and black must remain unchanged. " + instruction + " Preserve all other content. " + prompt
    elif operation == "head_swap":
        source = decode(request.source_image)
        head = decode(request.head_image)
        if request.reference_crop_box is not None:
            box_mask(head, request.reference_crop_box)
            head = head.crop(tuple(request.reference_crop_box))
        if request.mask_box is not None:
            mask = box_mask(source, request.mask_box)
        references = [working_image(source), working_image(head)]
        prompt = "Use Image 1 as the base. Replace only its head using the face, hair and visible attributes of the person in Image 2. Match Image 1 head pose, expression, rendering style and lighting. Preserve its body, clothes and background. Do not add glasses or accessories absent in the reference. " + prompt
    elif operation in ("background_cleanup", "background_remove"):
        source = decode(request.source_image)
        references = [working_image(source)]
        if operation == "background_remove":
            prompt = "Extract the foreground subject from Image 1, including fine hair and openings, at the same size and position. This is an RGBA image with transparency. The background is transparent. Preserve the original subject, geometry and colors without repainting. " + prompt
        else:
            prompt = "Keep the subject in Image 1 at its original size and position. Replace only the background with a clean pure white studio background. Preserve all subject details. " + prompt
    elif operation == "identity_edit":
        if request.source_image:
            source = decode(request.source_image)
            references.insert(0, working_image(source))
        if not references:
            raise HTTPException(400, "identity_edit requires a source or references")
    elif operation == "reference_generate":
        if not references:
            raise HTTPException(400, "reference_generate requires references")
    elif operation == "generate" and request.source_image:
        raise HTTPException(400, "use identity_edit with source_image")
    if not prompt:
        raise HTTPException(400, "a prompt is required")
    if len(references) > MAX_REFERENCES:
        raise HTTPException(400, "Qwen Image 2.1 supports at most ten references")
    requested_size = output_size(request.size)
    return prompt, references, source, mask, preserve_box, requested_size


def graph(prompt, references, seed, size, prefix, operation):
    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen-image21-uc-nvfp4.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "vae": ["3", 0], "resolution": 1024}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "6": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "int8"}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "seed": seed, "steps": 40, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["4", 2] if references and operation != "reference_generate" else ["5", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }
    paths = []
    INPUT.mkdir(parents=True, exist_ok=True)
    for number, image in enumerate(references, 1):
        name = prefix + "-reference-" + str(number) + ".png"
        path = INPUT / name
        image.save(path)
        paths.append(path)
        node = str(20 + number)
        nodes[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        nodes["4"]["inputs"]["images.image_" + str(number)] = [node, 0]
    return nodes, paths


async def wait_idle(client):
    for attempt in range(120):
        response = await client.get(COMFY + "/queue")
        response.raise_for_status()
        state = response.json()
        if not state["queue_running"] and not state["queue_pending"]:
            return
        await asyncio.sleep(.25)
    raise RuntimeError("Qwen Image 2.1 inference has not stopped")


async def execute(nodes, request):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(COMFY + "/prompt", json={"prompt": nodes})
        response.raise_for_status()
        value = response.json()
        if value.get("node_errors"):
            raise HTTPException(500, "Qwen Image 2.1 graph rejected: " + str(value["node_errors"])[:2000])
        prompt_id = value["prompt_id"]
        completed = False
        try:
            deadline = time.monotonic() + 30 * 60
            while time.monotonic() < deadline:
                if await request.is_disconnected():
                    raise HTTPException(499, "image request canceled")
                response = await client.get(COMFY + "/history/" + prompt_id)
                response.raise_for_status()
                history = response.json().get(prompt_id)
                if history:
                    if history.get("status", {}).get("status_str") == "error":
                        raise HTTPException(500, "Qwen Image 2.1 inference failed: " + str(history["status"])[-2000:])
                    images = history.get("outputs", {}).get("9", {}).get("images", [])
                    if images:
                        result = await client.get(COMFY + "/view", params=images[0])
                        result.raise_for_status()
                        completed = True
                        data = result.content
                        output = Path("/tmp/qwen-image21/output") / images[0].get("subfolder", "") / images[0]["filename"]
                        output.unlink(missing_ok=True)
                        await client.post(COMFY + "/history", json={"delete": [prompt_id]})
                        return data
                await asyncio.sleep(.25)
            raise HTTPException(504, "Qwen Image 2.1 image timeout")
        finally:
            if not completed:
                await client.post(COMFY + "/interrupt")
                await wait_idle(client)


def resident_state():
    try:
        os.kill(int((JOB / "worker.pid").read_text()), 0)
        if not (JOB / "ready.json").exists() or (JOB / "stop").exists():
            raise ValueError("worker is loading")
        state = json.loads((JOB / "worker-state.json").read_text())
        loaded = state.get("resident", {}).get("qwim", 0)
        if loaded <= 0:
            raise ValueError("DiT is not resident")
        return state
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Qwen Image DiT worker is not ready") from error


async def execute_resident(prompt, paths, seed, size, operation, request):
    resident_state()
    case = uuid.uuid4().hex
    payload = {"case": case, "kind": "qwim", "prompt": prompt, "seed": seed,
               "width": size[0], "height": size[1], "operation": operation,
               "reference_files": [p.name for p in paths], "progress_id": case}
    temporary = JOB / "requests" / (case + ".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(temporary.with_suffix(".json"))
    result_path = JOB / "results" / (case + ".json")
    finished = False
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if await request.is_disconnected():
                raise HTTPException(499, "Qwen Image generation canceled")
            resident_state()
            if result_path.exists():
                value = json.loads(result_path.read_text())
                finished = True
                if value.get("status") != "success":
                    raise HTTPException(502, "Qwen Image generation failed; inspect worker logs")
                return (JOB / "output" / (case + ".png")).read_bytes()
            await asyncio.sleep(.2)
        raise HTTPException(504, "Qwen Image generation timeout")
    finally:
        if not finished:
            (JOB / "cancel" / case).touch()
            # Keep the API lease and reference files until the worker releases
            # its auxiliary models. Canceling a job must retain the DiT.
            for _ in range(240):
                if result_path.exists():
                    break
                try:
                    resident_state()
                except HTTPException:
                    break
                await asyncio.sleep(.25)
            else:
                # A wedged CUDA worker cannot safely keep using references
                # after this lease ends. Treat unresponsive cancellation as
                # a fatal worker failure; the container supervisor exits too.
                (JOB / "stop").touch()
                os.kill(int((JOB / "worker.pid").read_text()), 9)
        for folder, suffix in (("requests", ".json"), ("requests", ".running"),
                               ("results", ".json"), ("output", ".png"), ("cancel", "")):
            (JOB / folder / (case + suffix)).unlink(missing_ok=True)


def restore(raw, request, source, mask, preserve_box, size):
    with Image.open(io.BytesIO(raw)) as result:
        result.load()
        output = result.copy()
    if source is not None and request.operation in ("inpaint", "object_remove", "head_swap", "background_cleanup", "background_remove", "outpaint"):
        output = output.resize(source.size, Image.Resampling.LANCZOS)
        if mask is not None:
            output = Image.composite(output.convert(source.mode), source, mask)
        elif request.operation == "background_remove":
            if "A" not in output.getbands() or output.getchannel("A").getextrema()[0] >= 250:
                raise HTTPException(422, "Qwen Image 2.1 did not produce genuine transparency")
            alpha = output.getchannel("A")
            output = source.convert("RGBA")
            output.putalpha(alpha)
        elif request.operation == "outpaint" and request.preserve_source:
            mask = Image.new("L", source.size, 255)
            left, top, right, bottom = preserve_box
            ImageDraw.Draw(mask).rectangle((left, top, right - 1, bottom - 1), fill=0)
            output = Image.composite(output.convert(source.mode), source, mask)
    elif not request.reference_images and request.operation == "generate":
        output = output.resize(size, Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    output.save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode()


async def generate(payload, request):
    async with runtime.job():
        prompt, references, source, mask, preserve_box, size = prepare(payload)
        references = [working_image(image) for image in references]
        seed = payload.seed if payload.seed is not None else secrets.randbits(63)
        nodes, paths = graph(prompt, references, seed, size, uuid.uuid4().hex, payload.operation)
        try:
            raw = (await execute_resident(prompt, paths, seed, size, payload.operation, request)
                   if DIT_RESIDENT else await execute(nodes, request))
            encoded = restore(raw, payload, source, mask, preserve_box, size)
            return {"created": int(time.time()), "model": MODEL, "seed": seed, "data": [{"b64_json": encoded}]}
        except httpx.HTTPError as error:
            raise HTTPException(503, "Qwen Image 2.1 runtime unavailable: " + str(error)) from error
        finally:
            for path in paths:
                path.unlink(missing_ok=True)


@app.post("/v1/images/generations")
async def images(payload: ImageRequest, request: Request):
    return await generate(payload, request)


@app.post("/v1/images/edits")
async def edits(request: Request, image: list[UploadFile] = File(...), prompt: str = Form(...), model: str = Form(MODEL), size: str = Form("1024x1024"), seed: int | None = Form(None), n: int = Form(1), response_format: str = Form("b64_json"), operation: str = Form("identity_edit")):
    if operation not in ("reference_generate", "identity_edit") or len(image) > MAX_REFERENCES:
        raise HTTPException(400, "invalid reference operation or too many images")
    values = []
    for upload in image:
        data = await upload.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise HTTPException(413, "reference upload too large")
        values.append("data:image/png;base64," + base64.b64encode(data).decode())
    return await generate(ImageRequest(model=model, operation=operation, prompt=prompt, reference_images=values, size=size, seed=seed, n=n, response_format=response_format), request)


async def backend_state():
    if DIT_RESIDENT:
        state = resident_state()
        return {"status": "ok", "model": MODEL, "busy": runtime.jobs > 0,
                "active": int(runtime.active), "queued": max(0, runtime.jobs-int(runtime.active)),
                "idle_for_seconds": max(0, time.monotonic()-runtime.last_completed),
                "quiescing": runtime.quiescing, "core_ready": True,
                "dit_loaded_gib": state["resident"]["qwim"],
                "loaded_models": state.get("loaded_models", []), "residency": "dit"}
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            ready, queue = await asyncio.gather(client.get(COMFY + "/system_stats"), client.get(COMFY + "/queue"))
        ready.raise_for_status()
        queue.raise_for_status()
        values = queue.json()
        active = len(values["queue_running"])
        pending = len(values["queue_pending"])
        return {"status": "ok", "model": MODEL, "busy": runtime.jobs > 0 or active > 0 or pending > 0,
                "active": max(int(runtime.active), active), "queued": max(max(0, runtime.jobs - int(runtime.active)), pending),
                "idle_for_seconds": max(0.0, time.monotonic() - runtime.last_completed),
                "quiescing": runtime.quiescing}
    except httpx.HTTPError as error:
        raise HTTPException(503, "Qwen Image 2.1 ComfyUI is not ready") from error


@app.get("/health")
async def health():
    return await backend_state()


@app.get("/v1/models")
async def models():
    await backend_state()
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}


@app.get("/v1/loras")
async def loras():
    return {"data": [], "model": MODEL, "operations": OPERATIONS, "styles": []}


@app.get("/v1/runtime/memory")
async def memory():
    state = await backend_state()
    return {**state, "memory_schema": 2, "workspace_kind": "whole-service", "memory_gib": 14.0 if DIT_RESIDENT else (24.0 if KEEP_MODELS_LOADED else 14.0),
            "workspace_gib": 6.0 if DIT_RESIDENT else 10.0, "reference_cache_dtype": "int8", "process_reclaim": True,
            "keep_models_loaded": DIT_RESIDENT or KEEP_MODELS_LOADED}


@app.post("/v1/runtime/quiesce")
async def quiesce():
    async with runtime.state_lock:
        state = await backend_state()
        if state["busy"]:
            raise HTTPException(409, "Qwen Image 2.1 has active or queued image work")
        runtime.quiescing = True
        return {**state, "quiescing": True}


@app.post("/v1/runtime/resume")
async def resume():
    async with runtime.state_lock:
        runtime.quiescing = False
    return {"status": "ok"}
