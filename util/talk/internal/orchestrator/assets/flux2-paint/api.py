"""FLUX generation/editing, LanPaint inpainting and fal outpainting LoRA."""
import asyncio
import base64
import binascii
import io
import os
import time
import uuid
from typing import Literal

import httpx
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageOps, ImageChops, ImageFilter, UnidentifiedImageError
from pydantic import ConfigDict, Field

import base_api as base
_cutout_worker = None

# All generation, reference and LoRA/LanPaint graphs originate here.
if os.getenv("SPARKTALK_FLUX_PHASED") == "1":
    original_workflow = base.workflow
    def phased_workflow(*args, **kwargs):
        graph = original_workflow(*args, **kwargs)
        graph['1']['class_type'] = 'SparkTalkUNETLoader'
        graph['3']['class_type'] = 'SparkTalkVAELoader'
        clip_name = graph.pop("2")["inputs"]["clip_name"]
        text = graph["4"]["inputs"]["text"]
        graph["4"] = {"class_type": "SparkTalkTextEncode", "inputs": {"text": text, "clip_name": clip_name}}
        reference_scales = [node for node in graph.values() if node['class_type'] == 'ImageScale']
        if reference_scales:
            for node in reference_scales:
                pixels = min(1048576, node['inputs']['width'] * node['inputs']['height']) // len(reference_scales)
                node['class_type'] = 'SparkTalkReferenceScale'
                node['inputs'] = {'image': node['inputs']['image'], 'max_pixels': max(256, pixels)}
        return graph
    base.workflow = phased_workflow

app = base.app
app.router.routes = [route for route in app.router.routes if getattr(route, "path", "") not in ("/v1/images/generations", "/health")]


class PaintRequest(base.ImageRequest):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["generate", "identity_edit", "head_swap", "inpaint", "outpaint", "object_remove", "background_cleanup", "background_remove"] = "generate"
    output_format: Literal["png"] = "png"
    preserve_source: bool = False
    background_method: Literal["rembg", "lora_rembg"] = "rembg"
    source_image: str | None = Field(default=None, max_length=32 * 1024 * 1024)
    head_image: str | None = Field(default=None, max_length=32 * 1024 * 1024)
    reference_crop_box: list[int] | None = None
    head_swap_strength: float = Field(default=1.0, ge=0.1, le=1.5)
    anypaint_image: str | None = Field(default=None, max_length=32 * 1024 * 1024)
    anypaint_mask: str | None = Field(default=None, max_length=32 * 1024 * 1024)
    mask_box: list[int] | None = None
    outpaint_left: int = Field(default=0, ge=0, le=1536, multiple_of=16)
    outpaint_top: int = Field(default=0, ge=0, le=1536, multiple_of=16)
    outpaint_right: int = Field(default=0, ge=0, le=1536, multiple_of=16)
    outpaint_bottom: int = Field(default=0, ge=0, le=1536, multiple_of=16)


def decode_image(value):
    try:
        header, data = value.split(",", 1)
        if not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("image data URL required")
        with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as image:
            if image.width * image.height > 16_777_216:
                raise ValueError("source exceeds 16 megapixels")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (ValueError, AttributeError, binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "invalid image data URL") from exc


def generation_size(width, height):
    if any(n < 256 or n > 1024 or n % 16 for n in (width, height)):
        raise HTTPException(400, "generation output must have each dimension 256..1024 and divisible by 16")


def prepare_paint(request):
    source = decode_image(request.anypaint_image)
    pads = (request.outpaint_left, request.outpaint_top, request.outpaint_right, request.outpaint_bottom)
    if request.operation in ("inpaint", "object_remove"):
        if any(pads):
            raise HTTPException(400, "padding is only supported for outpaint")
        if bool(request.anypaint_mask) == (request.mask_box is not None):
            raise HTTPException(400, "inpaint requires exactly one mask image or mask_box")
        if request.anypaint_mask:
            mask = decode_image(request.anypaint_mask).convert("L")
            if mask.size != source.size:
                raise HTTPException(400, "mask dimensions must match source; white edits and black preserves")
        else:
            box = request.mask_box
            if len(box) != 4 or not (0 <= box[0] < box[2] <= source.width and 0 <= box[1] < box[3] <= source.height):
                raise HTTPException(400, "mask_box must be [left, top, right, bottom] within source pixels")
            mask = Image.new("L", source.size, 0)
            ImageDraw.Draw(mask).rectangle((box[0], box[1], box[2]-1, box[3]-1), fill=255)
        if mask.getextrema() == (0, 0):
            raise HTTPException(400, "mask contains no editable pixels")
    else:
        if not any(pads) or request.anypaint_mask or request.mask_box is not None:
            raise HTTPException(400, "outpaint requires nonzero padding and no mask")
        left, top, right, bottom = pads
        size = (source.width + left + right, source.height + top + bottom)
        if size[0] * size[1] > 16_777_216:
            raise HTTPException(400, "expanded canvas exceeds 16 megapixels")
        canvas = Image.new("RGB", size, (0, 255, 0))
        canvas.paste(source, (left, top))
        mask = Image.new("L", size, 255)
        mask.paste(0, (left, top, left + source.width, top + source.height))
        source = canvas
    # ComfyUI LoadImage treats transparent pixels as the edit mask.
    rgba = source.convert("RGBA")
    rgba.putalpha(ImageOps.invert(mask))
    return rgba


def fit_masked_edit(original):
    """Bound GPU work; resize image and edit mask together, then pad if needed."""
    scale = min(1.0, 1024 / max(original.size))
    content_size = tuple(max(1, round(n * scale)) for n in original.size)
    rgb = original.convert("RGB").resize(content_size, Image.Resampling.LANCZOS)
    alpha = original.getchannel("A").resize(content_size, Image.Resampling.NEAREST)
    size = tuple(max(256, (n + 15) // 16 * 16) for n in content_size)
    canvas = Image.new("RGBA", size, (0, 0, 0, 255))
    rgb.putalpha(alpha)
    canvas.paste(rgb, (0, 0))
    return canvas, content_size


def restore_masked_edit(encoded, original, content_size):
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as result:
        edited = result.convert("RGB").crop((0, 0, *content_size))
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    mask = ImageOps.invert(original.getchannel("A"))
    result = Image.composite(edited, original.convert("RGB"), mask)
    return base64.b64encode(png_bytes(result)).decode()


def restore_output_size(encoded, output_size, content_size):
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as result:
        result = result.convert("RGB").crop((0, 0, *content_size))
        result = result.resize(output_size, Image.Resampling.LANCZOS)
    return base64.b64encode(png_bytes(result)).decode()


def paint_workflow(request, image_name, size, seed, prefix):
    width, height = size
    graph = base.workflow(request.prompt.strip(), width, height, seed, prefix)
    graph["20"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    graph["21"] = {"class_type": "LanPaint_ImageEncode", "inputs": {"image": ["20", 0], "mask": ["20", 1], "vae": ["3", 0]}}
    graph["22"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["4", 0], "latent": ["21", 0]}}
    graph["9"]["inputs"]["conditioning"] = ["22", 0]
    graph["10"]["class_type"] = "LanPaint_SamplerCustomAdvanced"
    graph["10"]["inputs"].update(latent_image=["21", 0], LanPaint_NumSteps=2,
        LanPaint_Lambda=5.0, LanPaint_StepSize=0.2, LanPaint_PromptMode="Image First", LanPaint_Info="SparkTalk")
    graph["11"] = {"class_type": "LanPaint_ImageDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0], "image": ["20", 0], "mask": ["20", 1], "blend_overlap": 9}}
    return graph


LORAS = {
    "outpaint": ("outpaint", "Fill the green spaces according to the image."),
    "object_remove": ("object-remove", "Remove the highlighted object from the scene."),
    "background_cleanup": ("background-remove", "Remove the background from the image. Use a clean white background."),
}


def lora_workflow(request, image_name, size, seed, prefix):
    operation = "background_cleanup" if request.operation == "background_remove" else request.operation
    kind, trigger = LORAS[operation]
    graph = base.workflow(trigger + " " + request.prompt.strip(), *size, seed, prefix, [image_name])
    graph["30"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["1", 0], "lora_name": f"fal-flux2-klein-4b-{kind}.safetensors", "strength_model": 1.1}}
    graph["9"]["inputs"]["model"] = ["30", 0]
    return graph


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


HEAD_SWAP_TRIGGER = ('head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. '
                     'Replace its head with the head from Picture 2, preserving the identity, hair, eyes, and nose of Picture 2. '
                     'Keep the head direction and expression from Picture 1. Keep the body, clothing, pose, framing, and background intact. ')


def checked_box(box, image, name):
    if not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box):
        raise HTTPException(400, f'{name} must be [left, top, right, bottom] in original image pixels')
    l, t, r, b = box
    if not (0 <= l < r <= image.width and 0 <= t < b <= image.height) or min(r-l, b-t) < 32:
        raise HTTPException(400, f'{name} must be within the image and at least 32 pixels per side')
    return tuple(box)


def head_swap_inputs(request):
    if not request.source_image or not request.head_image:
        raise HTTPException(400, 'head_swap requires target source_image first and head_image reference second')
    original = decode_image(request.source_image)
    head = decode_image(request.head_image)
    box = checked_box(request.mask_box, original, 'mask_box') if request.mask_box is not None else None
    target = original.crop(box) if box else original.copy()
    if request.reference_crop_box is not None:
        head = head.crop(checked_box(request.reference_crop_box, head, 'reference_crop_box'))
    scale = min(1024 / max(target.size), max(1.0, 512 / min(target.size)))
    size = tuple(max(256, min(1024, int(n * scale) // 16 * 16)) for n in target.size)
    target = target.resize(size, Image.Resampling.LANCZOS)
    return original, target, head, box


def restore_head_swap(encoded, original, box):
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as output:
        output = output.convert('RGB')
    if box is None:
        return base64.b64encode(png_bytes(output.resize(original.size, Image.Resampling.LANCZOS))).decode()
    l, t, r, b = box
    edited = output.resize((r-l, b-t), Image.Resampling.LANCZOS)
    # Feather strictly INSIDE the selected rectangle; every outside pixel
    # remains an exact copy of the original, including other people's heads.
    edge = min(12, max(2, min(edited.size) // 16))
    alpha = Image.new('L', edited.size, 0)
    ImageDraw.Draw(alpha).rectangle((edge, edge, edited.width-edge-1, edited.height-edge-1), fill=255)
    alpha = alpha.filter(ImageFilter.GaussianBlur(edge / 2))
    canvas = original.copy()
    canvas.paste(edited, (l, t), alpha)
    return base64.b64encode(png_bytes(canvas)).decode()


async def generate_head_swap(request, seed, prefix):
    original, target, head, box = head_swap_inputs(request)
    paths = []
    directory = base.INPUT_ROOT / 'nvfp4-api'
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for image in (target, head):
            path = directory / f'{uuid.uuid4().hex}.png'
            paths.append(path)
            image.save(path)
        graph = base.workflow(HEAD_SWAP_TRIGGER + request.prompt.strip(), *target.size, seed, prefix,
                              [f'nvfp4-api/{p.name}' for p in paths])
        graph['40'] = {'class_type': 'SparkTalkBFSLoader', 'inputs': {
            'model': ['1', 0], 'lora_name': 'bfs-head-v1-flux2-klein-4b.safetensors',
            'strength_model': request.head_swap_strength}}
        graph['9']['inputs']['model'] = ['40', 0]
        encoded = await base.execute_workflow(graph)
        return restore_head_swap(encoded, original, box)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


async def cutout(data):
    global _cutout_worker
    worker = await asyncio.create_subprocess_exec(
        "/opt/rembg-venv/bin/python", "/opt/nvfp4-api/rembg_worker.py",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _cutout_worker = worker
    try:
        output, error = await asyncio.wait_for(worker.communicate(data), timeout=120)
        if worker.returncode != 0:
            raise RuntimeError("rembg failed: " + error.decode(errors="replace")[-1200:])
        if not output.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("rembg did not return a PNG image")
        return base64.b64encode(output).decode("ascii")
    finally:
        if _cutout_worker is worker:
            _cutout_worker = None
        if worker.returncode is None:
            worker.kill()
            await worker.wait()


def preserve_original(encoded, canvas, request):
    mask = Image.new("L", canvas.size, 0)
    left, top = request.outpaint_left, request.outpaint_top
    mask.paste(255, (left, top, canvas.width - request.outpaint_right, canvas.height - request.outpaint_bottom))
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as generated:
        output = Image.composite(canvas.convert("RGB"), generated.convert("RGB"), mask)
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/health")
async def health():
    if not await base.comfy_ready():
        raise HTTPException(503, "NVFP4 runtime is starting")
    return {"status": "ok", "model": base.MODEL_ID,
            "outpaint_lora": "fal/flux-2-klein-4B-outpaint-lora", "outpaint_lora_strength": 1.1, "remove_loras": ["object-remove", "background-remove"], "background_cutout": "rembg u2net CPU",
            "head_swap_lora": "Alissonerdx/BFS-Best-Face-Swap", "head_swap_version": "Klein 4B V1 BF16"}


@app.post("/v1/images/generations")
async def generate(request: PaintRequest):
    if request.model not in base.MODEL_ALIASES or request.n != 1 or request.response_format != "b64_json":
        raise HTTPException(400, "expected the FLUX model, n=1, response_format=b64_json")
    if not request.prompt.strip():
        raise HTTPException(400, "prompt is required")
    if base.generation_lock.locked():
        raise HTTPException(409, "image generation is already running")
    # Fail instead of silently dropping arguments from a different operation.
    paint_fields = {"anypaint_image", "anypaint_mask", "mask_box", "outpaint_left", "outpaint_top", "outpaint_right", "outpaint_bottom"}
    allowed_paint = paint_fields if request.operation != 'head_swap' else paint_fields - {'mask_box'}
    if request.operation not in ("inpaint", "outpaint", "object_remove") and request.model_fields_set & allowed_paint:
        raise HTTPException(400, "mask and padding fields require inpaint or outpaint")
    if request.operation not in ("identity_edit", "head_swap", "background_cleanup", "background_remove") and request.source_image is not None:
        raise HTTPException(400, "source_image requires identity_edit")
    if request.preserve_source and request.operation != "outpaint":
        raise HTTPException(400, "preserve_source is only supported for outpaint")
    if request.background_method != "rembg" and request.operation != "background_remove":
        raise HTTPException(400, "background_method requires background_remove")
    head_fields = {'head_image', 'reference_crop_box', 'head_swap_strength'}
    if request.operation != 'head_swap' and request.model_fields_set & head_fields:
        raise HTTPException(400, 'head reference/crop/strength fields require head_swap')
    seed = base.request_seed(request.seed)
    prefix = f"nvfp4-api/{uuid.uuid4().hex}"
    path = None
    original_paint = None
    original_canvas = None
    output_size = None
    working_content_size = None
    async with base.generation_lock:
        try:
            if request.operation == 'head_swap':
                encoded = await generate_head_swap(request, seed, prefix)
                return {"created": int(time.time()), "seed": seed, "data": [{"b64_json": encoded}]}
            if request.operation in ("inpaint", "outpaint", "object_remove"):
                image = prepare_paint(request)
                if request.operation in ("inpaint", "object_remove"):
                    original_paint = image.copy()
                    image, working_content_size = fit_masked_edit(image)
                if request.operation == "outpaint":
                    original_canvas = image.convert("RGB")
                    output_size = original_canvas.size
                    image, working_content_size = fit_masked_edit(image)
                    image = image.convert("RGB")
                elif request.operation == "object_remove":
                    mask = ImageOps.invert(image.getchannel("A")).point(lambda n: 255 if n >= 128 else 0)
                    border = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(9)))
                    image = Image.composite(Image.new("RGB", image.size, (255, 0, 0)), image.convert("RGB"), border)
            elif request.operation in ("identity_edit", "background_cleanup", "background_remove"):
                image = decode_image(request.source_image)
            else:
                image = None
            if request.operation == "background_remove" and request.background_method == "rembg":
                encoded = await cutout(png_bytes(image))
                return {"created": int(time.time()), "seed": 0, "data": [{"b64_json": encoded}]}
            if request.operation in ("background_cleanup", "background_remove"):
                output_size = image.size
                image, working_content_size = fit_masked_edit(image.convert("RGBA"))
                image = image.convert("RGB")
            if image is not None:
                directory = base.INPUT_ROOT / "nvfp4-api"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{uuid.uuid4().hex}.png"
                image.save(path)
                image_name = f"nvfp4-api/{path.name}"
            if request.operation in ("outpaint", "object_remove", "background_cleanup", "background_remove"):
                graph = lora_workflow(request, image_name, image.size, seed, prefix)
            elif request.operation == "inpaint":
                graph = paint_workflow(request, image_name, image.size, seed, prefix)
            else:
                width, height = base.parse_size(request.size)
                generation_size(width, height)
                graph = base.workflow(request.prompt.strip(), width, height, seed, prefix,
                    [image_name] if image is not None else None)
            encoded = await base.execute_workflow(graph)
            if original_paint is not None:
                encoded = restore_masked_edit(encoded, original_paint, working_content_size)
            if output_size is not None:
                encoded = restore_output_size(encoded, output_size, working_content_size)
            if request.operation == "background_remove":
                encoded = await cutout(base64.b64decode(encoded))
            if request.operation == "outpaint" and request.preserve_source:
                encoded = preserve_original(encoded, original_canvas, request)
        except (httpx.HTTPError, KeyError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(500, str(exc)) from exc
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
    return {"created": int(time.time()), "seed": seed, "data": [{"b64_json": encoded}]}


@app.get("/v1/runtime/memory")
async def runtime_memory():
    async with httpx.AsyncClient(timeout=3) as client:
        response = await client.get(f"{base.COMFY_URL}/sparktalk/memory")
        response.raise_for_status()
        state = response.json()
        state['busy'] = bool(state.get('busy')) or base.generation_lock.locked()
        return state


async def runtime_control(action):
    if base.generation_lock.locked():
        raise HTTPException(409, 'Image generation is active')
    async with base.generation_lock:
        async with httpx.AsyncClient(timeout=180) as client:
            if action == 'prepare':
                response = await client.get(f'{base.COMFY_URL}/sparktalk/memory')
                response.raise_for_status()
                state = response.json()
                # A resident LoRA view also owns the same core weights. Do not
                # switch it back to the canonical patcher on every retry.
                if state.get('core_ready'):
                    return state
            payload = {'unet_name': base.DIFFUSION_MODEL, 'vae_name': base.VAE} if action == 'prepare' else {}
            response = await client.post(f'{base.COMFY_URL}/sparktalk/{action}', json=payload)
            if not response.is_success:
                raise HTTPException(response.status_code, response.text[-2000:])
            state = response.json()
            if action == 'prepare' and not state.get('core_ready'):
                graph = base.workflow('warmup', 256, 256, 0, f'nvfp4-api/warmup-{uuid.uuid4().hex}')
                graph.pop('2', None)
                graph['4'] = {'class_type': 'SparkTalkWarmupConditioning', 'inputs': {'model': ['1', 0]}}
                graph['8']['inputs']['steps'] = 1
                # No text encoder or LoRA; output is discarded and the common
                # executor removes the temporary PNG after reading it.
                await base.execute_workflow(graph)
                response = await client.get(f'{base.COMFY_URL}/sparktalk/memory')
                response.raise_for_status()
                state = response.json()
                if not state.get('core_ready'):
                    raise HTTPException(503, 'Core residency not confirmed after warmup')
            return state


@app.post('/v1/runtime/prepare')
async def runtime_prepare():
    return await runtime_control('prepare')


@app.post('/v1/runtime/reclaim')
async def runtime_reclaim():
    return await runtime_control('reclaim')


@app.post('/v1/runtime/cancel')
async def runtime_cancel():
    # Interrupt first, without waiting on the generation lock. The Comfy queue
    # and API request must both finish before the caller may change weights.
    async with httpx.AsyncClient(timeout=10) as client:
        owned_prompt = base.active_prompt_id
        worker = _cutout_worker
        if owned_prompt is not None:
            response = await client.get(f'{base.COMFY_URL}/queue')
            response.raise_for_status()
            running = response.json().get('queue_running', [])
            if any(len(entry) > 1 and entry[1] == owned_prompt for entry in running):
                response = await client.post(f'{base.COMFY_URL}/interrupt')
                response.raise_for_status()
            # Delete only this API request's queued prompt, never another
            # client's pending work. Its wait loop also needs cancellation.
            response = await client.post(f'{base.COMFY_URL}/queue', json={'delete': [owned_prompt]})
            response.raise_for_status()
            base.cancelled_prompts.add(owned_prompt)
        if worker is not None and worker.returncode is None:
            try:
                worker.kill()
            except ProcessLookupError:
                pass
            await worker.wait()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            response = await client.get(f'{base.COMFY_URL}/queue')
            response.raise_for_status()
            running = response.json().get('queue_running', [])
            owned_running = owned_prompt is not None and any(len(entry) > 1 and entry[1] == owned_prompt for entry in running)
            if not owned_running and not base.generation_lock.locked():
                return {'status': 'ok'}
            await asyncio.sleep(.1)
    raise HTTPException(409, 'Image cancellation is not confirmed; core retained')
