"""FLUX generation/editing, LanPaint inpainting and fal outpainting LoRA."""
import asyncio
import base64
import binascii
import io
import time
import uuid
from typing import Literal

import httpx
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageOps, ImageChops, ImageFilter, UnidentifiedImageError
from pydantic import ConfigDict, Field

import base_api as base

app = base.app
app.router.routes = [route for route in app.router.routes if getattr(route, "path", "") not in ("/v1/images/generations", "/health")]


class PaintRequest(base.ImageRequest):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["generate", "identity_edit", "inpaint", "outpaint", "object_remove", "background_cleanup", "background_remove"] = "generate"
    output_format: Literal["png"] = "png"
    preserve_source: bool = False
    background_method: Literal["rembg", "lora_rembg"] = "rembg"
    source_image: str | None = Field(default=None, max_length=32 * 1024 * 1024)
    anypaint_image: str | None = Field(default=None, max_length=32 * 1024 * 1024)
    anypaint_mask: str | None = Field(default=None, max_length=32 * 1024 * 1024)
    mask_box: list[int] | None = None
    outpaint_left: int = Field(default=0, ge=0, le=512, multiple_of=16)
    outpaint_top: int = Field(default=0, ge=0, le=512, multiple_of=16)
    outpaint_right: int = Field(default=0, ge=0, le=512, multiple_of=16)
    outpaint_bottom: int = Field(default=0, ge=0, le=512, multiple_of=16)


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


def trial_size(width, height):
    if any(n < 256 or n > 1024 or n % 16 for n in (width, height)):
        raise HTTPException(400, "trial output dimensions must be 256..1024 multiples of 16; use a smaller source before extending a 1024 image")


def prepare_paint(request):
    source = decode_image(request.anypaint_image)
    trial_size(*source.size)
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
        trial_size(*size)
        canvas = Image.new("RGB", size, (0, 255, 0))
        canvas.paste(source, (left, top))
        mask = Image.new("L", size, 255)
        mask.paste(0, (left, top, left + source.width, top + source.height))
        source = canvas
    # ComfyUI LoadImage treats transparent pixels as the edit mask.
    rgba = source.convert("RGBA")
    rgba.putalpha(ImageOps.invert(mask))
    return rgba


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


async def cutout(data):
    worker = await asyncio.create_subprocess_exec(
        "/opt/rembg-venv/bin/python", "/opt/nvfp4-api/rembg_worker.py",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, error = await asyncio.wait_for(worker.communicate(data), timeout=120)
        if worker.returncode != 0:
            raise RuntimeError("rembg failed: " + error.decode(errors="replace")[-1200:])
        if not output.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("rembg did not return a PNG image")
        return base64.b64encode(output).decode("ascii")
    finally:
        if worker.returncode is None:
            worker.kill()
            await worker.wait()


def preserve_original(encoded, canvas, request):
    # Feather only INSIDE the original region. Green padding must never leak
    # through at the canvas edge or into the generated extension.
    mask = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(mask)
    left, top = request.outpaint_left, request.outpaint_top
    right, bottom = canvas.width - request.outpaint_right - 1, canvas.height - request.outpaint_bottom - 1
    for inset in range(16):
        box = (left + (inset if left else 0), top + (inset if top else 0),
               right - (inset if request.outpaint_right else 0), bottom - (inset if request.outpaint_bottom else 0))
        draw.rectangle(box, fill=round(255 * (inset + 1) / 16))
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
            "outpaint_lora": "fal/flux-2-klein-4B-outpaint-lora", "outpaint_lora_strength": 1.1, "remove_loras": ["object-remove", "background-remove"], "background_cutout": "rembg u2net CPU"}


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
    if request.operation not in ("inpaint", "outpaint", "object_remove") and request.model_fields_set & paint_fields:
        raise HTTPException(400, "mask and padding fields require inpaint or outpaint")
    if request.operation not in ("identity_edit", "background_cleanup", "background_remove") and request.source_image is not None:
        raise HTTPException(400, "source_image requires identity_edit")
    if request.preserve_source and request.operation != "outpaint":
        raise HTTPException(400, "preserve_source is only supported for outpaint")
    if request.background_method != "rembg" and request.operation != "background_remove":
        raise HTTPException(400, "background_method requires background_remove")
    seed = base.request_seed(request.seed)
    prefix = f"nvfp4-api/{uuid.uuid4().hex}"
    path = None
    async with base.generation_lock:
        try:
            if request.operation in ("inpaint", "outpaint", "object_remove"):
                image = prepare_paint(request)
                if request.operation == "outpaint":
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
                trial_size(*image.size)
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
                trial_size(width, height)
                graph = base.workflow(request.prompt.strip(), width, height, seed, prefix,
                    [image_name] if image is not None else None)
            encoded = await base.execute_workflow(graph)
            if request.operation == "background_remove":
                encoded = await cutout(base64.b64decode(encoded))
            if request.operation == "outpaint" and request.preserve_source:
                encoded = preserve_original(encoded, image, request)
        except (httpx.HTTPError, KeyError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(500, str(exc)) from exc
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
    return {"created": int(time.time()), "seed": seed, "data": [{"b64_json": encoded}]}
