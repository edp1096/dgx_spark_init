"""Masked image editing primitives for the few-step SANA-Sprint pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from diffusers import SanaSprintImg2ImgPipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class OutpaintCanvas:
    image: Image.Image
    mask: Image.Image


def build_outpaint_canvas(
    source: Image.Image,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    overlap: int,
    feather: int,
) -> OutpaintCanvas:
    """Place the source on a larger canvas and make a white generation mask."""
    source = source.convert("RGB")
    if min(left, right, top, bottom, overlap, feather) < 0:
        raise ValueError("outpaint margins, overlap, and feather must not be negative")
    horizontal_overlap = overlap * int(left > 0) + overlap * int(right > 0)
    vertical_overlap = overlap * int(top > 0) + overlap * int(bottom > 0)
    if horizontal_overlap >= source.width or vertical_overlap >= source.height:
        raise ValueError("overlap must leave a non-empty protected source region")
    width = source.width + left + right
    height = source.height + top + bottom
    canvas = Image.new("RGB", (width, height), (127, 127, 127))
    canvas.paste(source, (left, top))

    mask = Image.new("L", (width, height), 255)
    keep_left = left + (overlap if left else 0)
    keep_top = top + (overlap if top else 0)
    keep_right = left + source.width - (overlap if right else 0)
    keep_bottom = top + source.height - (overlap if bottom else 0)
    if keep_right > keep_left and keep_bottom > keep_top:
        mask.paste(0, (keep_left, keep_top, keep_right, keep_bottom))
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return OutpaintCanvas(canvas, mask)


class SprintMaskedEditor:
    """Training-free latent blending on top of SanaSprintImg2ImgPipeline.

    White mask pixels are generated. Black mask pixels are restored from the
    source latent at every scheduler timestep and from source pixels at output.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = SanaSprintImg2ImgPipeline(**pipeline.components)

    def edit(
        self,
        *,
        source: Image.Image,
        mask: Image.Image,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
    ) -> Image.Image:
        pipe = self.pipeline
        device = pipe._execution_device
        source = source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        mask = mask.convert("L").resize((width, height), Image.Resampling.BILINEAR)

        init_image = pipe.prepare_image(source, width, height, device, pipe.vae.dtype)
        with torch.inference_mode():
            source_latents = pipe.vae.encode(init_image).latent
            source_latents = source_latents * pipe.vae.config.scaling_factor * pipe.scheduler.config.sigma_data

        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = randn_tensor(source_latents.shape, generator=generator, device=device, dtype=torch.float32)
        noise = noise * pipe.scheduler.config.sigma_data

        pipe.scheduler.set_timesteps(
            steps,
            device=device,
            max_timesteps=1.5708,
            intermediate_timesteps=1.3,
        )
        first_timestep = pipe.scheduler.timesteps[0]
        initial_latents = torch.cos(first_timestep) * source_latents.float() + torch.sin(first_timestep) * noise

        mask_values = np.asarray(mask, dtype=np.float32) / 255.0
        latent_mask = torch.from_numpy(mask_values).to(device=device, dtype=torch.float32)[None, None]
        latent_mask = functional.interpolate(
            latent_mask,
            size=source_latents.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        def preserve_known_region(current_pipe, step_index, _timestep, callback_kwargs):
            next_index = min(step_index + 1, len(current_pipe.scheduler.timesteps) - 1)
            next_timestep = current_pipe.scheduler.timesteps[next_index]
            known_latents = (
                torch.cos(next_timestep) * source_latents.float()
                + torch.sin(next_timestep) * noise
            )
            generated = callback_kwargs["latents"]
            callback_kwargs["latents"] = generated * latent_mask + known_latents * (1.0 - latent_mask)
            return callback_kwargs

        with torch.inference_mode():
            generated = pipe(
                prompt=prompt,
                image=source,
                width=width,
                height=height,
                strength=1.0,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
                latents=initial_latents,
                use_resolution_binning=False,
                callback_on_step_end=preserve_known_region,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]

        # Latent blending protects structure; this final composite makes the
        # fully unmasked source pixels exact while retaining a feathered seam.
        return Image.composite(generated.convert("RGB"), source, mask)
