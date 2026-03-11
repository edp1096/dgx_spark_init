"""Negative prompt support for distilled pipelines via denoising-level guidance.

Provides CFG-like guidance for distilled models (IC-LoRA, Distilled, Retake-distilled)
by monkey-patching simple_denoising_func to run the transformer twice per step
(positive + negative context) and combining the results.

Usage in worker.py:
    from mod.nag import encode_negative_prompt, nag_guidance

    neg_v, neg_a = encode_negative_prompt(pipeline.model_ledger, neg_prompt)
    with nag_guidance(neg_v, neg_a, scale=5.0):
        video_frames, audio = pipeline(**gen_kwargs)
"""

import logging
import sys
import torch
from contextlib import contextmanager

logger = logging.getLogger("ltx2-worker")

# Modules that import simple_denoising_func directly
_PATCH_MODULES = [
    "ltx_pipelines.ic_lora",
    "ltx_pipelines.distilled",
    "ltx_pipelines.retake",
]


def get_model_ledger(pipeline):
    """Extract model_ledger from any pipeline type."""
    for attr in ("model_ledger", "stage_1_model_ledger"):
        if hasattr(pipeline, attr):
            return getattr(pipeline, attr)
    raise AttributeError(f"Cannot find model_ledger on {type(pipeline).__name__}")


def encode_negative_prompt(model_ledger, negative_prompt: str):
    """Encode negative prompt text. Loads/frees text encoder within this call.

    Returns (neg_video_context, neg_audio_context) tensors.
    """
    from ltx_pipelines.utils.helpers import encode_prompts

    (ctx_n,) = encode_prompts(
        [negative_prompt],
        model_ledger,
        enhance_first_prompt=False,
    )
    return ctx_n.video_encoding, ctx_n.audio_encoding


def _make_guided_denoising_func(original_factory, neg_video_ctx, neg_audio_ctx, scale):
    """Wrap simple_denoising_func to add CFG-style guidance."""
    from ltx_pipelines.utils.helpers import modality_from_latent_state

    def guided_factory(video_context, audio_context, transformer):
        original_step = original_factory(video_context, audio_context, transformer)

        def guided_step(video_state, audio_state, sigmas, step_index):
            # Positive pass
            pos_video, pos_audio = original_step(video_state, audio_state, sigmas, step_index)

            # Negative pass
            sigma = sigmas[step_index]
            neg_video_mod = modality_from_latent_state(video_state, neg_video_ctx, sigma)
            neg_audio_mod = modality_from_latent_state(audio_state, neg_audio_ctx, sigma)
            neg_video, neg_audio = transformer(
                video=neg_video_mod, audio=neg_audio_mod, perturbations=None
            )

            # CFG: uncond + scale * (cond - uncond)
            guided_video = neg_video + scale * (pos_video - neg_video)
            guided_audio = neg_audio + scale * (pos_audio - neg_audio)

            return guided_video, guided_audio

        return guided_step

    return guided_factory


@contextmanager
def nag_guidance(neg_video_ctx, neg_audio_ctx, scale=5.0):
    """Context manager: patches simple_denoising_func for guidance on distilled models.

    Args:
        neg_video_ctx: Negative video context from encode_negative_prompt.
        neg_audio_ctx: Negative audio context from encode_negative_prompt.
        scale: Guidance scale (1.0 = no effect, higher = stronger). Default 5.0.
    """
    if scale <= 1.0:
        yield
        return

    from ltx_pipelines.utils import helpers as _helpers

    original = _helpers.simple_denoising_func
    patched = _make_guided_denoising_func(original, neg_video_ctx, neg_audio_ctx, scale)

    # Patch in helpers module and all modules that imported the function directly
    patched_modules = [_helpers]
    for mod_name in _PATCH_MODULES:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "simple_denoising_func"):
            patched_modules.append(mod)

    for mod in patched_modules:
        setattr(mod, "simple_denoising_func", patched)

    logger.info("NAG guidance enabled (scale=%.1f)", scale)

    try:
        yield
    finally:
        for mod in patched_modules:
            setattr(mod, "simple_denoising_func", original)
        logger.info("NAG guidance restored")
