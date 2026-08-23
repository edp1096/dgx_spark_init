# VideoX-Fun model classes adapted for single-GPU ZIT deployment.
# Source: https://github.com/ali-vilab/VideoX-Fun (videox_fun/)
#
# Changes from original:
#   - Removed: videox_fun.dist (multi-GPU sequence parallel, ZMultiGPUsSingleStreamAttnProcessor)
#   - Removed: xfuser dependencies
#   - All multi-GPU code paths (sp_world_size > 1, all_gather) replaced with single-GPU no-ops
#   - Import paths rewritten from `videox_fun.models.*` / `..models.*` to local relative imports
#   - AutoencoderKL imported directly from diffusers (not from videox_fun.models)

from .z_image_transformer2d import (
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
    ZSingleStreamAttnProcessor,
    FinalLayer,
    FeedForward,
    TimestepEmbedder,
    RopeEmbedder,
)

from .z_image_transformer2d_control import (
    ZImageControlTransformer2DModel,
    ZImageControlTransformerBlock,
    BaseZImageTransformerBlock,
)

from .pipeline_z_image_control import (
    ZImageControlPipeline,
    ZImagePipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)

from .fp8_optimization import (
    replace_parameters_by_name,
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    undo_convert_weight_dtype_wrapper,
)

from .attention_utils import (
    attention,
    flash_attention,
    flash_attention_naive,
    SparseLinearAttention,
)

__all__ = [
    # Transformer models
    "ZImageTransformer2DModel",
    "ZImageTransformerBlock",
    "ZSingleStreamAttnProcessor",
    "FinalLayer",
    "FeedForward",
    "TimestepEmbedder",
    "RopeEmbedder",
    # ControlNet
    "ZImageControlTransformer2DModel",
    "ZImageControlTransformerBlock",
    "BaseZImageTransformerBlock",
    # Pipeline
    "ZImageControlPipeline",
    "ZImagePipelineOutput",
    "calculate_shift",
    "retrieve_timesteps",
    # FP8
    "replace_parameters_by_name",
    "convert_model_weight_to_float8",
    "convert_weight_dtype_wrapper",
    "undo_convert_weight_dtype_wrapper",
    # Attention
    "attention",
    "flash_attention",
    "flash_attention_naive",
    "SparseLinearAttention",
]
