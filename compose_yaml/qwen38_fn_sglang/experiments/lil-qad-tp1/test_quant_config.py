"""Exercise the actual file-based quantization loader, not only from_config."""
from types import SimpleNamespace
from sglang.srt.model_loader.weight_utils import get_quant_config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig

model = SimpleNamespace(
    model_path='/metadata', quantization='modelopt_mixed', revision=None,
    hf_config=SimpleNamespace(architectures=['Qwen4ExpForConditionalGeneration']),
    is_draft_model=False, is_draft_quantization_explicit=False)
config = get_quant_config(model, SimpleNamespace(download_dir=None), {})
assert isinstance(config, ModelOptMixedPrecisionConfig), type(config)
print('Actual file-based loader accepts the pinned ModelOpt config_groups schema')
