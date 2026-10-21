"""Fail before loading weights if multimodal preprocessing silently falls back."""
import sys
from PIL import Image
from sglang.srt.utils.hf_transformers_utils import get_processor

processor = get_processor(sys.argv[1])
if type(processor).__name__ != 'Glm5NextProcessor':
    raise RuntimeError(f'Expected Glm5NextProcessor, got {type(processor).__name__}')
result = processor(
    text=['<|begin_of_image|><|image|><|end_of_image|>'],
    images=[Image.new('RGB', (256, 256), (255, 0, 0))],
    return_tensors='pt',
)
if result['input_ids'].shape[1] <= 3 or result['pixel_values'].numel() == 0:
    raise RuntimeError('GLM image processor did not expand image tokens')
print('GLM image processor verified:', result['input_ids'].shape[1], 'tokens', flush=True)
