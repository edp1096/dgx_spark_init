"""Run with the existing offline serving image and vendored ModelOpt on PYTHONPATH."""
import numpy as np
import torch
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from weights_core.quantization import fp8_encode, quantize_nvfp4

rng = np.random.default_rng(20260915)
for shape in ((1,16),(7,64),(32,128)):
    for magnitude in (0.001, 1, 100):
        tensor = torch.tensor(rng.normal(size=shape)*magnitude, dtype=torch.bfloat16)
        values = tensor.float().numpy()
        actual, scales, global_scale, _ = quantize_nvfp4(values)
        expected, expected_scales, expected_global = NVFP4QTensor.quantize(tensor,16)
        np.testing.assert_array_equal(actual, expected._quantized_data.numpy())
        np.testing.assert_array_equal(scales, expected_scales.view(torch.uint8).numpy())
        np.testing.assert_allclose(global_scale, expected_global.numpy(), rtol=1e-7)
for values in (rng.normal(size=10000).astype(np.float32)*100,
               np.array([.25,.75,1.25,1.75,2.5,3.5,5,448,-448],dtype=np.float32)):
    expected = torch.from_numpy(values.clip(-448,448)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    np.testing.assert_array_equal(fp8_encode(values), expected)
print('PASS: 9 NVFP4 tensors match ModelOpt packed weights/scales; 10009 FP8 values match PyTorch')
