
# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import torch
from streaming_moe import dequant
# Independent scalar decoder checks both nibble ordering and signed magnitudes.
raw = bytes(range(256))
scale = bytes([120, 127, 130, 125, 127, 127, 128, 129, 120, 127, 130, 125, 127, 127, 128, 129])
expected = []
for i, packed in enumerate(raw):
    for code in (packed % 16, packed // 16):
        exponent, mantissa = (code % 8)//2, code % 2
        magnitude = mantissa*.5 if exponent == 0 else (1+mantissa*.5)*2**(exponent-1)
        expected.append((-1 if code >= 8 else 1)*magnitude*2**(scale[(i*2)//32]-127))
actual = dequant((raw,(1,256)),(scale,(1,16)),torch.device('cpu'))
assert torch.equal(actual, torch.tensor(expected,dtype=torch.bfloat16).reshape(1,512))
print('PASS: all 16 E2M1 codes, nibble ordering, and varying E8M0 blocks')
