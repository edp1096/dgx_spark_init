"""Materialize the reviewed JSON token ranges as SGLang's tensor-map format."""
import json
import pathlib
import sys

import torch

source, target = map(pathlib.Path, sys.argv[1:])
data = json.loads(source.read_text())
assert data["version"] == 1
ids = [i for start, end in data["ranges"] for i in range(start, end)]
assert len(ids) == data["size"] == 65536
assert ids == sorted(set(ids)) and min(ids) >= 0 and max(ids) < 248320
torch.save(ids, target)
assert torch.load(target, weights_only=True) == ids
print(f"Installed {len(ids)} draft token IDs; tokenizer SHA256 {data['tokenizer_sha256']}")
