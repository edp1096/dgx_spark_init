import os
import time

import torch
from diffusers import Flux2KleinPipeline


model = os.environ.get("MODEL", "black-forest-labs/FLUX.2-klein-4B")
pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("PHASE=IDLE", flush=True)
time.sleep(10)

for size in (1024, 2048):
    print(f"PHASE=GENERATE_{size}", flush=True)
    started = time.monotonic()
    image = pipe(
        prompt="A compact yellow AI workstation on a clean desk, professional product photography",
        height=size,
        width=size,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    image.save(f"/output/flux-{size}.png")
    print(f"PHASE=DONE_{size} SECONDS={time.monotonic() - started:.3f}", flush=True)
    time.sleep(10)

print("PHASE=COMPLETE", flush=True)
time.sleep(5)
