"""Short-lived CPU worker; exits to release ONNX and image buffers after a job."""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("U2NET_HOME", "/root/.cache/huggingface/rembg")
from rembg import new_session, remove

session = new_session("u2net", providers=["CPUExecutionProvider"])
if "--prepare" in sys.argv:
    print("rembg u2net CPU model ready", flush=True)
else:
    data = sys.stdin.buffer.read(32 * 1024 * 1024 + 1)
    if not data or len(data) > 32 * 1024 * 1024:
        raise ValueError("input image must be smaller than 32 MiB")
    sys.stdout.buffer.write(remove(data, session=session))
