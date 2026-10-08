"""Create the CUDA context before importing the large Comfy/PyTorch stack.

On GB10 the resident QAD engine can leave sufficient MemAvailable but few
immediate free pages after library loading. Retain the same device primary
context that PyTorch uses, then run the unchanged Comfy entrypoint.
"""
import ctypes
import runpy
import sys

cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
cuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
cuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
cuda.cuCtxSetCurrent.restype = ctypes.c_int


def checked(name, *args):
    code = getattr(cuda, name)(*args)
    if code:
        detail = "out of memory" if code == 2 else f"driver status {code}"
        raise RuntimeError(f"CUDA error: {detail} ({name})")


checked("cuInit", 0)
primary = ctypes.c_void_p()
checked("cuDevicePrimaryCtxRetain", ctypes.byref(primary), 0)
checked("cuCtxSetCurrent", primary)
print("[sparktalk] CUDA primary context prepared before Comfy imports", flush=True)

main = "/opt/ComfyUI/main.py"
sys.path.insert(0, "/opt/ComfyUI")
sys.argv[0] = main
runpy.run_path(main, run_name="__main__")
