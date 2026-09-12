"""Low-frequency, opt-in startup allocation diagnostics; no tensor contents."""
import json,os,time
from pathlib import Path
ENABLED=bool(os.environ.get('DSV41_PROBE_TOKEN'))
def note(phase,**fields):
 if not ENABLED:return
 import torch
 available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 print('STARTUP_PROBE '+json.dumps({'time_ns':time.time_ns(),'phase':phase,'available_bytes':available,'cuda_allocated':torch.cuda.memory_allocated(),'cuda_reserved':torch.cuda.memory_reserved(),**fields}),flush=True)
