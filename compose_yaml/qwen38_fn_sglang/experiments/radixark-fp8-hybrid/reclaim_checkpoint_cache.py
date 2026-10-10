"""Return only unused checkpoint file pages, never the active /ple mmap table.

Run inside the trial container, whose /proc exposes all model workers. This is
not a global cache drop; refuse if any checkpoint is still mapped or open.
"""
import json
import os
from pathlib import Path
import sys


def reclaim(root):
    root=Path(root)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    files={str((root/name).resolve()) for name in set(index.values())}
    active=set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            for line in (process/'maps').read_text().splitlines():
                parts=line.split()
                if len(parts)>=6 and parts[-1].startswith('/'):active.add(parts[-1])
            for fd in (process/'fd').iterdir():
                try:active.add(str(fd.resolve(strict=True)))
                except (FileNotFoundError,ProcessLookupError):pass
        except (FileNotFoundError,ProcessLookupError):pass
    busy=files&active
    if busy:raise RuntimeError(f'Checkpoint files still in use: {sorted(busy)}')
    for filename in files:
        with open(filename,'rb') as handle:
            os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return dict(checkpoint_files=len(files),active_checkpoint_files=0,active_ple_untouched=True)


if __name__=='__main__':print(json.dumps(reclaim(sys.argv[1])))
