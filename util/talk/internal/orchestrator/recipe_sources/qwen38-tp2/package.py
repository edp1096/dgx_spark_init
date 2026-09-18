#!/usr/bin/env python3
"""Pack the canonical TP2 runtime and portable Talk wrappers reproducibly."""
from pathlib import Path
import gzip,io,tarfile
here=Path(__file__).resolve().parent
root=next(p for p in here.parents if (p/'compose_yaml/qwen38_fn_sglang').is_dir())
source=root/'compose_yaml/qwen38_fn_sglang'
files={name:source/name for name in ['ensure_rail.py','manage_tp2.py','compose.tp2.yaml','tp2/entrypoint.py','tp2/watchdog.py']}
files.update({name:here/name for name in ['manage.sh','models.sh','runtime.sh','env.sample']})
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
 for name,path in sorted(files.items()):
  data=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o755 if name.endswith('.sh') else 0o644;tar.addfile(info,io.BytesIO(data))
(here.parent.parent/'assets/recipes/qwen38-tp2.tar.gz').write_bytes(gzip.compress(buf.getvalue(),mtime=0))
