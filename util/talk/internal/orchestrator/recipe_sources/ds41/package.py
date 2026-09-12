#!/usr/bin/env python3
"""Pack the standalone tested runtime plus Talk's portable lifecycle adapter."""
from pathlib import Path
import gzip, io, tarfile
here = Path(__file__).resolve().parent
root = next(p for p in here.parents if (p / 'compose_yaml/ds41f_vllm').is_dir())
source = root / 'compose_yaml/ds41f_vllm'
files = {p.name: p for p in here.glob('*') if p.name != 'package.py' and p.is_file()}
files.update({p.name: p for p in source.glob('*.py')})
files['launch.sh'] = source / 'launch.sh'
files['expert-hot-profile.json'] = source / 'expert-hot-profile.json'
files.update({str(p.relative_to(source)): p for p in (source/'patches').rglob('*') if p.is_file() and '__pycache__' not in p.parts})
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w') as tar:
 for name, path in sorted(files.items()):
  data = path.read_bytes()
  info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o755 if name.endswith('.sh') else 0o644
  tar.addfile(info, io.BytesIO(data))
(here.parent.parent/'assets/recipes/ds41.tar.gz').write_bytes(gzip.compress(buf.getvalue(), mtime=0))
