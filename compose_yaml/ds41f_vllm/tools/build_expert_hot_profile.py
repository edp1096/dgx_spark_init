"""Build a version-bound frequency seed from a separate historic routing trace."""
import argparse,collections,hashlib,json,struct
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trace',type=Path,required=True)
p.add_argument('--packed-header',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();raw=a.trace.read_bytes();counts=collections.defaultdict(collections.Counter)
for line in raw.splitlines():
 row=json.loads(line)
 if row['layer']<40 and row['tokens']<=16:counts[row['layer']].update(row['needed'])
with a.packed_header.open('rb') as f:
 size=struct.unpack('<Q',f.read(8))[0]
 assert 0<size<65528
 meta=json.loads(f.read(size))
assert set(counts)==set(range(40))
result={'schema':1,'revision':meta['revision'],'format':meta['format'],'training_sha256':hashlib.sha256(raw).hexdigest(),
        'layers':{str(l):[e for e,_ in counts[l].most_common()] for l in range(40)}}
a.output.write_text(json.dumps(result,separators=(',',':'))+'\n')
