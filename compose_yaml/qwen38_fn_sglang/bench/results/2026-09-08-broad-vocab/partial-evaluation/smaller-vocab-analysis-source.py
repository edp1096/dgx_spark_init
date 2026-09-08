import sys,json,collections
sys.path.insert(0,'/bench')
from build_vocab import byte_decoder
from tokenizers import Tokenizer
data=json.load(open('/tokenizer.json')); t=Tokenizer.from_file('/tokenizer.json'); inv=byte_decoder(); protected=set(range(256))|{x['id'] for x in data['added_tokens']}
for text,ident in data['model']['vocab'].items():
 try: decoded=bytes(inv[c] for c in text).decode('utf-8')
 except (KeyError,UnicodeDecodeError):continue
 if any('\uac00'<=c<='\ud7a3' for c in decoded):protected.add(ident)
freq=collections.Counter()
for row in json.load(open('/bench/results/2026-09-06/corpus.json'))['rows']:
 for field in ['prompt','content']:freq.update(t.encode(row[field],add_special_tokens=False).ids)
valid=set(t.get_vocab().values());order=sorted(freq,key=lambda i:(-freq[i],i))+sorted(valid);out={}
for n in [16384,32768,65536]:
 chosen=set(protected)
 assert len(chosen)<=n
 for i in order:
  if len(chosen)>=n:break
  chosen.add(i)
 out[str(n)]=sorted(chosen)
print(json.dumps(dict(protected=len(protected),sets=out)))
