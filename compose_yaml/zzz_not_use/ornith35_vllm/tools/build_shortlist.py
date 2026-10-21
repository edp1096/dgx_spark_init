"""Build a tokenizer-specific draft-only vocabulary; never alter target weights."""
import argparse,collections,hashlib,json,pathlib
from tokenizers import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=65536);a=p.parse_args()
raw=pathlib.Path(a.tokenizer).read_bytes();data=json.loads(raw);tok=Tokenizer.from_file(a.tokenizer)
valid=set(tok.get_vocab().values());protected={t['id'] for t in data.get('added_tokens',[])}
# Preserve byte fallback tokens and single-byte BPE tokens, plus full Hangul/Jamo.
for token,i in tok.get_vocab().items():
 text=tok.decode([i],skip_special_tokens=False)
 if i<256 or (token.startswith('<0x') and token.endswith('>')) or any('\uac00'<=c<='\ud7a3' or '\u1100'<=c<='\u11ff' or '\u3130'<=c<='\u318f' for c in text):
  protected.add(i)
freq=collections.Counter()
for row in json.loads(pathlib.Path(a.corpus).read_text())['rows']:
 for field in ['prompt','content','reasoning']:
  freq.update(tok.encode(row.get(field,'') or '',add_special_tokens=False).ids)
selected=set(protected)
if not len(protected)<=a.size<=len(valid):raise ValueError('size cannot preserve mandatory tokens')
for i in sorted(freq,key=lambda i:(-freq[i],i))+sorted(valid):
 if len(selected)>=a.size:break
 selected.add(i)
out=dict(version=1,ids=sorted(selected),size=len(selected),tokenizer_sha256=hashlib.sha256(raw).hexdigest(),corpus_sha256=hashlib.sha256(pathlib.Path(a.corpus).read_bytes()).hexdigest(),protected_tokens=len(protected),calibration_coverage=sum(v for k,v in freq.items() if k in selected)/sum(freq.values()),policy='special + byte fallback + Hangul/Jamo + independent corpus frequency + ascending IDs')
pathlib.Path(a.out).write_text(json.dumps(out,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='ids'},ensure_ascii=False))
