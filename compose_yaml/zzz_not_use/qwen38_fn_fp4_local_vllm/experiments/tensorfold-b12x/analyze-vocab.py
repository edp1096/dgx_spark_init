import json,pathlib
from tokenizers import Tokenizer
r=pathlib.Path('/bridge');t=Tokenizer.from_file('/model/tokenizer.json')
stock=set(map(int,(r/'draft-vocab-stock.txt').read_text().split()))
vocab=t.get_vocab_size();texts=t.decode_batch([[i] for i in range(vocab)],skip_special_tokens=False)
def hangul(s):return any(0xac00<=ord(c)<=0xd7a3 or 0x1100<=ord(c)<=0x11ff or 0x3130<=ord(c)<=0x318f for c in s)
def han(s):return any(0x3400<=ord(c)<=0x9fff or 0x20000<=ord(c)<=0x323af or 0xf900<=ord(c)<=0xfaff for c in s)
ko={i for i,s in enumerate(texts) if hangul(s)};zh={i for i,s in enumerate(texts) if han(s)}
x={'vocab':vocab,'stock':len(stock),'hangul_total':len(ko),'hangul_in_stock':len(ko&stock),'han_total':len(zh),'han_in_stock':len(zh&stock),'korean_union_size':len(stock|ko),'multilingual_union_size':len(stock|ko|zh)}
(r/'vocab-analysis.json').write_text(json.dumps(x,indent=2));(r/'draft-vocab-korean.txt').write_text('\n'.join(map(str,sorted(stock|ko)))+'\n');print(json.dumps(x))
