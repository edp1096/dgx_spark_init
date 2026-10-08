"""Summarize measured throughput without treating short replies as long-form speed."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent;results=root/'results';rows=[]
for name,stem in [('Full vocabulary / 1024','baseline'),('ko64k / 1024','ko64k-1024'),('ko64k / 2048','ko64k-2048'),('ko64k / 4096','ko64k-4096'),('ko64k / 8192','ko64k-8192')]:
 suffix='-clean' if stem=='baseline' else ''
 sp=results/(stem+'-speed'+suffix+'.json');pp=results/(stem+'-prefill'+suffix+'.json')
 if not sp.exists() or not pp.exists():continue
 s=json.loads(sp.read_text());p=json.loads(pp.read_text())
 if not s.get('completed') or not p.get('completed'):continue
 rs=[r for r in s['rows'] if r['name'].startswith('ko_')]
 n=sum(r['usage']['completion_tokens']-1 for r in rs);duration=sum((r['usage']['completion_tokens']-1)/r['tg'] for r in rs)
 row={'variant':name,'korean_tg_weighted':n/duration,'context':s['server']['context_length'],'kv_capacity':s['server']['max_total_num_tokens'],'prefill':{}}
 for key in ['needle_1800','needle_7200','needle_14400']:
  matches=[r for r in p['rows'] if r['name']==key]
  t=statistics.median([r['ttft'] for r in matches]);tokens=statistics.median([r['usage']['prompt_tokens'] for r in matches])
  row['prefill'][key]={'tokens':tokens,'ttft':t,'effective_pp':tokens/t,'correct':all(r['text'].strip()=='KEY_'+key.split('_')[1]+'_'+str(r['repeat']) for r in matches)}
 row['eligible']=all(x['correct'] for x in row['prefill'].values())
 rows.append(row)
(results/'performance-summary.json').write_text(json.dumps(rows,indent=2))
print('| Variant | Korean tg | 25K TTFT | 101K TTFT | 202K TTFT |')
print('|---|---:|---:|---:|---:|')
for row in rows:
 ts=[row['prefill'][k]['ttft'] for k in ['needle_1800','needle_7200','needle_14400']]
 print('| '+row['variant']+(' [REJECTED]' if not row['eligible'] else '')+' | '+f"{row['korean_tg_weighted']:.2f}"+' | '+' | '.join(f'{t:.2f}s' for t in ts)+' |')
