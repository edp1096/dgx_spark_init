from pathlib import Path
import json,statistics
r=Path(__file__).parent
def summarize(path):
    rows=json.loads(path.read_text());out={'requests':len(rows),'replacement_characters':sum(x['replacement_characters'] for x in rows)}
    for family in ['ko','code']:
        group=[x for x in rows if x['name'].startswith('speed_'+family+'_')]
        if not group:continue
        m=[x['meta_info'] for x in group]
        num=sum(x.get('spec_num_correct_drafts',0) for x in m)
        den=sum(x.get('spec_num_proposed_drafts',0) for x in m)
        out[family]={'tg_median':statistics.median(x['tg_tok_s'] for x in group),
                     'tg_values':[x['tg_tok_s'] for x in group],
                     'strict_accepted':num,'strict_proposed':den,
                     'acceptance':num/den if den else None,
                     'completion_tokens':[x['completion_tokens'] for x in group]}
    out['pp']={x['name']:x.get('pp_engine_tok_s') for x in rows if x['name'].startswith('recall') or x['name'].startswith('long_')}
    out['long']={x['name']:{k:x.get(k) for k in ['prompt_tokens','completion_tokens','pp_engine_tok_s','tg_tok_s','ttft_s','replacement_characters']} for x in rows if x['name'].startswith('long_')}
    return out
out={p.parent.name:summarize(p) for p in sorted(r.glob('*/results.json'))}
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for arm,s in out.items():
    print(arm,s['requests'],*(f'{k}={s[k]["tg_median"]:.2f} accept={s[k]["acceptance"]}' for k in ['ko','code'] if k in s))
